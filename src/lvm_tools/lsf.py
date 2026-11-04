"""Per-fibre, non-parametric line spread function (LSF) of the LVM spectrographs.

The DRP's `LSF` extension gives a Gaussian width per fibre and pixel. The real LSF is not
Gaussian: it is flat-topped, by an amount that depends on the fibre and the wavelength. This
module reads the per-fibre survey LSF measured by the LVM sky-subtraction code
([lvmsky](https://github.com/sdss/lvmsky)): for each physical fibre and spectrograph channel
(B, R, Z), a continuous density

    L(d, λ) = Σ_kj c_kj M_k(d) B_j(λ),

where d is the offset from the line centre in Angstrom, M_k are 11 cubic M-splines (cubic
B-splines normalised to unit area) on uniform knots over [-2.5, 2.5] Å, and B_j are 4 cubic
B-splines in wavelength per channel. The coefficients are non-negative and sum to one over k
at every wavelength, so L integrates to one. The profile is the LSF before integration over
detector pixels.

`SurveyLSF` evaluates the profiles and their moments. `SurveyLSF.mixture` approximates a
profile by a few Gaussians with free offsets, so that a model can convolve the LSF with a
Gaussian line and integrate it over pixels analytically. `FitData.lsf_mixture` returns those
mixtures for every spaxel of a dataset, with each mixture stretched so that its standard
deviation equals the DRP's LSF width for that spaxel.

The tables are downloaded once from a fixed lvmsky commit, checked against the SHA-256 sums
in lvmsky's manifest, and cached (see `fetch_survey_lsf`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

__all__ = [
    "LVMSKY_COMMIT",
    "LSFMixture",
    "SurveyLSF",
    "fetch_survey_lsf",
]

LVMSKY_COMMIT = "50d2cac730169973a6e0f00e0a3f2242ab9ce525"
"""lvmsky commit (branch `skydecomp-test-final`) the survey LSF tables are taken from."""

LVMSKY_LSF_URL = (
    f"https://raw.githubusercontent.com/sdss/lvmsky/{LVMSKY_COMMIT}/skysub/sky_decomp/data/lsf"
)
"""Where the tables are downloaded from."""

DATA_FILES = (
    "wavelength_knots.csv",
    "survey_lsf_B.csv",
    "survey_lsf_R.csv",
    "survey_lsf_Z.csv",
    "coverage.csv",
)
"""Files of the survey LSF, besides `manifest.json`, which lists their SHA-256 sums."""

CHANNELS = ("B", "R", "Z")
CHANNEL_UPPER = {"B": 5787.0, "R": 7454.0, "Z": np.inf}
"""Upper wavelength bound of each channel, in Angstrom (B starts at the bottom of the grid)."""

N_OFFSET = 11
OFFSET_HALF_WIDTH = 2.5
"""The M-splines cover offsets in [-OFFSET_HALF_WIDTH, OFFSET_HALF_WIDTH] Angstrom."""

_KNOTS = -OFFSET_HALF_WIDTH + 2 * OFFSET_HALF_WIDTH * np.arange(N_OFFSET + 4) / (N_OFFSET + 3)
_H = _KNOTS[1] - _KNOTS[0]
# Each M-spline is the density of a sum of four uniform variables of width _H, so its mean is the
# centre of its knots and its variance is 4 _H² / 12.
_M_MEAN = _KNOTS[:N_OFFSET] + 2 * _H
_M_VAR = _H**2 / 3


def _mspline_basis(offset: NDArray) -> NDArray:
    """The 11 M-splines at `offset` (Angstrom), shape `offset.shape + (11,)`."""
    offset = np.asarray(offset, dtype=float)
    out = np.empty(offset.shape + (N_OFFSET,))
    for k in range(N_OFFSET):
        b = BSpline.basis_element(_KNOTS[k : k + 5], extrapolate=False)(offset)
        out[..., k] = np.nan_to_num(b) / _H
    return out


def default_cache_dir() -> Path:
    """Cache directory for the survey LSF tables.

    `$LVM_TOOLS_CACHE/survey_lsf/<commit>` if `LVM_TOOLS_CACHE` is set, otherwise
    `~/.cache/lvm_tools/survey_lsf/<commit>`, where `<commit>` is the first 12 characters of
    `LVMSKY_COMMIT`.
    """
    root = os.environ.get("LVM_TOOLS_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "lvm_tools"
    return base / "survey_lsf" / LVMSKY_COMMIT[:12]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(dest)


def fetch_survey_lsf(cache_dir: str | Path | None = None, base_url: str = LVMSKY_LSF_URL) -> Path:
    """Download the survey LSF tables, if they aren't cached yet, and check them.

    Downloads `manifest.json` and the files in `DATA_FILES` from `base_url` into `cache_dir`
    (missing files only), and checks every file against the SHA-256 sums in the manifest.

    Args:
        cache_dir: Where to keep the files. Defaults to `default_cache_dir()`.
        base_url: Where to download them from. Defaults to the lvmsky commit `LVMSKY_COMMIT`.

    Returns:
        The directory holding the files, for `SurveyLSF.from_directory`.

    Raises:
        ValueError: If a file doesn't match its checksum (the file is removed, so the next
            call downloads it again).
    """
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "manifest.json"
    if not manifest.exists():
        _download(f"{base_url}/manifest.json", manifest)
    sums = json.loads(manifest.read_text())["files_sha256"]
    for name in DATA_FILES:
        path = cache / name
        if not path.exists():
            _download(f"{base_url}/{name}", path)
        if _sha256(path) != sums[name]:
            path.unlink()
            raise ValueError(f"{name} does not match its SHA-256 sum in the manifest")
    return cache


@dataclass(frozen=True)
class LSFMixture:
    """An LSF approximated by a weighted sum of Gaussians.

    For component k, `weights[..., k]` is its weight (the weights sum to one),
    `offsets[..., k]` its centre relative to the line centre and `widths[..., k]` its standard
    deviation, both in Angstrom. Convolving the mixture with a Gaussian line of width σ gives
    Gaussians of widths √(widths² + σ²) at the same offsets.

    Attributes:
        weights: Component weights, shape `(..., n_components)`.
        offsets: Component centres in Angstrom, shape `(..., n_components)`.
        widths: Component standard deviations in Angstrom, shape `(..., n_components)`.
        measured: Whether the shape comes from a measured LSF (`True`) or is the Gaussian
            fallback for a fibre without one (`False`), shape `(...)`.
    """

    weights: NDArray
    offsets: NDArray
    widths: NDArray
    measured: NDArray

    @property
    def n_components(self) -> int:
        """Number of Gaussians."""
        return self.weights.shape[-1]

    @property
    def mean(self) -> NDArray:
        """Mean offset of the mixture in Angstrom, shape `(...)`."""
        return np.sum(self.weights * self.offsets, axis=-1)

    @property
    def sigma(self) -> NDArray:
        """Standard deviation of the mixture in Angstrom, shape `(...)`."""
        m2 = np.sum(self.weights * (self.widths**2 + self.offsets**2), axis=-1)
        return np.sqrt(m2 - self.mean**2)

    def density(self, offset: ArrayLike) -> NDArray:
        """The mixture's density in 1/Angstrom at `offset` (Angstrom from the line centre).

        Returns:
            Shape `(...) + offset.shape`.
        """
        d = np.asarray(offset, dtype=float)
        nd = d.ndim
        w, m, s = (a.reshape(a.shape[:-1] + (1,) * nd + a.shape[-1:]) for a in self._arrays())
        z = (d[..., None] - m) / s
        return np.sum(w * np.exp(-0.5 * z**2) / (s * np.sqrt(2 * np.pi)), axis=-1)

    def _arrays(self):
        return self.weights, self.offsets, self.widths


class SurveyLSF:
    """The per-fibre survey LSF from lvmsky, for every fibre and channel.

    Measured by lvmsky on 555 bright-Moon exposures of DRP 1.3.2: R and Z from sky emission
    lines, B from solar absorption lines in moonlight. Each fibre and channel is measured
    separately and averaged over observing nights, so the profile is fixed per fibre; its width
    changes by a few percent between calibration epochs. Fibres without a measurement (e.g. the
    standard-star fibres) fall back to a Gaussian.

    Load it with `SurveyLSF.load()`, which downloads and caches the tables on first use.

    Example:
        ```python
        from lvm_tools.lsf import SurveyLSF

        lsf = SurveyLSF.load()
        d = np.linspace(-2.5, 2.5, 201)
        profile = lsf.density(fibre_id=[100, 101], wavelength=6562.8, offset=d)  # (2, 201)
        mean, sigma = lsf.moments([100, 101], 6562.8)
        mix = lsf.mixture([100, 101], 6562.8, n_components=3)
        ```
    """

    def __init__(
        self,
        coefficients: dict[str, NDArray],
        knots: dict[str, NDArray],
        measured: dict[str, NDArray],
    ):
        """Use `load` or `from_directory` instead.

        Args:
            coefficients: Per channel, shape `(n_fibre_ids, 11, 4)`, indexed by fibre ID.
                Rows of fibres without a measurement are ignored.
            knots: Per channel, the wavelength B-spline knots in Angstrom.
            measured: Per channel, shape `(n_fibre_ids,)`: which fibre IDs have a measured LSF.
        """
        self.coefficients = coefficients
        self.knots = knots
        self.measured = measured

    @classmethod
    def load(
        cls, path: str | Path | None = None, *, cache_dir: str | Path | None = None
    ) -> SurveyLSF:
        """Load the survey LSF, downloading it first if needed.

        Args:
            path: A directory holding the tables (as in lvmsky's `skysub/sky_decomp/data/lsf`).
                If `None`, they are downloaded with `fetch_survey_lsf` (or taken from its cache).
            cache_dir: Cache directory for the download, when `path` is `None`.

        Returns:
            The survey LSF.
        """
        directory = Path(path) if path is not None else fetch_survey_lsf(cache_dir)
        return cls.from_directory(directory)

    @classmethod
    def from_directory(cls, directory: str | Path) -> SurveyLSF:
        """Read the tables from a directory, without downloading or checking them."""
        directory = Path(directory)
        knots = {c: [] for c in CHANNELS}
        with open(directory / "wavelength_knots.csv") as f:
            for row in csv.DictReader(f):
                knots[row["channel"]].append(float(row["wavelength_angstrom"]))
        coefficients, measured = {}, {}
        for c in CHANNELS:
            with open(directory / f"survey_lsf_{c}.csv") as f:
                rows = list(csv.DictReader(f))
            n = max(int(r["fiberid"]) for r in rows) + 1
            coef = np.full((n, N_OFFSET, 4), np.nan)
            ok = np.zeros(n, dtype=bool)
            names = [[f"coef_{k:02d}_{j:02d}" for j in range(4)] for k in range(N_OFFSET)]
            for r in rows:
                i = int(r["fiberid"])
                coef[i] = [[float(r[name]) for name in row] for row in names]
                ok[i] = r["quality"] == "measured"
            coefficients[c], measured[c] = coef, ok
        return cls(coefficients, {c: np.asarray(k) for c, k in knots.items()}, measured)

    @staticmethod
    def channel(wavelength: float) -> str:
        """The channel (`"B"`, `"R"` or `"Z"`) a wavelength in Angstrom falls in."""
        return next(c for c in CHANNELS if wavelength < CHANNEL_UPPER[c])

    def _lookup(self, fibre_id: ArrayLike, wavelength: float):
        """M-spline weights `(..., 11)` and measured flags `(...)` for fibres at one wavelength."""
        c = self.channel(float(wavelength))
        t = self.knots[c]
        w = float(np.clip(wavelength, t[3], t[-4]))
        basis = np.array([BSpline(t, np.eye(4)[j], 3)(w) for j in range(4)])
        fid = np.asarray(fibre_id, dtype=int)
        coef, measured = self.coefficients[c], self.measured[c]
        inside = (fid >= 0) & (fid < len(measured))
        ok = np.zeros(fid.shape, dtype=bool)
        ok[inside] = measured[fid[inside]]
        a = np.full(fid.shape + (N_OFFSET,), np.nan)
        a[ok] = coef[fid[ok]] @ basis
        return a, ok

    def weights(self, fibre_id: ArrayLike, wavelength: float) -> NDArray:
        """Weights of the 11 M-splines for each fibre at one wavelength, shape
        `fibre_id.shape + (11,)`. NaN for fibres without a measured LSF."""
        return self._lookup(fibre_id, wavelength)[0]

    def is_measured(self, fibre_id: ArrayLike, wavelength: float) -> NDArray:
        """Whether each fibre has a measured LSF in the channel of `wavelength`."""
        return self._lookup(fibre_id, wavelength)[1]

    def density(self, fibre_id: ArrayLike, wavelength: float, offset: ArrayLike) -> NDArray:
        """The LSF in 1/Angstrom at offsets from the line centre.

        Args:
            fibre_id: Fibre IDs (the slitmap's `fiberid`), any shape.
            wavelength: Wavelength in Angstrom (one value).
            offset: Offsets from the line centre in Angstrom, any shape; zero outside ±2.5 Å.

        Returns:
            Shape `fibre_id.shape + offset.shape`; NaN for fibres without a measured LSF.
        """
        a = self.weights(fibre_id, wavelength)
        m = _mspline_basis(np.asarray(offset, dtype=float))
        return np.tensordot(a, m, axes=([-1], [-1]))

    def moments(self, fibre_id: ArrayLike, wavelength: float) -> tuple[NDArray, NDArray]:
        """Mean offset and standard deviation of each fibre's LSF, in Angstrom (exact).

        Returns:
            `(mean, sigma)`, each of shape `fibre_id.shape`; NaN for fibres without a
            measured LSF.
        """
        a = self.weights(fibre_id, wavelength)
        mean = a @ _M_MEAN
        m2 = a @ (_M_MEAN**2 + _M_VAR)
        return mean, np.sqrt(m2 - mean**2)

    def mixture(
        self,
        fibre_id: ArrayLike,
        wavelength: float,
        n_components: int = 3,
        sigma: ArrayLike | None = None,
    ) -> LSFMixture:
        """Approximate each fibre's LSF by a sum of Gaussians.

        The mixture is fitted by least squares to the LSF centred on its mean and in units of
        its standard deviation, so it describes the shape only. It is then stretched by
        `sigma` (by default the LSF's own standard deviation): the LSF's mean is put at the
        line centre and its standard deviation becomes `sigma`. The mixture's own mean and
        standard deviation match these up to the fit error (`LSFMixture.mean` and
        `LSFMixture.sigma` give them). Fibres without a measured LSF get a single Gaussian (one component of
        weight one, the others of weight zero) at the line centre, of standard deviation
        `sigma` (NaN if `sigma` is not given).

        Offset components are what allow a flat top: Gaussians that all share one centre
        always give a profile at least as peaked as a Gaussian.

        Args:
            fibre_id: Fibre IDs, any shape.
            wavelength: Wavelength in Angstrom (one value).
            n_components: Number of Gaussians, 1 to 4.
            sigma: Standard deviation to give each mixture, in Angstrom, broadcastable to
                `fibre_id` (e.g. the DRP's LSF width). Defaults to the LSF's own.

        Returns:
            The mixtures, with arrays of shape `fibre_id.shape + (n_components,)`.
        """
        if not 1 <= n_components <= 4:
            raise ValueError("n_components must be between 1 and 4")
        fid = np.asarray(fibre_id, dtype=int)
        a, ok = self._lookup(fid, wavelength)
        mean, sd = self.moments(fid, wavelength)
        target = np.broadcast_to(sd if sigma is None else np.asarray(sigma, float), fid.shape)

        shape = fid.shape + (n_components,)
        w = np.zeros(shape)
        m = np.zeros(shape)
        s = np.ones(shape)
        w[..., 0] = 1.0
        t = np.linspace(-4.0, 4.0, 161)
        cache: dict[bytes, tuple[NDArray, NDArray, NDArray]] = {}
        for idx in zip(*np.nonzero(ok)) if fid.ndim else ([()] if ok else []):
            key = a[idx].tobytes()
            if key not in cache:
                profile = (a[idx] @ _mspline_basis(mean[idx] + t * sd[idx]).T) * sd[idx]
                cache[key] = _fit_standardised(t, profile, n_components)
            w[idx], m[idx], s[idx] = cache[key]
        # The fit is in units of the LSF's standard deviation, so stretching to the target is
        # a multiplication.
        return LSFMixture(w, m * target[..., None], s * target[..., None], ok)


def _fit_standardised(t: NDArray, profile: NDArray, k: int):
    """Fit k Gaussians to a profile sampled at standardised offsets t (mean 0, sd 1)."""
    if k == 1:
        return np.ones(1), np.zeros(1), np.ones(1)
    start = {
        2: ([0.5, 0.5], [-0.6, 0.6], [0.8, 0.8]),
        3: ([0.4, 0.3, 0.3], [0.0, -0.8, 0.8], [0.7, 0.7, 0.7]),
        4: ([0.25] * 4, [-1.0, -0.35, 0.35, 1.0], [0.6] * 4),
    }[k]
    w0, m0, s0 = (np.asarray(x) for x in start)
    p0 = np.r_[np.log(w0), m0, np.log(s0)]

    def unpack(p):
        lw = p[:k] - p[:k].max()
        w = np.exp(lw) / np.exp(lw).sum()
        return w, p[k : 2 * k], np.exp(p[2 * k :])

    def resid(p):
        w, mu, sg = unpack(p)
        z = (t[:, None] - mu) / sg
        return (w * np.exp(-0.5 * z**2) / (sg * np.sqrt(2 * np.pi))).sum(1) - profile

    w, mu, sg = unpack(least_squares(resid, p0, method="lm").x)
    order = np.argsort(mu)
    return w[order], mu[order], sg[order]
