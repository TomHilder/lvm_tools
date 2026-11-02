"""fit_data.py - FitData class for holding data ready to be fitted."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from jax.numpy import pi as π
from xarray import DataArray, Dataset

from lvm_tools.physical_properties.barycentric_corr import get_v_barycentric

if TYPE_CHECKING:
    from jaxtyping import Array as JaxArray
    from spectracles.model.data import SpatialDataLVM

try:
    from spectracles.model.data import SpatialDataLVM
except Exception as e:  # noqa: BLE001 (any failure importing spectracles, e.g. a broken jax-finufft build)
    _import_err = e
    msg = (
        "Could not import SpatialDataLVM from spectracles.model.data. "
        "Install the optional dependency: pip install spectracles"
    )

    class SpatialDataLVM:  # stub preserves the API shape
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(msg) from _import_err

    import warnings

    warnings.warn(msg, ImportWarning)


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"FitData.{old} (coordinates scaled into [-π, π]) is deprecated: use {new}, with a "
        "spectracles Domain for physical length scales.",
        DeprecationWarning,
        stacklevel=3,
    )


def to_π_domain(x):
    # return x * 2 * π - π
    return x * 2 * np.pi - np.pi


def from_π_domain(x):
    return (x + π) / (2 * π)


def to_jax_array(arr: DataArray, dtype=np.float64) -> JaxArray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return jnp.array(arr, dtype=dtype)


@dataclass(frozen=True)
class FitData:
    processed_data: Dataset
    normalise_flux: Callable
    predict_flux: Callable
    normalise_ivar: Callable
    predict_ivar: Callable
    normalise_α: Callable
    _predict_α: Callable
    normalise_δ: Callable
    _predict_δ: Callable
    sky_origin: tuple[float, float] | None = None

    @property
    def _flux(self) -> JaxArray:
        return self.normalise_flux(to_jax_array(self.processed_data["flux"].values))

    @property
    def flux(self) -> JaxArray:
        return jnp.nan_to_num(self._flux)

    @property
    def _i_var(self) -> JaxArray:
        return self.normalise_ivar(to_jax_array(self.processed_data["i_var"].values))

    @property
    def i_var(self) -> JaxArray:
        return jnp.nan_to_num(self._i_var, nan=1e-4)

    @property
    def _u_flux(self) -> JaxArray:
        return self._i_var**-0.5

    @property
    def u_flux(self) -> JaxArray:
        return jnp.nan_to_num(self._u_flux, nan=1e2)

    # Legacy: coordinates scaled into [-π, π] (deprecated, use the sky_* offsets)

    def _α(self) -> JaxArray:
        return to_π_domain(self.normalise_α(to_jax_array(self.processed_data["ra"].values)))

    def _δ(self) -> JaxArray:
        return to_π_domain(self.normalise_δ(to_jax_array(self.processed_data["dec"].values)))

    @property
    def α(self) -> JaxArray:
        """Deprecated: RA scaled into [-π, π]. Use `sky_x` (arcsec)."""
        _deprecated("α", "sky_x")
        return self._α()

    @property
    def δ(self) -> JaxArray:
        """Deprecated: Dec scaled into [-π, π]. Use `sky_y` (arcsec)."""
        _deprecated("δ", "sky_y")
        return self._δ()

    def predict_α(self, x: JaxArray) -> JaxArray:
        """Deprecated: scaled RA back to degrees. Use `sky_to_radec`."""
        _deprecated("predict_α", "sky_to_radec")
        return self._predict_α(from_π_domain(x))

    def predict_δ(self, x: JaxArray) -> JaxArray:
        """Deprecated: scaled Dec back to degrees. Use `sky_to_radec`."""
        _deprecated("predict_δ", "sky_to_radec")
        return self._predict_δ(from_π_domain(x))

    @property
    def αδ_data(self) -> SpatialDataLVM:
        """Deprecated: spaxel positions scaled into [-π, π]. Use `sky_data` (arcsec)."""
        _deprecated("αδ_data", "sky_data")
        return SpatialDataLVM(
            x=self._α(),
            y=self._δ(),
            idx=self.spaxel_idx,
            tile_idx=self.tile_idx,
            ifu_idx=self.ifu_idx,
            fib_idx=self.fibre_idx,
        )

    # Sky positions as tangent-plane offsets in arcsec, east and north of a centre

    @property
    def sky_centre(self) -> tuple[float, float]:
        """The (RA, Dec) in degrees that `sky_x` and `sky_y` are measured from:
        `sky_origin` if set, otherwise the centre of the spaxels' bounding box."""
        if self.sky_origin is not None:
            return float(self.sky_origin[0]), float(self.sky_origin[1])
        ra = np.asarray(self.processed_data["ra"].values, dtype=float)
        dec = np.asarray(self.processed_data["dec"].values, dtype=float)
        # RA relative to one spaxel, so a footprint across RA = 0 isn't split
        dra = (ra - ra[0] + 180.0) % 360.0 - 180.0
        ra_c = (ra[0] + 0.5 * (dra.min() + dra.max())) % 360.0
        return float(ra_c), float(0.5 * (dec.min() + dec.max()))

    def _sky_frame(self) -> SkyOffsetFrame:
        ra_c, dec_c = self.sky_centre
        return SkyOffsetFrame(origin=SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg))

    def _sky_xy(self) -> tuple[np.ndarray, np.ndarray]:
        # float64: positions are stored in float32, which resolves only ~60 mas at RA ~ 270°
        coords = SkyCoord(
            ra=np.asarray(self.processed_data["ra"].values, dtype=float) * u.deg,
            dec=np.asarray(self.processed_data["dec"].values, dtype=float) * u.deg,
        ).transform_to(self._sky_frame())
        return coords.lon.to_value(u.arcsec), coords.lat.to_value(u.arcsec)

    @property
    def sky_x(self) -> JaxArray:
        """Each spaxel's offset east of `sky_centre`, in arcsec (on the tangent plane)."""
        return jnp.asarray(self._sky_xy()[0])

    @property
    def sky_y(self) -> JaxArray:
        """Each spaxel's offset north of `sky_centre`, in arcsec (on the tangent plane)."""
        return jnp.asarray(self._sky_xy()[1])

    @property
    def sky_data(self) -> SpatialDataLVM:
        """The spaxel positions for spectracles, as arcsec offsets from `sky_centre`.

        Length scales of spectracles fields are then in arcsec. Build the
        field's region with `spectracles.Domain.from_points(d.x, d.y)`.
        """
        x, y = self._sky_xy()
        return SpatialDataLVM(
            x=jnp.asarray(x),
            y=jnp.asarray(y),
            idx=self.spaxel_idx,
            tile_idx=self.tile_idx,
            ifu_idx=self.ifu_idx,
            fib_idx=self.fibre_idx,
        )

    def sky_to_radec(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """(RA, Dec) in degrees of offsets (x east, y north) in arcsec from `sky_centre`,
        e.g. to plot a map made on a grid of `sky_x`, `sky_y`."""
        coords = SkyCoord(
            lon=np.asarray(x, dtype=float) * u.arcsec,
            lat=np.asarray(y, dtype=float) * u.arcsec,
            frame=self._sky_frame(),
        ).icrs
        return coords.ra.to_value(u.deg), coords.dec.to_value(u.deg)

    def with_sky_origin(self, ra: float, dec: float) -> FitData:
        """The same data, with sky offsets measured from (RA, Dec) in degrees, e.g. to
        put several datasets on one frame. The origin is a choice of frame, not part
        of the data, so `FitDataBuilder.hash` doesn't include it."""
        return replace(self, sky_origin=(float(ra), float(dec)))

    @property
    def λ(self) -> JaxArray:
        return to_jax_array(self.processed_data["wavelength"].values)

    @property
    def _lsf_σ(self) -> JaxArray:
        return to_jax_array(self.processed_data["lsf_sigma"].values)

    @property
    def lsf_σ(self) -> JaxArray:
        median_lsf_σ = jnp.nanmedian(self._lsf_σ)
        return jnp.nan_to_num(self._lsf_σ, nan=median_lsf_σ)

    @property
    def mjd(self) -> JaxArray:
        return to_jax_array(self.processed_data["mjd"].values)

    @property
    def mask(self) -> JaxArray:
        return ~jnp.isnan(self._flux)

    @property
    def λ_idx(self) -> JaxArray:
        return jnp.arange(len(self.λ), dtype=np.int64)

    @property
    def spaxel_idx(self) -> JaxArray:
        return jnp.arange(self.processed_data["ra"].size, dtype=np.int64)

    @property
    def tile_idx(self) -> JaxArray:
        tile = to_jax_array(self.processed_data["tile"].values, dtype=np.int64)
        return jnp.unique(tile, return_inverse=True)[1]

    @property
    def ifu_idx(self) -> JaxArray:
        ifu = self.processed_data["ifu_label"].values
        return to_jax_array(np.unique(ifu, return_inverse=True)[1], dtype=np.int64)

    @property
    def fibre_idx(self) -> JaxArray:
        fib = self.processed_data["fibre_id"].values
        # CANNOT do unique trick because we need to preserve "gaps"
        return to_jax_array(fib, dtype=np.int64)

    @property
    def v_bary(self) -> JaxArray:
        return to_jax_array(
            get_v_barycentric(
                mjd=self.mjd,
                α=to_jax_array(self.processed_data["ra"].values),
                δ=to_jax_array(self.processed_data["dec"].values),
                unit="km/s",
            )
        )

    def arcsec_to_model_domain(self, arcsec: float) -> float:
        """Deprecated: a length in arcsec in the [-π, π] coordinates. With `sky_data`,
        positions are already in arcsec."""
        _deprecated("arcsec_to_model_domain", "sky_data (already in arcsec)")
        deg = arcsec / 3600
        probe = 1.0
        return float(
            to_π_domain(self.normalise_α(probe + deg)) - to_π_domain(self.normalise_α(probe))
        )

    def __repr__(self):
        # TODO: add something here
        raise NotImplementedError
