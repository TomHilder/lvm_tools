from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from numpy.typing import ArrayLike, NDArray


# def get_v_barycentric(fit_data: FitData, unit="km/s") -> NDArray:
def get_v_barycentric(mjd: ArrayLike, α: ArrayLike, δ: ArrayLike, unit="km/s") -> NDArray:
    """Barycentric radial velocity correction at Las Campanas Observatory.

    Uses astropy's `SkyCoord.radial_velocity_correction` with the `"barycentric"` kind. The
    inputs broadcast against each other.

    Args:
        mjd: Modified Julian Date(s) of the observation(s).
        α: Right ascension(s) (ICRS), in degrees.
        δ: Declination(s) (ICRS), in degrees.
        unit (str): Unit of the result, any velocity unit astropy understands.

    Returns:
        The correction(s), in `unit`: a NumPy array, or a float for scalar inputs.

    Example:
        ```python
        import numpy as np
        from lvm_tools.physical_properties.barycentric_corr import get_v_barycentric

        # Single observation
        v_bary = get_v_barycentric(mjd=60000.5, α=180.0, δ=-30.0)
        print(f"v_bary = {v_bary:.2f} km/s")

        # Several observations
        v_bary = get_v_barycentric(
            mjd=np.array([60000.5, 60001.5]),
            α=np.array([180.0, 180.1]),
            δ=np.array([-30.0, -30.1]),
        )
        ```
    """
    times = Time(mjd, format="mjd")
    coords = SkyCoord(ra=α, dec=δ, obstime=times, unit="deg", frame="icrs")
    location = EarthLocation.of_site("Las Campanas Observatory")
    return coords.radial_velocity_correction("barycentric", location=location).to_value(unit)
