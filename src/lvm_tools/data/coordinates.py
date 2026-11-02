"""coordinates.py - observation time and location extraction for LVM data processing."""

from astropy.coordinates import EarthLocation
from astropy.io.fits import Header
from astropy.time import Time


def get_mjd(header: Header) -> float:
    """The Modified Julian Date of the midpoint of an exposure.

    Args:
        header: A DRP primary header with the `INTSTART` and `INTEND` keywords (ISO times
            of the start and end of the integration).

    Returns:
        The MJD halfway between `INTSTART` and `INTEND`.

    Raises:
        ValueError: If either keyword is missing.
    """
    if "INTSTART" in header and "INTEND" in header:
        start_time = Time(header["INTSTART"], format="isot")
        end_time = Time(header["INTEND"], format="isot")
        mid_time = start_time + (end_time - start_time) / 2
        return mid_time.mjd
    else:
        raise ValueError("Could not find sufficient time information in header")


def get_observatory_code(header: Header) -> str:
    """The observatory code in a FITS header's `OBSERVAT` keyword.

    Args:
        header: A DRP primary header.

    Returns:
        The code, e.g. `"LCO"`.

    Raises:
        ValueError: If the code is missing or not a known observatory (only `"LCO"` is
            known).
    """
    observatory = header.get("OBSERVAT", "").strip()
    known_observatories = ["LCO"]
    if observatory in known_observatories:
        return observatory
    else:
        raise ValueError(
            f"Unknown observatory: {observatory}. Known observatories: {known_observatories}"
        )


def get_observatory_location(observatory: str) -> EarthLocation:
    """The location on Earth of an observatory.

    Args:
        observatory: Observatory code. Only `"LCO"` (Las Campanas Observatory) is supported.

    Returns:
        The observatory's location, from astropy's site registry.

    Raises:
        ValueError: If the observatory is unknown.
    """
    if observatory == "LCO":
        return EarthLocation.of_site("Las Campanas Observatory")
    else:
        raise ValueError(f"Unknown observatory: {observatory}")
