# Fit Data API

`FitDataBuilder` turns tiles and a `DataConfig` into `FitData`: normalised, model-ready JAX
arrays with sky positions and indices. `FitDataBuilder` is exported from the top-level
`lvm_tools` namespace; `FitData` lives in `lvm_tools.fit_data.fit_data`.

```python
import jax.numpy as jnp
from lvm_tools import DataConfig, FitDataBuilder, LVMTile

tile = LVMTile.from_file("/data/observation.fits")
config = DataConfig.from_tiles(tile, λ_range=(6500.0, 6800.0))
fit_data = FitDataBuilder(tiles=tile, config=config).build()

# Spectra have shape (n_wavelengths, n_spaxels); average over the spaxels with data
weights = fit_data.mask.astype(jnp.float32)
mean_flux = jnp.sum(fit_data.flux * weights, axis=1) / jnp.sum(weights, axis=1)

# Spaxel positions in arcsec east and north of fit_data.sky_centre, and back to RA, Dec
x, y = fit_data.sky_x, fit_data.sky_y
ra, dec = fit_data.sky_to_radec(x, y)

print(f"Mean barycentric velocity: {float(fit_data.v_bary.mean()):.2f} km/s")
```

## FitDataBuilder

::: lvm_tools.FitDataBuilder

## FitData

::: lvm_tools.fit_data.fit_data.FitData
    options:
      members: false

### Sky positions

Positions are offsets in arcsec on the tangent plane at `sky_centre`.

::: lvm_tools.fit_data.fit_data.FitData.sky_x
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.sky_y
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.sky_centre
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.sky_to_radec
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.with_sky_origin
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.sky_data
    options:
      heading_level: 4
      show_root_full_path: false

### Spectral data

Arrays of shape `(n_wavelengths, n_spaxels)`, plus the wavelength grid.

::: lvm_tools.fit_data.fit_data.FitData.flux
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.i_var
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.u_flux
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.mask
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.λ
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.lsf_σ
    options:
      heading_level: 4
      show_root_full_path: false

### Indices and per-spaxel properties

::: lvm_tools.fit_data.fit_data.FitData.λ_idx
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.spaxel_idx
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.tile_idx
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.ifu_idx
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.fibre_idx
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.mjd
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.v_bary
    options:
      heading_level: 4
      show_root_full_path: false

### Deprecated coordinates

!!! warning "Deprecated"
    These members scale RA and Dec into [-π, π] and emit `DeprecationWarning`. Use the
    arcsec offsets instead: `sky_x` for `α`, `sky_y` for `δ`, `sky_data` for `αδ_data`,
    and `sky_to_radec` for `predict_α` and `predict_δ`. Positions from `sky_data` are
    already in arcsec, so `arcsec_to_model_domain` isn't needed.

::: lvm_tools.fit_data.fit_data.FitData.α
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.δ
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.αδ_data
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.predict_α
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.predict_δ
    options:
      heading_level: 4
      show_root_full_path: false

::: lvm_tools.fit_data.fit_data.FitData.arcsec_to_model_domain
    options:
      heading_level: 4
      show_root_full_path: false

## Helper functions

These are in `lvm_tools.fit_data.fit_data`; the [-π, π] mappings are used only by the
deprecated coordinates.

::: lvm_tools.fit_data.fit_data.to_jax_array

::: lvm_tools.fit_data.fit_data.to_π_domain

::: lvm_tools.fit_data.fit_data.from_π_domain
