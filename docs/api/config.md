# Configuration API

`DataConfig` holds every setting for preparing data for fitting: which wavelengths and sky
region to keep, which data to exclude, and how to normalise the flux and coordinates.
It is exported from the top-level `lvm_tools` namespace.

```python
from lvm_tools import DataConfig

config = DataConfig.from_tiles(tile, λ_range=(6500.0, 6800.0))  # computed from the data
config = DataConfig.default()  # all defaults
d = config.to_dict()
config = DataConfig.from_dict(d)
```

## DataConfig

::: lvm_tools.DataConfig

## Type aliases

::: lvm_tools.fit_data.clipping.Range

::: lvm_tools.fit_data.clipping.Ranges

::: lvm_tools.fit_data.filtering.ExcludeStrategy

::: lvm_tools.fit_data.filtering.FibreStatus

::: lvm_tools.fit_data.normalisation.NormaliseStrategy

## Validation functions

`DataConfig` uses these to check its settings when it is created.

::: lvm_tools.config.validation.validate_range

::: lvm_tools.config.validation.validate_excl_strategy

::: lvm_tools.config.validation.validate_norm_strategy

::: lvm_tools.config.validation.validate_fib_status_incl

::: lvm_tools.config.validation.validate_offset

::: lvm_tools.config.validation.validate_scale

::: lvm_tools.config.validation.validate_apply_mask
