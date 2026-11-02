# Utilities API

Physical corrections, masking for plots, and helpers for reading FITS data.

## Barycentric corrections

::: lvm_tools.physical_properties.barycentric_corr.get_v_barycentric

## Spatial masking

`mask_near_points` is useful for restricting a model evaluated on a grid (e.g. a map made
on a grid of `sky_x`, `sky_y`) to the region covered by data, so sparse regions aren't
extrapolated into.

::: lvm_tools.utils.mask.mask_near_points

## Helper functions

These are in `lvm_tools.data.helper`.

::: lvm_tools.data.helper.daskify_native

::: lvm_tools.data.helper.numpyfy_native

::: lvm_tools.data.helper.summarize_with_units

::: lvm_tools.data.helper.convert_sci_to_int
