# Data Loading API

Classes for reading LVM DRP files lazily and combining exposures. `LVMTile` and
`LVMTileCollection` are exported from the top-level `lvm_tools` namespace; the rest lives
in `lvm_tools.data`.

## LVMTile

::: lvm_tools.LVMTile

## LVMTileMeta

::: lvm_tools.data.tile.LVMTileMeta

## LVMTileCollection

::: lvm_tools.LVMTileCollection

## Type aliases

::: lvm_tools.data.tile.LVMTileLike

## Constants

These are defined in `lvm_tools.data.tile`.

::: lvm_tools.data.tile.SIGMA_TO_FWHM

::: lvm_tools.data.tile.FWHM_TO_SIGMA

::: lvm_tools.data.tile.FLUX_UNIT

::: lvm_tools.data.tile.SPECTRAL_UNIT

::: lvm_tools.data.tile.SPATIAL_UNIT

::: lvm_tools.data.tile.CHUNKSIZE

## Reading DRP files

::: lvm_tools.data.tile.get_science_inds

::: lvm_tools.data.coordinates.get_mjd

::: lvm_tools.data.coordinates.get_observatory_code

::: lvm_tools.data.coordinates.get_observatory_location
