# Processing API

The functions behind `DataConfig.from_tiles` and `FitDataBuilder.build`: clipping to
wavelength and sky ranges, setting bad data to NaN, and normalisation. Use them directly to
inspect intermediate steps, e.g. `filter_inspector` to see how much data each filter
removes.

## Pipeline functions

These apply a `DataConfig` to tiles. `FitDataBuilder.build` runs `process_tile_data`, then
`flatten_tile_coord`, then `get_normalisation_functions`.

::: lvm_tools.fit_data.processing.process_tile_data

::: lvm_tools.fit_data.processing.clip_data

::: lvm_tools.fit_data.processing.filter_tile_data

::: lvm_tools.fit_data.processing.get_αδ_ranges

::: lvm_tools.fit_data.processing.get_normalisations

::: lvm_tools.fit_data.processing.get_normalisation_functions

::: lvm_tools.fit_data.processing.flatten_tile_coord

## Clipping functions

::: lvm_tools.fit_data.clipping.clip_dataset

::: lvm_tools.fit_data.clipping.clip_wavelengths

::: lvm_tools.fit_data.clipping.bounding_square

::: lvm_tools.fit_data.clipping.ensure_ranges

::: lvm_tools.fit_data.clipping.slice_mask

::: lvm_tools.fit_data.clipping.verify_range

::: lvm_tools.fit_data.clipping.verify_ranges

## Filtering functions

::: lvm_tools.fit_data.filtering.filter_dataset

::: lvm_tools.fit_data.filtering.filter_inspector

::: lvm_tools.fit_data.filtering.get_where_nan

::: lvm_tools.fit_data.filtering.get_where_bad

::: lvm_tools.fit_data.filtering.get_where_bad_median

::: lvm_tools.fit_data.filtering.get_where_bad_max

::: lvm_tools.fit_data.filtering.get_where_badfib

::: lvm_tools.fit_data.filtering.get_where_mask

::: lvm_tools.fit_data.filtering.combine_wheres

## Normalisation functions

::: lvm_tools.fit_data.normalisation.calc_normalisation

::: lvm_tools.fit_data.normalisation.normalise

::: lvm_tools.fit_data.normalisation.denormalise

::: lvm_tools.fit_data.normalisation.get_norm_funcs

## Constants

::: lvm_tools.fit_data.filtering.BAD_FLUX_THRESHOLD

::: lvm_tools.fit_data.normalisation.NORM_PADDING
