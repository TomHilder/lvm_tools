# lvm_tools

Lightweight wrapper of [LVM DRP](https://github.com/sdss/lvmdrp) data with an emphasis on modularity. Allows for lazy reading via [`dask`](https://www.dask.org), especially useful for fitting large models. Designed for use with spectrospatial models via [`spectracles`](https://github.com/TomHilder/spectracles) but probably useful for other things too.

Feel free to contact me personally if you have any questions at all.

## Installation

Easiest is from PyPI either with `pip`

```sh
pip install lvm-tools
```

or `uv` (recommended)

```sh
uv add lvm-tools
```

Or, you can clone and build from source

```sh
git clone git@github.com:TomHilder/lvm_tools.git
cd lvm_tools
pip install -e .
```

## Usage

TODO

## Citation

TODO

## Help

TODO

## TODO

- [x] Extend wavelength range support in `DataConfig` to support multiple segments in λ
  - The point of this is to be able to fit two lines at very diffferent λ at once without reading _all_ the data between the lines
- [ ] The `DataConfig.from_tiles` is too slow, which I think is because the data does get all read in for `calc_normalisation`. Most of the time reading this in, the normalisation has been chosen in practice, so we should skip this step in that case right?
- [x] Relax version requirements from being strictly my environment (which is very up-to-date)
- [ ] repr for FitData
- [ ] Logging/hashing
- [ ] Cache
- [ ] OptConfig
- [ ] Testing ?
