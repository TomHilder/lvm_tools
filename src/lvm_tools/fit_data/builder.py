"""builder.py - FitDataBuilder class for constructing FitData with reproducibility."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from lvm_tools.config.data_config import DataConfig
from lvm_tools.data.tile import LVMTile, LVMTileLike
from lvm_tools.fit_data.fit_data import FitData
from lvm_tools.fit_data.processing import (
    flatten_tile_coord,
    get_normalisation_functions,
    process_tile_data,
)


@dataclass(frozen=True)
class FitDataBuilder:
    """Builds `FitData` from tiles and a config, and identifies the pair by a hash.

    Attributes:
        tiles: The data: one tile or a collection.
        config: The processing config, usually from `DataConfig.from_tiles` on the same
            tiles.

    Example:
        ```python
        from lvm_tools import DataConfig, FitDataBuilder, LVMTile

        tile = LVMTile.from_file(path)
        config = DataConfig.from_tiles(tile, λ_range=(6500.0, 6800.0))

        builder = FitDataBuilder(tiles=tile, config=config)
        fit_data = builder.build()
        ```
    """

    tiles: LVMTileLike
    config: DataConfig

    def build(self) -> FitData:
        """Process the tiles with the config and package the result for fitting.

        The steps are:

        1. Clip the data to the config's wavelength, RA and Dec ranges.
        2. Set the data the config excludes to NaN (NaNs, bad flux, fibre status, mask).
        3. Flatten the tile and spaxel dimensions into one spaxel dimension.
        4. Make the normalisation functions from the config's offsets and scales (the
           normalisation strategies are not applied again here).

        Returns:
            The data, ready for fitting.

        Raises:
            ValueError: If no data are left after clipping or filtering, or a strategy is
                not supported by the filtering.
        """
        return FitData(
            flatten_tile_coord(process_tile_data(self.tiles, self.config)),
            *get_normalisation_functions(self.config),
        )

    def hash(self) -> str:
        """A SHA-256 hash identifying the config and the tiles.

        It is computed from the config's attributes and each tile's metadata (filename,
        tile ID, exposure number and DRP version), both serialised as JSON with sorted
        keys. It identifies the inputs, not the data values.

        Returns:
            The hash, as a hexadecimal string.

        Example:
            ```python
            pipeline_hash = builder.hash()
            print(f"Pipeline: {pipeline_hash[:16]}...")
            ```
        """
        data = {
            "config": json.dumps(self._configdict, sort_keys=True),
            "tiles": json.dumps(self._metadict, sort_keys=True),
        }
        serialised = json.dumps(data, sort_keys=True)
        return sha256(serialised.encode()).hexdigest()

    @property
    def _configdict(self) -> dict:
        return self.config.to_dict()

    @property
    def _metadict(self) -> dict:
        # Ensure meta is always a dict of LVMTileMeta
        if isinstance(self.tiles, LVMTile):
            meta = {self.tiles.meta.exp_num: self.tiles.meta}
        else:
            meta = self.tiles.meta
        return {key: asdict(value) for key, value in meta.items()}
