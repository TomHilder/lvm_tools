# Roadmap

Planned work. Nothing here is a commitment until it is released.

## Remove the [-π, π] coordinates

`FitData.α`, `δ`, `αδ_data`, `predict_α`, `predict_δ` and
`arcsec_to_model_domain` are deprecated in favour of the arcsecond offsets
(`sky_x`, `sky_y`, `sky_data`, `sky_to_radec`) and will be removed in a future
release.

## Docs on Zensical

The documentation is built with Material for MkDocs, which reaches end of life
in May 2027. It will move to [Zensical](https://zensical.org), its successor
from the same team, which reads the existing `mkdocs.yml`. spectracles' docs,
built the same way, will move with it.
