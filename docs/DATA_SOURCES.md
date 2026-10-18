# HR dataset — data sources, licences, access

Scope: the Phase 2 high-resolution Swiss training/evaluation set built by
`server/app/pipeline/hr_dataset.py` + `hr_download.py` (60 sites, 4 tiles
each, 1 m EPSG:2056 grids; extended 2026-10 with 32 train/val-only sites),
and the headroom analysis in
`docs/benchmarks/hr_headroom.md`. Verified 2026-10; re-check upstream terms
before redistribution.

## Imagery and elevation products

| product | role | resolution | source / access | licence |
| --- | --- | --- | --- | --- |
| SWISSIMAGE DOP10 | `rgb` | 0.1 m (delivered 0.1/0.2) | swisstopo STAC `ch.swisstopo.swissimage-dop10` | free, FSDI/OGD, attribution |
| swissSURFACE3D Raster | `dsm_hr`, `dtm_hr` (reference) | 0.5 m | swisstopo STAC `ch.swisstopo.swisssurface3d-raster` | free, FSDI/OGD, attribution |
| swissALTI3D | `prior_90m`, `prior_250m` (resampled) | 0.5–2 m base | swisstopo STAC `ch.swisstopo.swissalti3d` | free, FSDI/OGD, attribution |
| swisstlmregio | water polygons (`water_frac`) | vector | swisstopo STAC `ch.swisstopo.swisstlmregio` (Product gpkg) | free, FSDI/OGD, attribution |
| Copernicus DEM GLO-30 | `coarse_dem` prior | 30 m | AWS Open Data `s3://copernicus-dem-30m` (no auth) | free; © DLR e.V. / Airbus DS under COPERNICUS (ESA/EU) |

- STAC endpoint: `https://data.geo.admin.ch/api/stac/v0.9`. **Recorded
  discrepancy:** the STAC item metadata reports `license: proprietary` for
  swisstopo collections while swisstopo terms are free use with attribution
  under the Federal Spatial Data Infrastructure (GeoIG/OGD). Treat the
  resolution as: free to use with attribution, redistribution only per
  swisstopo terms.
- Attribution strings: "© swisstopo" for all swisstopo products;
  "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018
  provided under COPERNICUS by ESA and the EU" for GLO-30.
- swisstopo publishes **no per-file checksums**; integrity is handled by a
  first-write-wins sha256 index (`server/data/hr_downloads.csv`) written at
  download time and re-checkable via `hr_dataset checksums --verify`.

## Vertical datums

- swissSURFACE3D heights are **LN02** (official Swiss datum); GLO-30 is
  **EGM2008** geoid. swissALTI3D is LN02 as well.
- Never silently removed: each tile records `datum_offset_m` measured on
  flat ground pixels (`|ndsm| < 2 m`, slope < 0.15, ≥ 512 px), with
  `datum_offset_method = site_median` fallback when a tile has no such
  pixels. Both the raw and the offset-corrected coarse arm are reported in
  `docs/benchmarks/hr_headroom.md`.
- Headroom finding (2026-10): offset-correcting the coarse arm did **not**
  improve tile-wide RMSE (median 8.07 → 8.79 m). Flat-pixel populations
  (valley floors, forest clearings) are not representative of the tile-wide
  ground difference, and in forest sites the measured offset partly reflects
  GLO-30 canopy blur over small clearings, not only the geoid. The stored
  offset stays the documented flat-ground estimate; it is not a tile-wide
  bias correction.

## Temporal alignment

- Products carry different epochs (RGB 2021–2025; swissSURFACE3D
  2020–2026). Each tile records `rgb_year`, `dsm_year`, `dtm_year`,
  `temporal_gap_years`; gaps > 4 years set `temporal_flag`.
- Flagged tiles (34): `napf_00`, `napf_01`, `samedan_*` (6 Phase-2 tiles),
  plus 28 extension tiles where 2024–25 RGB meets 2019–23 DSM/DEM epochs
  (`andermatt`, `arosa`, `chur`, `fluela`, `san_bernardino`, `val_mustair`,
  `val_onernonica`). They are listed, kept on disk, and excluded from the
  headroom primary aggregates; any Phase 3 primary test claim must exclude
  them too.

## Water

- `water_frac` comes from swisstlmregio lake / stagnant-water / flowing-water
  polygons; a fully water tile is invalid for elevation supervision.
  **Limitation:** rivers are mapped as lines upstream and are not buffered
  for this flag.

## Downloads and rebuild

```
cd server
uv run python -m app.pipeline.hr_dataset download --workers 4
uv run python -m app.pipeline.hr_dataset checksums --verify
uv run python -m app.pipeline.hr_dataset build
uv run python -m app.pipeline.headroom --report
```

Downloads are resumable (`.part` staging + sha256). Cache:
`server/data/hr_downloads/` (823 indexed files, ~23.7 GB at 2026-10); built
tiles: `server/data/hr/` (240 tiles); provenance:
`server/data/manifest_hr.csv`, `server/data/hr_downloads.csv`,
`server/data/hr_sites.csv`.
Coordinates come from the committed site table (`hr_sites.csv`); the
`val_colla` forest test site was replaced by `monte_san_giorgio` after a
coverage check showed the original point inside Italian territory (no
swisstopo data), verified by nDSM tree-cover and STAC checks. The extension
sites `lac_dix` and `schwarzsee` were initially placed on open water (one
tile at ≤ 4 % valid pixels); both points were shifted using the TLMregio
water polygons (all tiles now < 1 % water) before rebuild.

## Reference-only

- `ISRO-DEPTHWIZ` repository: no licence file; consulted as a reference
  only, no code copied. Any future use requires a licence check.
