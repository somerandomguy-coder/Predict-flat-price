# Public data: what was found and whether it helps

All scores are real data, **Fold B only**, median absolute error in $/week. Fold A was not scored. Differences under $1 are treated as noise.

## Result
**No public source improved Fold B overall. Do not add these features to the submission.**

- Best feature set after re-testing: strata + OSM, $43.27 and $43.19 against a baseline of $42.94 and $42.66 (two sets of 10 fresh seeds).
- It looked best at first ($42.55 on 3 seeds), but that was the top of 13 tries. It did not repeat.
- It may help first-time flats by $1 to $2, but it costs flats with an earlier advert about $1. The two cancel.
- Using it for first-time flats only: -$0.19, 90% interval [-0.59, +0.19]. Under the $1 bar, and the rule was chosen after seeing Fold B.

## Fold B results
Baseline here is the current 33-column model, log(rent / anchor). G-NAF is the Feb 2026 release, dated before the Fold B cutoff.

**Screen: 3 seeds (100-102), 13 sets, complete run.** First-time counts 577 adverts, before dropping unknown parking.

| Set | Columns | Fold B | Seed sd | Within $100 | Bias | First-time | Repeat |
|---|---|---|---|---|---|---|---|
| Baseline | 33 | $43.01 | 0.34 | 81.2% | +$6.21 | $58.25 | $38.92 |
| + G-NAF strata plan and lot | 38 | $43.34 | 1.01 | 81.2% | +$5.17 | $56.58 | $39.30 |
| + G-NAF site size, unit number | 35 | $43.14 | 0.34 | 81.3% | +$5.68 | $58.29 | $39.64 |
| + G-NAF floor | 36 | $43.09 | 0.25 | 81.1% | +$6.27 | $57.66 | $39.52 |
| + G-NAF created year | 34 | $43.18 | 0.28 | 81.2% | +$5.97 | $58.88 | $39.55 |
| + floor proxy (lot rank x storeys) | 34 | $42.80 | 0.43 | 81.4% | +$5.98 | $56.62 | $39.06 |
| + OSM buildings | 38 | $43.26 | 0.18 | 81.2% | +$5.44 | $57.23 | $39.23 |
| + cadastre site area | 35 | $43.12 | 0.29 | 81.1% | +$5.65 | $58.22 | $39.43 |
| + Development Monitoring | 38 | $43.11 | 0.41 | 81.2% | +$6.37 | $58.47 | $39.09 |
| + Census and mesh block counts | 42 | $43.51 | 0.45 | 81.1% | +$5.67 | $58.09 | $39.23 |
| + transport and amenities | 41 | $42.71 | 0.49 | 81.3% | +$5.92 | $58.87 | $38.64 |
| + strata + OSM | 43 | $42.55 | 0.20 | 81.2% | +$4.68 | $56.34 | $38.62 |
| + everything | 74 | $43.28 | 0.69 | 81.2% | +$3.92 | $56.62 | $39.58 |

**Re-test of strata + OSM: paired by advert, 1,000 bootstrap resamples, 90% intervals.** First-time here is the 556 scored adverts.

| Group | Seeds | Baseline | Strata + OSM | Gap | Interval |
|---|---|---|---|---|---|
| All (2,182) | 200-209 | $42.94 | $43.27 | +$0.33 | [-0.22, +1.10] |
| First-time (556) | 200-209 | $58.72 | $56.40 | -$2.33 | [-3.72, -0.96] |
| Repeat (1,626) | 200-209 | $38.68 | $39.42 | +$0.74 | [+0.14, +1.52] |
| All (2,182) | 300-309 | $42.66 | $43.19 | +$0.53 | [-0.04, +1.21] |
| First-time (556) | 300-309 | $58.36 | $57.18 | -$1.18 | [-2.88, +0.33] |
| Repeat (1,626) | 300-309 | $38.50 | $39.56 | +$1.06 | [+0.33, +1.68] |
| Routed, all (2,182) | 300-309 | $42.66 | $42.46 | -$0.19 | [-0.59, +0.19] |

The intervals cover advert sampling, not the choice of this set from 13. The first-time gain therefore leans optimistic, and one of the two seed sets does not exclude zero. Both sets use the same adverts, so they are not independent evidence.

## Gaps, as asked
| Gap | Outcome |
|---|---|
| Floor area | **Not found per flat.** Development Monitoring gives average area per dwelling for 123 buildings only (2.7% of flats). OSM footprint x storeys is a rough building figure. |
| Building age | **Proxy only.** Strata plan number covers 87.1% of flats and rises over time, but was not converted to years. No real build year at scale: OSM `start_date` is on 78 objects. Completion dates cover 123 buildings. |
| `floor_level` blanks | **Lead failed.** Blank count is 31,955 after the existing over-90 rule (31,381 in the raw file). G-NAF recovered 177. The supplied field matches G-NAF exactly on all 22,467 flats where both exist, so it appears to come from G-NAF already. |
| Floor proxy | Lot rank x OSM storeys: correlation 0.44 with known floors, within 2 floors 58%, against 43% for a mid-building guess. It fills 80% of flats, but it is an estimate, not a recovered floor. |
| `car_space` blanks | **Not filled.** No public source gives parking per flat. Development Monitoring has spaces per dwelling for 2.7% of flats. |
| Block field blanks | **Filled, no gain.** All 4,246 flats get 2021 mesh block counts and Census fields. They are absent from the blocks file because their mesh blocks are; G-NAF gives the same codes. |
| First-time flats | Possibly $1 to $2 better, at a cost to other flats. See Result. |

## Coverage of the 65,777 flats
All 65,777 `gnaf_id`s match G-NAF. Buildings: 3,656.

| Source | Field | Filled |
|---|---|---|
| G-NAF | Strata plan | 87.1% |
| G-NAF | Lot number, lot rank | 85.1% |
| G-NAF | Unit number | 99.0% |
| G-NAF | Floor (level or unit number) | 34.4% |
| G-NAF + supplied | Any known floor | 51.7% |
| OSM | Footprint | 92.9% |
| OSM | Storeys | 76.5% |
| OSM | Height | 7.0% |
| Cadastre | Site area | 37.2% |
| Development Monitoring | Area, parking, levels, completion | 2.7% |
| Census and mesh block | All fields | 100% |
| Transport and amenities | All fields | 100% |

## Skew test
Blank rate in Fold B rows against final rows. **All passed; the largest gap is 2.1 points.**

| Feature | Fold B | Final |
|---|---|---|
| `floor_filled` | 50.5% | 48.3% |
| `cad_site_m2` | 61.7% | 62.9% |
| `g_lot` | 15.8% | 14.8% |
| `floor_est` | 20.6% | 19.7% |
| `dm_*` | 96.6% | 97.3% |
| `osm_levels` | 23.9% | 23.5% |
| `g_sp_number` | 12.9% | 12.9% |
| Census, transport, `g_created_year` | 0% | 0% |

Full table: `out/real/public_data_feb.log` (not committed). This run gave final rows the Feb 2026 G-NAF too. May 2026 differs by under 0.1 points on these fields.

## Sources
| Source | Vintage and date check | Licence | Attribution |
|---|---|---|---|
| G-NAF, Geoscape, data.gov.au | Feb 2026 for Fold B (latest change 2 Feb 2026). May 2026 is the last release before 31 July, for final rows. August 2026 not used. | Open G-NAF EULA: CC BY 4.0 terms plus a limit on compiling addresses for mail, which does not apply here | Keep Geoscape's supplied notices, link the dataset and licence, and say the data was modified. No exact wording is prescribed. |
| OpenStreetMap via Overpass | Queried as at 2026-07-31 12:00 UTC, by bounding box | ODbL | "© OpenStreetMap contributors". Publishing a derived database brings share-alike duties. |
| NSW cadastre Lot layer, Spatial Services | **Later snapshot, Nov 2026.** Lots created after 31 July were removed. Flagged. | CC BY 3.0 AU, from the catalogue entry for the cadastral fabric. Terms of the REST service were not read separately. | Credit NSW Spatial Services |
| Development Monitoring, City of Sydney | Extract dated 30 June 2026. Forecast completions dropped. Completions after a row's cutoff are hidden. | CC BY 4.0 | Credit City of Sydney |
| ABS 2021 Census GCP (SA1), Mesh Block Counts, ASGS allocation | 2021; released 2022 | CC BY 4.0 | Credit ABS; see "Attributing ABS material" |

Only bounding boxes or whole public layers were requested. No address, `gnaf_id` or supplied row was sent anywhere. Files are in `data/external/<source>/` and row-level tables in `data/external/derived/`, all gitignored.

## Time safety
- `g_created_year` and Development Monitoring are masked by each row's cutoff in `attach()`.
- The other fields are undated and treated as fixed. Training origins back to 2024 still see 2026-vintage values. Feb and May G-NAF agree on 99.95% of strata plans and all lots and floors.
- 2021 Census predates every training origin used.
- G-NAF `DATE_CREATED` is not a build date: it clumps in load years such as 2008, 2019 and 2021.

## Could not get, or not tried
- **NSW Property Sales Information and bulk land values: blocked.** `valuergeneral.nsw.gov.au` and `valuation.property.nsw.gov.au` answer automated requests with a Cloudflare challenge (`cf-mitigated: challenge`). It was not bypassed. No mirror was found in the data.nsw catalogue. This is the best remaining lead for per-flat area, but the file contents were not seen, so whether strata area is populated is unconfirmed. A person could download the files in a browser into `data/external/`. Sales would need masking by each row's cutoff.
- **NSW bond data: not tried.** Files are listed, June 2026 is the last usable lodgement file, but the page states no licence and the rent index already tracks the market.
- **TfNSW GTFS: not used.** OSM stops stood in for it. No frequency data, so it measures distance, not service.
- **City of Sydney 3D model: not found** in a title search of the City's catalogue. **NSW building footprints: not properly searched;** only the first 60 entries of the Spatial Services list were read. FES 2022 block data was not tried; it is block-level, and its licence field is blank.
- **Cadastre history service:** needs a token, so it was not used.

## Known limits
- Features were added to the cached panel, so they do not reach the anchors. In particular, `anchor_bldg` still uses the supplied floor only.
- No tuning was tried. A model with more columns on the same settings may be under-regularised, which could explain the small repeat-flat cost.
- Distances to roads, rail lines and parks use way centre points, so they are rough.
- OSM storeys are below the known floor on 10.3% of tested flats.
- `osm_gfa_per_flat` has extreme values where one footprint covers a small flat count.
- Runs used `log=False`, so nothing was added to `experiments.csv`. Results are in `out/real/`.

## How to repeat
Scripts are in `local/public_data/`. Geometry builders need shapely and pyproj, which the project venv lacks. Run one job at a time; the harness uses 2 threads.

```
python local/run_public_data.py out/real/features_panel.parquet --seeds 3 --vintage feb2026 --final out/real/final_rows.parquet
python local/public_data/confirm.py out/real/features_panel.parquet feb2026 300
```
