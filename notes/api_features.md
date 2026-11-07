# Microburbs API features

**Result: nothing from the API earned a place in the model.** No adverts, geography or market series beat the current model on Fold B by the $1 bar with a paired interval clear of zero. Fold A was not read. Nothing from the API is in the submission, and `src/features.py` is unchanged.

All figures are Fold B (Apr-May 2026), median absolute error in $/week, from `src.models.run_experiment` with `report_fold_a` False. Row-level output is under `out/real/api/` and `api_cache/`, both gitignored.

## Fold B effect
Control, 5 seeds: **$42.98** overall, **$58.58** first-time flats (`anchor_own` blank in the base panel, 556 scored adverts), $38.99 repeat flats. The "about $56" in the older notes came from the pre-removal model.

Adverts, 5 seeds, against the 5-seed control:

| Added to history | Overall | First-time | Paired overall (95% CI) | Paired first-time (95% CI) |
|---|---|---|---|---|
| Input flats' extra adverts | $42.79 | $58.41 | -$0.37 (-1.71, +1.15) | +$0.74 (-3.31, +4.20) |
| Also units in input buildings | $42.95 | $59.13 | +$0.35 (-1.30, +1.65) | +$2.84 (-2.92, +6.23) |

Feature groups, 3 seeds (7, 11, 23), against a 3-seed control of $42.61 / $58.48:

| Added | Overall | First-time | Paired overall (95% CI) | Paired first-time (95% CI) |
|---|---|---|---|---|
| All geography (31 columns) | $43.39 | $57.61 | +$0.59 (-0.32, +2.08) | -$0.93 (-4.17, +1.60) |
| Market series | $41.84 | $58.52 | -$0.96 (-2.17, +0.23) | -$0.17 (-3.35, +2.40) |
| Crime (not recommended, see dating) | $43.39 | $59.23 | +$0.97 (-0.01, +2.07) | +$1.30 (-1.66, +3.46) |
| Transport | $43.34 | $58.55 | +$0.20 (-0.50, +1.75) | +$1.02 (-1.56, +3.13) |
| Amenities | $43.43 | $58.88 | +$0.67 (-0.31, +2.03) | +$2.53 (-1.30, +4.82) |
| Schools | $42.54 | $57.38 | -$0.49 (-1.45, +0.73) | -$1.67 (-5.02, +0.59) |
| Risks | $43.43 | $58.70 | +$0.29 (-0.58, +1.45) | +$0.18 (-2.99, +2.08) |
| Roads and noise | $42.67 | $57.95 | -$0.27 (-1.28, +0.86) | +$0.02 (-3.64, +2.54) |
| Planning | $42.59 | $57.49 | +$0.11 (-0.86, +1.47) | -$1.45 (-3.92, +1.67) |

The 3-seed control is flattering: seed SD was $0.02 on those three and $0.71 on all five. So the two leads were rerun on 5 seeds. They were picked after seeing Fold B, so this checks seed noise and is not a fresh test.

| 5 seeds | Overall | First-time | Bias | Paired overall (95% CI) | Paired first-time (95% CI) |
|---|---|---|---|---|---|
| Control | $42.98 | $58.58 | +$6.38 | | |
| Market series | $42.02 | $57.93 | +$3.55 | -$0.66 (-1.89, +0.53) | +$0.13 (-3.26, +2.15) |
| Schools | $42.78 | $57.59 | +$5.84 | -$0.42 (-1.42, +0.73) | -$1.24 (-3.41, +1.33) |

Paired figures compare seed-averaged predictions, so they differ from the gap between seed means. Paired intervals bootstrap adverts only.

**Market series are the one lead.** Bias roughly halves, but that is one time-state in one fold, the gain is under $1 and within-$100 does not improve. Testing it needs a second window, and Fold A is off limits for selection.

## What failed or surprised
- **More history barely moved anchors.** 49 of 577 first-time rows gained an own anchor and building-anchor coverage stayed at 548; the mean building count rose from 37.2 to 43.4. First-time error did not improve, so the gap looks like missing flat-level detail, not thin history.
- **R² rose from 0.878 to about 0.895 with extra adverts,** but from a few huge misses (one error fell from $1,717 to $127). Median error is unchanged, so this is not a rank-metric gain.
- **Geography adds little to latitude, longitude and building anchors,** which already carry location.
- **`properties/{id}/profile` has no floor area or year built.** It is ABS geography only, so the biggest data gap remains.
- **`development/all` overlays are not point values.** They list a whole area's values, so height and FSR come from `building-controls` polygons instead.
- **`transport/nearby` mode labels are unreliable.** `tram` was null for all 500 buildings sampled and `train` mixes in light rail. Stops are split by name, and 43 unnamed points were placed on light rail because they follow the Inner West and George St lines. This classification was judged by eye.
- **Default Python client is refused.** Cloudflare error 1010 blocks the Python-urllib signature, so `src/api.py` sends an honest project User-Agent. My first run wrongly cached those refusals as "outside coverage"; the 9 files were removed, and only the API's own JSON 403 now counts.

## Sale history: partial, not tested
SALE_STATUS_PLACEHOLDER

It was deliberately not tested as a feature. Partial coverage would differ between fold rows and final rows, the skew removed in DECISIONS 20. `add_sales` is written and unit-tested but not called, and `local/api_experiment.py` has no sales test. Every sale is cut to 31 July before caching, and `add_sales` cuts to each row's origin.

**Follow-up for teammate compute:** finish the pull (`local/pull_api.py sales`, resumable, about 100 minutes at 8 workers), confirm with `local/check_api_cache.py`, compare blank rates, then test on Fold B. It is the best untested idea for first-time flats, because a flat's price is a stand-in for size and quality.

## Endpoints and dating evidence
`src/api.py` allows only the endpoints below, and any other is refused in code.

| Endpoint | Treatment | Evidence |
|---|---|---|
| `listings/rent-history` | `to_date=2026-07-31` is required; a later or undated advert discards the response | Latest cached date is 2026-07-31; 141 of 141 pages have nothing later |
| `median-rent-series`, `vacancy-rate-series` | Cut to 31 July before caching, then to each row's origin | Latest cached date 2026-07-26. Publication lag and revisions cannot be seen |
| `sale-history` | Cut to 31 July before caching, then to each row's origin | Unit tested |
| `transport`, `amenities`, `schools`, `risks/all`, `development/all`, `busy-roads`, `noise-heatmap`, `building-controls` | Static, one value per building | Undated: they show the world today, so a recently opened shop can appear in early training rows. Small, and the same for fold and final rows |
| `crime/by-mesh-block` | Tested alone, **left out** | Definition says trained on QLD incidents from 2020-2022, but `period` is only "annual" and the inputs of the prediction are undated. Its Fold B result was no better anyway |
| `profile` | Inspected, not used | No useful field |

Not called: every endpoint on the banned list. `properties/{id}/rent-history` was also avoided, because it has no date bound.

## Comparing API adverts with the supplied file
- 123,261 API adverts to 31 July in the 29 suburbs. 36,592 are not in the file by flat and date.
- 6,415 of those are for input flats, across 5,311 flats. 635 of those flats are marked never advertised. 1,321 repeat a file advert's rent within 30 days.
- About 15,100 more are units at an input building's coordinates.
- New adverts are 6-9% of the file's monthly count through 2025-26 with no drop before July, so folds and final rows should see them alike.
- 16,533 file rows in covered suburbs are not in the API by flat and date, so neither source contains the other.
- 4.6% of adverts in both have different rents (API higher 79% of the time, no recent spike). The file's rent always wins. API-only rows may carry the same noise.
- API series against our index to the Fold B cutoff: 12-month growth 5.4% against 5.3% in median suburb, which is reassuring. Suburb correlation is weak (0.08 at 12 months), partly because API medians move in coarse steps.

## Coverage
- Representatives: one flat for each of 3,023 (suburb, building_key) pairs. 99% of keys span under about 70 m. The brief's 7,400 coordinates were reduced for time.
- Cache check (`local/check_api_cache.py`): all suburb series and shapes 29 of 29, advert pages 141 of 141, all five building endpoints 3,023 of 3,023, no temporary files, nothing later than the cutoff. Vacancy data exists for 25 of 29 suburbs.
- Newtown, Paddington, Annandale and St Peters (5,225 flats) are outside coverage and stay blank.

Blank rate, Fold B rows / training rows / final rows. Rates match closely, so the earlier fold skew is not present:

| Feature | Fold B | Training | Final |
|---|---|---|---|
| Most features (uncovered suburbs) | 7.8% | 7.6% | 7.9% |
| `mkt_vacancy` | 8.9% | 8.7% | 8.9% |
| `supermarket_m` | 15.0% | 15.2% | 15.0% |
| `height_limit_m` | 17.3% | 17.6% | 17.4% |
| `environment_m` | 38.4% | 40.2% | 38.7% |
| `ferry_m` | 43.5% | 44.4% | 43.9% |
| `flood_m` | 51.2% | 51.2% | 52.3% |

Blanks above 7.8% mostly mean nothing was found in the search radius.

## Run notes
- Targets and labels are always the file's. API adverts enter history only, each only if dated on or before that row's origin. 2,183 adverts score instead of 2,182 with extra history, because one gains known parking.
- The earlier jobs ran side by side and added memory load during the 12:54 and 12:55 kills. Those killed processes were not these jobs, and every result here is from a run that exited 0. After that, one process and two threads; the control reproduced exactly on two threads.
- `tests/test_api.py` covers the leak guards.
- Rerun: `local/pull_api.py`, then `local/check_api_cache.py`, then `local/api_experiment.py history_test`, `geo_test`, `leads_test`. Run them one at a time.
