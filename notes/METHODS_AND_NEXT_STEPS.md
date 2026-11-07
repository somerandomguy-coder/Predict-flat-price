# Methods, leakage audit and next steps

Team 2 Bulbs 1 Source. Numbers are real data unless marked. Median error is the rank metric, in $/week.
Run history is in `experiments.csv`; choices and reasons are in `DECISIONS.md`.

## Current result
| | Median error | Within $100 | R² | Bias |
|---|---|---|---|---|
| Fold A (Jun-Jul 2026) | $40.32 | 80.3% | 0.899 | +$3.44 |
| Fold B (Apr-May 2026) | $42.62 | 81.0% | 0.879 | +$6.56 |

Expect the real score to be worse than this. See "Why folds may flatter us".

## What has been tried
| Step | Fold A | Fold B | Outcome |
|---|---|---|---|
| Suburb-only median | $160 | $160 | Comparison only |
| Suburb x bedrooms x parking median | $100 | $95 | Baseline |
| Last rent, no index | $63.58 | $75.00 | Bias -$50 and -$66 |
| Indexed best anchor | $44.61 | $52.10 | The main win |
| Own rent anchor alone | $40.92 | $49.09 | Covers about 74% of adverts |
| Building anchor alone | $57.59 | $56.64 | |
| Block anchor alone | $67.78 | $65.93 | |
| LightGBM on rent, all features | $41.69 | $43.12 | Before removing skewed features |
| LightGBM on rent, date ages removed | not scored | $43.63 | |
| **LightGBM on log(rent / anchor), date ages removed** | **$40.32** | **$42.62** | **Current** |

Tried and rejected: forward trend projection (Fold B chose flat), within-building premium as the submitted gap (lost by rule, see below).

**Not tried:** hyperparameter tuning (fixed 400 rounds, one parameter set), other model families, seed or model ensembles, separate models for first-time flats, any API or external data, sample weighting by recency, `was_leased` handling.

## Leakage audit
Two different questions. Results are in `out/real/leak_audit.json` and from `local/leak_audit.py`.

**Could the frozen Aug-Sep adverts be in our model? No evidence of it.**
- The lease file, `attrs_last_seen` and `first_ever_listed` all end on 2026-07-31.
- `load_leases` also drops anything later.
- Nothing from the API is in the submission.
- Not provable: undated input fields such as parking and bedrooms. July 2026 adverts agree with them, which fits a 31 July snapshot, but we cannot see how the file was built.

**Do our folds leak? Checked and fixed where found.**
| Check | Result |
|---|---|
| Poison all rows after the cutoff (rent x5, bedrooms, parking) | 0 of the features moved |
| Labels later than a fold cutoff in training | Blocked by assertion and test |
| A row's own advert in its features | Blocked by rolling origins |
| Target advert's parking reaching anchors | Found and fixed |
| Input parking and bedrooms taken from the test advert | Found and fixed for flats with earlier adverts |
| Date-age features unlike final rows | Found and removed |

**Why folds may flatter us (not fully fixable):**
- About 25% of test adverts are first-time flats. Their input fields may come from the test advert itself. Making 5% of their bedrooms wrong moved error only from $56.20 to $56.91.
- Blank advert parking in history is filled from the input.
- 2026 block figures are used at earlier training origins. The task allows them, but it is a small look-ahead.
- Fold rows and final rows still differ (classifier AUC 0.79). Part of that is expected, because final rows include never-advertised flats.
- Folds hold about 1,300 adverts a month, while 1,699 are scored over two months. The mix may differ.
- Fold A has been run several times and its window was used in the leak check. No setting was tuned on it, but it is no longer pristine. Treat new Fold A numbers as a final check only.

**Selection noise:** the target choice changed when features changed, and the premium choice rests on 85 adverts. Differences under about $1 on Fold B are not meaningful.

## Where the error is
| Group (Fold B) | Adverts | Median error | Within $100 |
|---|---|---|---|
| Flats with an earlier advert | 1,626 | $39.82 | 83.5% |
| First-time flats | 556 | $56.20 | 73.9% |
| Parking flipped since last known | 85 | $53 | |

Figures are from the pre-removal model, so use them as a guide. Anchors older than 5 years also keep a bias near -$51.

## Data gaps
| Field | Blank | Idea |
|---|---|---|
| `floor_level` | 31,381 | Building storeys and unit number patterns |
| `car_space` | 5,793 | Building neighbours, API profile |
| Block fields | 4,246 | ABS mesh block counts |
| `first_ever_listed` | 279 | Low value |
| No floor area or building age | all | Biggest missing signal, and it would also clean the parking premium |

## API
Spec: `https://api-ht.microburbs.com.au/openapi.json`. Calls are free; 29 suburbs are covered, and anything else returns 403.

| Use | Endpoints |
|---|---|
| Safe: static geography | `transport/nearby`, `amenities/nearby`, `schools/nearby`, `risks/all`, zoning, `shapes/busy-roads`, `shapes/noise-heatmap`, `development/building-controls` |
| Safe if date-filtered | `listings/rent-history` with `to_date=2026-07-31`, `sale-history`, `market/median-rent-series`, `market/vacancy-rate-series` |
| Check first | `crime/by-mesh-block` (undated, possibly slow-moving) |
| Do not use | `listings/for-rent`, `rent-history-latest`, `surroundings/listing-history-signals`, `street/profile`, `vacancy-map`, `rent-map`, `context/mesh-blocks`, valuation endpoints: undated or may be built from later data |

A probe hit `listings/rent-history` without a date bound and got 200 adverts dated after 31 July. Only their dates were counted; nothing was saved or used.

## Next steps, by expected value
1. Help first-time flats: building and location features from the API and open data.
2. Add adverts dated by 31 July that the supplied file lacks, if the API has any.
3. Method comparison on Fold B: tuning, seeds, ensembles, blending with the anchor.
4. Fill floor, parking and block blanks from public data.
5. Fix the compressed parking gap by rebuilding anchors for each pass.

## Rules for new experiments
- Choose on Fold B, and read Fold A once at the end.
- Compare seed-averaged results, and ignore gaps under $1.
- Check any new feature's blank rate in fold rows against final rows.
- Never send addresses or rows from the supplied files to outside services.
