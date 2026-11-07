# Decisions

Choices made without asking, so the team can overrule them.

1. **Layout.** `src/` is shared, `local/` and `colab/` are the two run folders. The Colab notebook lives in `colab/`, not `notebooks/`, because the later instruction asked for two folders.
2. **Mock isolation.** Mock runs need `FLATS_MOCK=1` and write only to `out/mock/`. The code never falls back to mock silently.
3. **Mock lease schema is a guess.** The real schema is unknown. `src/load.py` maps likely names and stops with an error on anything else.
4. **Rows after as-of are dropped in `load_leases`,** so every later step inherits the filter.
5. **Input columns can leak.** `ever_advertised_for_lease` is rebuilt from leases dated by the cutoff. `attrs_last_seen` and `first_ever_listed` are masked if later than the cutoff. This also applies to the final fit.
6. **Training uses rolling origins.** `build_features` only sees rows before a pseudo-cutoff, so each training row mimics the two-month horizon and cannot see its own rent.
7. **"MAE" in `experiments.csv` means median absolute error,** the rank metric. Bias is signed (prediction minus advert) and comes from Fold A.
8. **`rents_*.csv` is gitignored,** following "never commit derived row-level data". The task page says to "add" this file to the hand-in, so check how it must be submitted before the deadline.
9. **NSW bond data counts by publication date, not data month.** The July 2026 file was uploaded in August, so the latest usable file is June 2026. Folds need the same lag.
10. **Microburbs API is not called yet.** It bills per call from a prepaid wallet and adverts must be date-filtered by us. Any bulk pull needs a cost check first.
11. **First commits went straight to `main`,** because the repo was empty and lanes need a base. Lanes then use branches.
