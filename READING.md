# Where we are

A plain summary for team 2 Bulbs 1 Source, written at about 12:35 pm Sydney on Saturday 7 November 2026.
Build time ends at 2:45 pm.

## The short version

We estimate the weekly rent of all 65,777 City of Sydney flats, with and without a car space.

On adverts we held back, our typical miss is **about $40 a week**, and about **80% land within $100**.
The Microburbs valuation missed by $86 on its own test. The two tests used different adverts, so this is a strong
sign, not a like-for-like win. Our real score will probably be worse than $40.

A valid file is ready: `rents_2-bulbs-1-source.csv`. It is deliberately not in git.

## How it works

1. **Find the flat's own last rent.** If it has none, use similar flats in its building, then its block or suburb.
2. **Bring that rent up to date** with a rent index built from flats that were leased more than once.
3. **Let a model correct it.** LightGBM learns how far to trust each of those anchors, given bedrooms, floor,
   location and how old the rent is.
4. **Make two prices.** The flat keeps the price matching what we know about its parking. The other price adds or
   subtracts the parking gap.

Step 2 is the clever part. Old rents run low simply because time passed.

## The journey

Median error in $/week. Fold A is June and July 2026 adverts, Fold B is April and May 2026.

| Step | Fold A | Fold B | What happened |
|---|---|---|---|
| 1. Local median | $100 | $95 | Suburb, bedrooms and parking, last 12 months |
| 2. Each flat's last rent | $65.20 | $79.19 | Much better, but stale, so about $50 low |
| 3. Rent index | $44.47 | $52.45 | The big win |
| 4. Leak fix | $44.61 | $52.10 | Fold A got slightly worse, and the result became believable |
| 5. LightGBM model | $41.53 | $43.12 | Learned how far to trust each anchor |
| 6. Cleaner inputs | $40.32 | $42.62 | Removed misleading fields and changed the target |

This is now the first section of `report.html`. New wins can be added as one more entry in `notes/report_text.json`.

## What we learned

- **A flat's own old rent is the best clue.** The typical one was about 20 months old, and once indexed it still
  missed by only about $41.
- **Time lag is why simple estimates run low.** The older the rent, the lower it runs: nothing for rents under
  6 months old, and $275 low for rents over 5 years old, before the index.
- **Projecting the trend forward did not help.** Holding the index flat at the cutoff did best on Fold B.
- **A car space is worth about $40 a week** (95% range $36 to $44), measured within buildings. By bedrooms it runs
  $12 for studios, $44 for one bedroom, $38 for two and $61 for three.
- **Parking value varies hugely by area,** from about $4 in Waterloo to about $232 in Barangaroo. Barangaroo has only
  81 flats behind it, so be careful with that number.
- **Each floor adds about $3.91 a week** (95% range $2.48 to $5.35).

## Can we trust it?

**Could the real Aug-Sep adverts be in our model?** We found no sign of it. Every dated field in the supplied files
ends on 31 July, the loader drops anything later, and nothing from the API is in the submission. We cannot prove how
undated fields like parking and bedrooms were built.

**Did our own tests leak?** Yes, and we fixed what we found:

- The input's parking and bedrooms were effectively copied from each flat's latest advert.
- The target advert's own parking was reaching the anchors.
- Two date fields were almost always blank in tests but filled in on the flats we predict.

Poisoning every row after the cutoff now moves no features.

**Why the real score may be worse:**

- About 25% of test adverts are first-time flats, and their input fields may still come from the advert itself.
- The mix of scored adverts may differ from our test adverts.
- Fold A has been run several times and its adverts were used in the leak check. Nothing was tuned on it, but it is
  no longer pristine.

## The awkward bit: parking

The measured premium is about $40, but the gap in our file has a median of about $17. The rule picks the model's gap
because it won by $9.60 on 85 adverts where parking changed. That is a small sample, and the model's gap is squeezed
because anchors stay fixed between the two prices. So say "$40" as a finding, not as what our CSV shows. The method
lane is testing a fix.

## Where the error is

- Flats with an earlier advert miss by about $40.
- First-time flats miss by about $56. They are the biggest opportunity.
- Rents older than 5 years still run about $51 low after the index.
- Parking and floor premiums may be too high, because we have no floor area or building age. Bigger, newer flats tend
  to have parking.

## What is running now

Results are not in yet, and all of these work on Fold B only.

| Lane | Question | Finish by |
|---|---|---|
| Public data | Can open data fill floor, parking and block blanks, or add floor area and building age? | 1:30 pm |
| API features | Are there extra adverts dated by 31 July? Do transport, schools, noise and risk help? | 1:30 pm |
| Methods | What is the noise floor? Do tuning, ensembles or CatBoost help? Can the parking gap be fixed? | 1:30 pm |
| Domain research | What drives flat rents and parking value, area by area? | 1:40 pm |
| Foundation models | Do TabPFN and similar models beat LightGBM, CatBoost and XGBoost? | 1:40 pm |

Only changes that clearly beat the noise get merged. Gaps under about $1 do not count. Fold A gets read once at the end.

## Rules we keep

- Use nothing dated after 31 July 2026. API calls for adverts must use `to_date=2026-07-31`.
- Undated market data from the API stays out, because it may be built from later adverts.
- Supplied files and row-level results stay out of git, and addresses never go to outside services.
- After the hackathon, delete the data and parquet files from Drive and empty the bin.

## What needs you

1. **Hand-in:** check how `rents_2-bulbs-1-source.csv` must be submitted. If it must be in the repo, that conflicts
   with our rule against committing row-level data.
2. **Repo visibility:** check whether it is public. `report.html` and `experiments.csv` are pushed, as aggregates only.
3. **Report:** read `report.html`, especially the surprising finding, which is a draft. Dark mode has not been checked.
4. **Videos:** `pitch.mp4` and `demo.mp4` are still placeholders. The script is in `PITCH_SCRIPT.filled.md`.
5. **Pitch wording:** don't call $40 against $86 a like-for-like win.
6. **Colab, optional:** `colab/eda.ipynb` is ready. Steps are in `colab/README.md`.

## Where to read more

| File | What is in it |
|---|---|
| `report.html` | The report for judges |
| `notes/METHODS_AND_NEXT_STEPS.md` | Everything tried, the leak audit, API safety and next steps |
| `DECISIONS.md` | Each choice made, and why |
| `PLAN.md` | How the work is split |
| `experiments.csv` | Every logged run |
