# What will this flat rent for?

**Team 2 Bulbs 1 Source · Microburbs Hackathon, Task 4**

We estimate the weekly asking rent of all 65,777 City of Sydney flats, with and without a car space, as at 31 July 2026.

| Deliverable | File |
|---|---|
| Submission | [`submission/rents_2-bulbs-1-source.csv`](submission/rents_2-bulbs-1-source.csv) |
| Report | [`report.html`](report.html) |
| 45-second pitch | [`pitch.mp4`](pitch.mp4) |
| Demo | [`demo.mp4`](demo.mp4) |

## Result

Held-out test on 2,635 adverts from June and July 2026, for the submitted file:

| Median error | Within $100 | R² | Bias |
|---|---|---|---|
| **$40 a week** | **82.47%** | **0.902** | **$0** |

Before rounding to $10, the blend's median error is $38.02. A suburb, bedrooms and parking median scores $100 on the same test. Microburbs reports $86 on different adverts, so that comparison is indicative only.

## Who it helps
- **Tenants** can check whether an asking rent looks fair, and what a car space adds.
- **Landlords** can set a rent from evidence, and see what a space could earn on its own.
- **Buyers and investors** can compare likely rent before buying.

## The clever step
An old rent reads low when the market is rising. We take the closest past rent for each flat (its own, its building's, then its block's) and bring it up to date with an index built from flats leased more than once.

We also show the model the raw last rent, because many re-lets return at exactly the old rent. A nine-model blend then decides how far to trust each clue.

Every test predicts two months ahead using only what was known at the time. We also found that input parking and bedrooms track each flat's latest advert, so we rebuilt them as known at each cutoff.

## Data
Microburbs supplied the flat list, past lease adverts, and block homes and people file. **They are not in this repo.** The submission CSV is the only row-level file here, as the task requires.

## How to run
Put the supplied files in `data/`, then:
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python local/run_model.py    # LightGBM model and submission
PYTHONPATH=. .venv/bin/python -m src.report         # report.html
```
The submitted blend comes from `local/zoo/stack.py`, using the jobs in `slurm/`. To explore on Google Colab, see [`colab/README.md`](colab/README.md).

## Sources
Used in the submitted model:
- Microburbs supplied files.

Tested in experiments, not in the final model:
- Microburbs API: adverts dated by 31 July 2026, transport, schools, amenities, risks and market series.
- G-NAF, Geoscape, Open G-NAF licence.
- ABS Census 2021 and mesh block counts, CC BY 4.0.
- OpenStreetMap, ODbL, © OpenStreetMap contributors.
- NSW cadastre, NSW Spatial Services, CC BY 3.0 AU.
- City of Sydney Development Monitoring, CC BY 4.0.

Pitch voice by ElevenLabs. Research sources are in [`notes/domain_research.md`](notes/domain_research.md).

## More detail
Decisions, experiments and research are in [`notes/`](notes/). Start with [`notes/READING.md`](notes/READING.md).
