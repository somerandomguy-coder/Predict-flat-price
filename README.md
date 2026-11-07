# Predict flat rents

For a plain summary of where the project stands, read [READING.md](READING.md).

We estimate the weekly asking rent for all 65,777 City of Sydney flats, with and without parking, as at 31 July 2026.

## Who it helps
- **Tenants** can check whether an advertised rent looks fair, and what a car space should add.
- **Landlords** can set a rent from evidence, not from an old lease.
- **Buyers and investors** can see the likely rent of a flat before they buy it, and what parking or a higher floor is worth.

## Data
- Flat list, past lease adverts, and the mesh block homes and people file, supplied by Microburbs and the hackathon.
- G-NAF address identifiers and ABS mesh blocks.
- NSW rental bond data, only if a run records that it was used. Then: © State of New South Wales. For current information go to www.nsw.gov.au.

**Supplied data and derived rows are not in this repo, and must never be committed.** That covers supplied files, API output, keys, and any row-level result, including `rents_*.csv`. See `.gitignore`. Only code and totals or averages are committed.

## The clever step
An old rent is a low rent when the market is rising. A naive estimate copies the last rent seen, so it runs low. We take the closest past rent for each flat (its own, then its building, then its block) and move it forward to the target date with a rent index. The model works from that adjusted rent, not the stale one.

Testing copies real use. Each test predicts two months ahead using only what was known at the time, so no flat can see its own advert. We tune on Fold B and never tune on Fold A, though it has been run several times and its adverts were used in our leak check.

## How to run

### Local
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the supplied files in data/
PYTHONPATH=. .venv/bin/python local/run_baseline.py
PYTHONPATH=. .venv/bin/python -m src.report           # writes report.html
PYTHONPATH=. .venv/bin/python -m src.report --pitch   # writes PITCH_SCRIPT.filled.md
```
Set `TEAM_SLUG` to name the submission file.

To try it without supplied data, make the mock data and set `FLATS_MOCK=1`. Everything then goes to `out/mock/`, and the report carries a large synthetic banner:
```
PYTHONPATH=. FLATS_MOCK=1 .venv/bin/python local/make_mock.py
PYTHONPATH=. FLATS_MOCK=1 .venv/bin/python local/run_baseline.py
PYTHONPATH=. FLATS_MOCK=1 .venv/bin/python -m src.report
PYTHONPATH=. FLATS_MOCK=1 .venv/bin/python -m src.report --fixture   # layout test, fake numbers
```

### Colab
Upload a notebook from `colab/` to Google Colab. Keep the supplied data in your own Drive and set `FLATS_DATA_DIR` to that folder. Results come back through Drive. Clear notebook outputs before committing.

## Report inputs
`src/report.py` reads aggregate JSON from `out/real/`. The expected shapes are in `INPUTS` at the top of that file. A missing input shows as pending. A real run stops if any input is not stamped real, so lanes should write `"provenance": src.report.provenance()` into each file. Hand-written text goes in `notes/report_text.json`.

See `PLAN.md` for how the work is split and `DECISIONS.md` for choices made along the way.
