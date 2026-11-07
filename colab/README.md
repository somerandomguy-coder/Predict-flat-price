# Running on Google Colab

Two notebooks: `eda.ipynb` explores the raw files, and `colab_experiments.ipynb` runs model experiments from cached features.

## Data rules

- Keep the Drive folder **private**. Do not share it, the files in it, or a link to it.
- The data is for use on the day only. **After the hackathon, delete the CSVs and parquet files from Drive and empty the Drive bin.**
- Do not share a notebook that has outputs. Clear outputs before saving a copy anywhere.
- Commit notebooks with outputs cleared, and never commit data.

## One-time setup

1. In Google Drive, create a private folder at `My Drive/flats_private`. If you use another path, change `DRIVE_DIR` in the config cell of each notebook.
2. Upload these three files into it:
   - `rents-input-flats-city-of-sydney.csv`
   - `rents-past-leases-city-of-sydney.csv`
   - `blocks-homes-and-people-2021-2026.csv`

## EDA

1. Open [colab.research.google.com](https://colab.research.google.com), then File > Upload notebook, and pick `colab/eda.ipynb`.
2. Run the first three code cells, and allow Drive access when asked.
3. Check the table in section 0a. It shows which lease columns were picked for date and rent. Check that the median rent looks weekly.
4. If a pick is wrong, set it in `OVERRIDE` in the config cell and rerun from there.
5. Run all. Section 14 is for free exploration.

Rows after 2026-07-31 are counted, then dropped in section 0b, before any analysis.

To try it on mock data first, copy the three CSVs from `data/mock/` into `flats_private/mock/` and set `DEMO = True`.

## Experiments

1. On the local machine, after the real data is in `data/`, build and pack:
   ```
   PYTHONPATH=. .venv/bin/python scripts/export_features.py
   .venv/bin/python scripts/pack_colab.py
   ```
2. Upload `out/real/features_foldA.parquet` and `out/real/features_foldB.parquet` to `flats_private`.
3. Upload `colab/colab_experiments.ipynb` to Colab.
4. Edit `CONFIG`, then Run all. Each run appends one row to `flats_private/experiments_colab.csv`.
5. Tune on Fold B only. Fold A is the untouched report.
6. Download `experiments_colab.csv`, then merge it locally. Rerunning is safe, because rows are deduped by `run_id`:
   ```
   PYTHONPATH=. .venv/bin/python scripts/merge_colab_results.py ~/Downloads/experiments_colab.csv
   ```

Re-export and re-upload the parquet files whenever `src/features.py` changes.

### Using src/models.py

The notebook has its own LightGBM L1 experiment. Once `src/models.py` lands:

1. Rerun `scripts/pack_colab.py`.
2. Unzip `dist/colab_bundle.zip` into `flats_private/colab_bundle/`, so that `colab_bundle/src/models.py` exists.
3. Set `USE_SRC_MODELS = True`.

### Mock runs

`DEMO = True` reads and writes in `flats_private/mock/`, and the parquet files must come from `FLATS_MOCK=1`. The notebook stops if the data does not match the switch. Results are labelled `MOCK:`, and the merge script skips them in a real run.

## Running locally without Colab

Drive mounting is skipped off Colab. Set `FLATS_DEMO=1`, plus `FLATS_DATA_DIR` for the EDA or `FLATS_FEATURES_DIR` for the experiments, to point at mock files.

## After the hackathon

1. Delete the CSVs, parquet files and `experiments_colab.csv` from `flats_private`, including `mock/` and `colab_bundle/`.
2. Empty the Drive bin.
3. Delete any notebook copies with outputs from `My Drive/Colab Notebooks`.
