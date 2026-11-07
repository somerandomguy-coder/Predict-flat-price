# Parallel plan

Build ends 2:45 pm Sydney. `main` always runs. Each lane works in its own worktree and branch,
owns its own files, and merges into `main` when its tests pass.

## Shared code
- `src/`: shared library, used by both run folders.
- `local/`: scripts run on this machine.
- `colab/`: notebooks to upload to Google Colab, which send results back through Drive.

## Lanes
| Lane | Branch | Stages | Owns | Needs |
|---|---|---|---|---|
| Features | `lane/features` | 3, 4 | `src/index.py`, `src/features.py` | foundation |
| Model | `lane/model` | 5, 7 | `src/models.py`, `src/predict.py`, `local/run_model.py` | column contract below |
| Parking | `lane/parking` | 6 + floor curve | `src/parking.py`, `local/run_parking.py` | foundation |
| Colab | `lane/colab` | ablations, EDA | `colab/`, `scripts/export_features.py` | column contract below |
| Report | `lane/report` | deliverables | `src/report.py`, `README.md`, `PITCH_SCRIPT.md`, placeholders | `out/<mode>/metrics.json` |

Merge order: Features, Parking, Model, Colab, Report. After that, the real-data run is serial:
stage 2, then 3-4, 5, 6, 7, each one rewriting the submission.

## Column contract from `build_features`
- Anchors: `anchor_own`, `anchor_bldg`, `anchor_block`, each index-adjusted to `target_date`.
- For each anchor: `<anchor>_age_days` and `<anchor>_n`.
- `best_anchor`: the most specific anchor available, never NaN.
- `src.features.FEATURE_COLS`: the model's input list. Genuine NaN stays NaN.
- `car_space_resolved` is an input feature, so the model can run two passes (1 and 0).

## Rules for every lane
- Run on mock with `FLATS_MOCK=1`. Nothing from mock may reach `experiments.csv`, `notes/`, the report or the pitch.
- Training rows use rolling origins: pseudo-cutoffs before the fold cutoff, each predicting the next two months.
  A row's features never include its own advert.
- Do not tune on Fold A. Tune scalars on Fold B and report Fold A untouched.
- Commit notebooks with outputs cleared. Never print addresses or row-level data into a committed file.
- `experiments.csv` is append-only, with `merge=union` so branches merge cleanly.

## Remote teammate
Pull `main`, take a branch named `lane/<topic>`, and keep to files no lane above owns. Good open topics:
building orientation, nearby buildings blocking light or views, and NSW bond data as an index cross-check.
