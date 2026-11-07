"""Stage 2: suburb x bedrooms x parking median. Scores both folds and writes the first submission."""
import pandas as pd

from src import config as C
from src.features import build_features, flats_asof
from src.load import fold_test_rows
from src.predict import write_submission
from src.validate import log_experiment, score


def predict(cutoff, rows):
    X = build_features(cutoff, rows)
    return X.assign(rent_with_car_space=X[["base_with", "base_without"]].max(axis=1), rent_without_car_space=X.base_without)


def main():
    res = {}
    for k, f in C.FOLDS.items():
        t = fold_test_rows(k)
        res[k] = score(predict(f["cutoff"], t))
        print(f"Fold {k}:", {a: round(b, 3) for a, b in res[k].items()})
    rid = log_experiment("Stage 2 baseline: 12-month suburb x bed x parking median", "suburb,bedrooms,car_space",
                         "hier_median", res["A"], res["B"])
    flats = flats_asof(C.AS_OF)
    rows = flats[["gnaf_id", "car_space"]].assign(target_date=C.SCORE_MID, car_space_resolved=flats.car_space)
    p = predict(C.AS_OF, rows)
    sub = write_submission(p[["gnaf_id", "rent_with_car_space", "rent_without_car_space"]])
    print(f"run {rid}; wrote {C.SUBMISSION_CSV.relative_to(C.ROOT)} rows={len(sub):,}")
    if C.MOCK:
        tr = pd.read_csv(C.MOCK_TRUTH_CSV, dtype={"gnaf_id": "string"}).merge(sub, on="gnaf_id")
        print("Mock frozen score:", {a: round(b, 3) for a, b in score(tr).items()})


if __name__ == "__main__":
    main()
