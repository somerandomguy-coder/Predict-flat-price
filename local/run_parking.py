"""Stage 6: fit the parking premium and floor curve as of 31 July 2026, write the aggregates,
and check the premium on both folds against the stage 2 baseline's own with/without gap.

Stage 7 repeats that check with the model's two passes. Choose a method on Fold B, then report Fold A.
"""
import pandas as pd

from src import config as C
from src.features import build_features
from src.load import fold_test_rows
from src.parking import compare_premium_methods, fit_premium, fold_premium, parking_report, floor_report, write_reports

TAG = "MOCK DATA - " if C.MOCK else ""


def baseline_gap(fold: str) -> pd.DataFrame:
    X = build_features(C.FOLDS[fold]["cutoff"], fold_test_rows(fold))
    return X.assign(rent_with_car_space=X.base_with, rent_without_car_space=X.base_without)


def main():
    fit = fit_premium(C.AS_OF)
    paths = write_reports(fit)
    park, floor = parking_report(fit), floor_report(fit)
    print(f"{TAG}premium overall: {park['overall']}")
    print(f"{TAG}share of premium moving with rents: {fit.kappa:.2f}; between suburbs ${fit.tau_suburb:.0f}, "
          f"between buildings ${fit.tau_building:.0f}")
    print(f"{TAG}floor slope: ${floor['per_floor_premium']}/week per floor "
          f"({floor['per_floor_low']} to {floor['per_floor_high']}, n={floor['per_floor_n']})")
    for k in ("B", "A"):
        table = fit_premium(C.FOLDS[k]["cutoff"]).table
        d = baseline_gap(k)
        res = compare_premium_methods(d, fold_premium(k, d, table))
        print(f"\n{TAG}Fold {k}, baseline predictor against premium table")
        print(res[["method", "segment", "n", "mae", "d_mae", "within100", "bias", "median_gap"]].round(2).to_string(index=False))
    print("\nwrote", ", ".join(str(p.relative_to(C.ROOT)) for p in paths.values()))


if __name__ == "__main__":
    main()
