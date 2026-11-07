"""Step 3: a small hyperparameter search on Fold B and the inner windows.

Three seeds per config here, then finalists are rerun on ten seeds with `confirm`, so a pick is not a lucky seed.
Rounds come free from checkpoints on one fit. One parameter moves at a time, then a few combinations.
"""
import sys

from local import methods_lib as L

SEARCH_SEEDS = L.SEEDS[:3]
TEN_SEEDS = L.SEEDS + (202, 303, 404, 505, 606)
CHECKS = [200, 400, 800]
SLOW_CHECKS = [400, 800, 1200, 2000]
GRID = {
    "base": ({}, CHECKS),
    "leaves15": (dict(num_leaves=15), CHECKS),
    "leaves31": (dict(num_leaves=31), CHECKS),
    "leaves127": (dict(num_leaves=127), [200, 400]),
    "min50": (dict(min_data_in_leaf=50), CHECKS),
    "min100": (dict(min_data_in_leaf=100), CHECKS),
    "min200": (dict(min_data_in_leaf=200), CHECKS),
    "l2_10": (dict(lambda_l2=10.0), CHECKS),
    "ff0.5": (dict(feature_fraction=0.5), CHECKS),
    "bag0.5": (dict(bagging_fraction=0.5), CHECKS),
    "lr0.1": (dict(learning_rate=0.1), [100, 200, 400]),
    "lr0.02": (dict(learning_rate=0.02), SLOW_CHECKS),
    "small_lr02": (dict(learning_rate=0.02, num_leaves=31, min_data_in_leaf=50), SLOW_CHECKS),
    "reg_lr02": (dict(learning_rate=0.02, num_leaves=31, min_data_in_leaf=100, lambda_l2=10.0, feature_fraction=0.5), SLOW_CHECKS),
}


def main():
    panel = L.load_panel()
    for name, (params, checks) in GRID.items():
        rows = L.run_lgb(panel, f"s3_{name}", params=params, seeds=SEARCH_SEEDS, checkpoints=checks)
        for r in rows:
            r["params"] = params
        L.append_results("step3", rows)


def confirm(finalists: dict):
    """finalists: name -> (params, rounds). Ten seeds on every window."""
    panel = L.load_panel()
    for name, (params, rounds) in finalists.items():
        rows = L.run_lgb(panel, f"s3c_{name}", params=params, rounds=rounds, seeds=TEN_SEEDS)
        for r in rows:
            r["params"] = params
        L.append_results("step3_confirm", rows)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        confirm(eval(sys.argv[1]))
    else:
        main()
