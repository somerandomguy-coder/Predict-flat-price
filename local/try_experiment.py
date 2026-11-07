"""Try one idea on Fold B. Safe: never reads Fold A and never touches the submission.

Examples:
  .venv/bin/python local/try_experiment.py
  .venv/bin/python local/try_experiment.py --drop floor_level bathrooms
  .venv/bin/python local/try_experiment.py --leaves 31 --rounds 600 --seeds 3
"""
import argparse

from src import models as M
from src.predict import build_panel

ap = argparse.ArgumentParser()
ap.add_argument("--drop", nargs="*", default=[], help="features to remove")
ap.add_argument("--leaves", type=int, default=63)
ap.add_argument("--rounds", type=int, default=M.NUM_BOOST_ROUND)
ap.add_argument("--seeds", type=int, default=len(M.SEEDS), help="how many seeds to average")
ap.add_argument("--method", default="log_ratio", choices=M.METHODS)
args = ap.parse_args()

r = M.run_experiment(dict(name="try", description=f"Manual try: {vars(args)}", frame=build_panel(M.N_ORIGINS), methods=[args.method],
                          drop_features=args.drop, num_boost_round=args.rounds, seeds=M.SEEDS[:args.seeds],
                          lgb_params=dict(num_leaves=args.leaves)))
b = r["folds"]["B"]
print(f"Fold B: ${b['mae']:.2f} median error, {b['within100']:.1%} within $100, R2 {b['r2']:.3f}, bias {b['bias']:+.2f}")
print("Reference: $40.37, 81.9%, 0.881, +2.42. Gaps under about $1 are noise.")
