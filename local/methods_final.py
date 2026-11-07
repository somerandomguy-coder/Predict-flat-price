"""Finalists on the same five seeds as the reference, so a paired comparison measures the setting and not the seeds."""
import sys

from local import methods_lib as L
from local.methods_s4 import half_life

RUNS = {
    "f_r800": dict(rounds=800),
    "f_hl12": dict(weight_fn=half_life(12)),
    "f_n6": dict(n=6),
    "f_l2_10": dict(params=dict(lambda_l2=10.0)),
}


def main():
    panel = L.load_panel()
    for name in sys.argv[1:]:
        L.append_results("step3_final", L.run_lgb(panel, name, seeds=L.SEEDS, **RUNS[name]))


if __name__ == "__main__":
    main()
