"""Stages 5 and 7: pick the method on Fold B, report Fold A once, refit at AS_OF, write the submission."""
import argparse
import time

import pandas as pd

from src import config as C
from src import models as M
from src.predict import build_panel, final_rows, load_premium_table, premium_for, write_submission
from src.validate import score


def rounded(d: dict) -> dict:
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items() if not isinstance(v, dict)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-origins", type=int, default=M.N_ORIGINS)
    ap.add_argument("--rounds", type=int, default=M.NUM_BOOST_ROUND)
    args = ap.parse_args()
    started = time.time()

    panel = build_panel(args.n_origins)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(C.OUT_DIR / "features_panel.parquet")
    print(f"panel: {len(panel):,} rows, {panel.origin.nunique()} origins, {time.time() - started:.0f}s")

    cfg = dict(name="stage5", frame=panel, n_origins=args.n_origins, num_boost_round=args.rounds, report_fold_a=True,
               description=f"Stage 5/7 LightGBM L1, {args.n_origins} rolling origins, method picked on Fold B",
               out_path=C.OUT_DIR / "metrics.json")
    res = M.run_experiment(cfg)
    for k, v in res["method_comparison"]["B"].items():
        print(f"Fold B {k}:", rounded(v))
    print(f"winner: {res['winner']} ({res['feature_source']}, {len(res['features'])} features)")
    print("Fold A:", rounded(res["folds"]["A"]))
    print("Fold A baseline:", rounded(res["baseline_fold_a"]))

    panel = M.load_panel(cfg)
    train = M.train_rows(panel, C.AS_OF, args.n_origins)
    model = M.fit_model(train, res["features"], res["winner"], cfg)
    X = final_rows()
    passes = M.predict_two_pass(model, X)
    table, source = load_premium_table()
    prices = M.compose_prices(passes, X.input_car_space, premium_for(X, table))
    pred = pd.concat([X[["gnaf_id"]], prices], axis=1)
    # Blanks fall back to the suburb x bed x parking median, then write_submission's overall median.
    baseline = X[["gnaf_id"]].assign(rent_with_car_space=X[["base_with", "base_without"]].max(axis=1),
                                     rent_without_car_space=X.base_without)
    sub = write_submission(pred, fallback=baseline)

    gap = sub.rent_with_car_space - sub.rent_without_car_space
    res["final"] = dict(premium_source=source, n_rows=len(sub), n_train=len(train),
                        n_blank_before_fallback=int(prices.isna().any(axis=1).sum()),
                        premium_mean=float(gap.mean()), premium_median=float(gap.median()),
                        premium_p10=float(gap.quantile(.1)), premium_p90=float(gap.quantile(.9)),
                        two_pass_negative_share=float((passes.premium < 0).mean()),
                        rent_without_median=float(sub.rent_without_car_space.median()))
    model.save(C.OUT_DIR / "model")
    M.write_json(M.shap_summary(model, X), C.OUT_DIR / "shap.json")
    if C.MOCK:
        tr = pd.read_csv(C.MOCK_TRUTH_CSV, dtype={"gnaf_id": "string"}).merge(sub, on="gnaf_id")
        res["final"]["mock_frozen_score"] = score(tr)
        print("Mock frozen score:", rounded(res["final"]["mock_frozen_score"]))
    res["total_runtime_s"] = round(time.time() - started, 1)
    M.write_json(res, C.OUT_DIR / "metrics.json")
    print("final:", rounded(res["final"]))
    print(f"run {res['run_id']}; wrote {C.SUBMISSION_CSV.relative_to(C.ROOT)} rows={len(sub):,} in {res['total_runtime_s']}s")


if __name__ == "__main__":
    main()
