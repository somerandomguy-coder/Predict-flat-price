"""Stage 5: LightGBM (L1) on rolling-origin rows, driven by a config dict.

This module works from a cached feature frame only. It imports nothing that reads the
raw CSVs, so a Colab notebook needs just src/models.py, src/validate.py (and the
src/config.py that validate uses) plus a parquet file from src.predict.build_panel.

The panel has one row per advert, built at its `origin`, the pseudo-cutoff two months
or less before the advert. Month-end origins are shared by every fold, so one panel
serves all of them: a fold's test rows are the rows whose origin is the fold cutoff, and
its training rows are the n origins before that.
"""
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .validate import log_experiment, score

STEP_MONTHS = 2
N_ORIGINS = 12
METHODS = ("log_ratio", "direct")
PRED_FLOOR = 100.0  # weekly dollars; matches the lowest rent load_leases accepts
LGB_PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=20, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, cat_smooth=10, verbosity=-1, num_threads=4,
                  deterministic=True, force_row_wise=True)
NUM_BOOST_ROUND = 400  # fixed: early stopping would need a test window

LABELS = {"rent", "lease_date"}
# Stand-ins for the flat's parking. Left in, they would absorb the effect that the two passes
# measure through car_space_resolved, so the premium would come out near zero.
PARKING_PROXIES = {"car_space", "car_spaces", "advert_car_space", "input_car_space"}
NON_FEATURES = LABELS | PARKING_PROXIES | {"gnaf_id", "target_date", "origin", "data_mode", "base_suburb", "building_key", "mb", "address",
                                           "building_address", "attrs_last_seen", "first_ever_listed"}
CATEGORICAL = ("suburb", "property_type")
NO_SCORE = dict(mae=float("nan"), within100=float("nan"), r2=float("nan"), bias=float("nan"))
METHOD_LABELS = dict(log_ratio="LightGBM on log(rent / anchor)", direct="LightGBM on rent")
FEATURE_LABELS = dict(car_space_resolved="Car space", base_with="Local median with parking",
                      base_without="Local median without parking", best_anchor="Best past-rent anchor",
                      flats_in_building="Flats in building", floor_level="Floor level")


def provenance() -> dict:
    """The stamp the report needs on every file. It is built here so Colab does not need src/report.py."""
    return dict(mode="mock" if C.MOCK else "real", generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def ensure_best_anchor(df: pd.DataFrame) -> pd.DataFrame:
    """Shim until the anchors merge: without a best_anchor column, use base_without."""
    return df if "best_anchor" in df.columns else df.assign(best_anchor=df["base_without"])


def resolve_features(df: pd.DataFrame, cfg: dict | None = None) -> tuple[list[str], str]:
    """Feature list and where it came from: config, then the panel's FEATURE_COLS, then the columns on hand."""
    cfg = cfg or {}
    if cfg.get("feature_cols"):
        cols, source = list(cfg["feature_cols"]), "config"
    elif df.attrs.get("feature_cols"):
        cols, source = list(df.attrs["feature_cols"]), "FEATURE_COLS"
    else:
        cols = [c for c in df.columns if c not in NON_FEATURES
                and (c in CATEGORICAL or pd.api.types.is_numeric_dtype(df[c]))]
        source = "available_columns_shim"
    if LABELS & set(cols):
        raise ValueError(f"label in feature list: {sorted(LABELS & set(cols))}")
    drop = PARKING_PROXIES | set(cfg.get("drop_features", ()))
    cols = [c for c in cols if c not in drop]
    if "car_space_resolved" not in cols:
        cols.append("car_space_resolved")  # the two passes have nothing to change without it
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns not in the frame: {missing}")
    return cols, source


def _matrix(df: pd.DataFrame, cols: list[str], categories: dict) -> pd.DataFrame:
    """Model input. NaN stays NaN; a category unseen in training becomes NaN."""
    X = pd.DataFrame(index=df.index)
    for c in cols:
        if c in categories:
            v = df[c].astype(object)
            X[c] = pd.Categorical(v.where(v.isin(categories[c])), categories=categories[c])
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return X


@dataclass
class Model:
    booster: lgb.Booster
    method: str
    cols: list
    categories: dict

    def predict(self, df: pd.DataFrame, car_space=None) -> np.ndarray:
        """Predicted weekly rent. car_space overrides car_space_resolved on every row."""
        X = _matrix(df, self.cols, self.categories)
        if car_space is not None:
            X["car_space_resolved"] = float(car_space)
        raw = self.booster.predict(X)
        return df["best_anchor"].to_numpy(float) * np.exp(raw) if self.method == "log_ratio" else raw

    def save(self, stem: Path) -> None:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(stem.with_suffix(".txt")))
        stem.with_suffix(".json").write_text(json.dumps(
            dict(provenance=provenance(), method=self.method, cols=self.cols, categories=self.categories), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, stem: Path) -> "Model":
        stem = Path(stem)
        meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        meta.pop("provenance", None)
        return cls(lgb.Booster(model_file=str(stem.with_suffix(".txt"))), **meta)


def fit_model(train: pd.DataFrame, cols: list[str], method: str, cfg: dict | None = None) -> Model:
    cfg = cfg or {}
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    train = ensure_best_anchor(train)
    if method == "log_ratio":
        train = train[train.best_anchor > 0]
        y = np.log(train.rent.to_numpy(float) / train.best_anchor.to_numpy(float))
    else:
        y = train.rent.to_numpy(float)
    categories = {c: sorted(train[c].dropna().astype(str).unique()) for c in cols
                  if c in CATEGORICAL or not pd.api.types.is_numeric_dtype(train[c])}
    params = {**LGB_PARAMS, "seed": cfg.get("seed", 7), **cfg.get("lgb_params", {})}
    data = lgb.Dataset(_matrix(train, cols, categories), y, categorical_feature=list(categories) or "auto")
    booster = lgb.train(params, data, num_boost_round=cfg.get("num_boost_round", NUM_BOOST_ROUND))
    return Model(booster, method, list(cols), categories)


def predict_two_pass(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """Both passes and the model's premium. The premium is raw here and can be negative."""
    w, wo = model.predict(df, car_space=1), model.predict(df, car_space=0)
    return pd.DataFrame({"pred_with": w, "pred_without": wo, "premium": w - wo}, index=df.index)


def compose_prices(passes: pd.DataFrame, car_space, premium=None) -> pd.DataFrame:
    """Stage 7 rule. A flat with known parking keeps its own pass and the other column is that minus or plus the
    premium; an unknown flat keeps both passes. premium overrides the two-pass difference where it is not NaN.
    Either way it is floored at $0.
    """
    prem = passes["premium"]
    if premium is not None:
        prem = pd.Series(premium, index=passes.index, dtype=float).fillna(prem)
    prem = prem.clip(lower=0)
    car = pd.Series(car_space, index=passes.index, dtype=float)
    w, wo = passes["pred_with"], passes["pred_without"]
    without = (w - prem).where(car > 0, wo).clip(lower=PRED_FLOOR)
    with_ = (wo + prem).where(car == 0, w)
    return pd.DataFrame({"rent_with_car_space": np.maximum(with_, without), "rent_without_car_space": without})


def origins_for(cutoff, n: int) -> list[pd.Timestamp]:
    """The n month-end pseudo-cutoffs before cutoff, each one step back."""
    cutoff = pd.Timestamp(cutoff)
    assert cutoff.is_month_end, f"{cutoff} is not a month end"
    return [cutoff + pd.offsets.MonthEnd(-STEP_MONTHS * k) for k in range(1, n + 1)]


def train_rows(panel: pd.DataFrame, cutoff, n: int) -> pd.DataFrame:
    t = panel[panel.origin.isin(origins_for(cutoff, n))]
    if t.empty:
        raise ValueError(f"no training rows before {cutoff}")
    assert t.target_date.max() <= pd.Timestamp(cutoff), "a training label is later than the cutoff"
    return t


def fold_rows(panel: pd.DataFrame, fold: str) -> pd.DataFrame:
    f = C.FOLDS[fold]
    return panel[(panel.origin == f["cutoff"]) & panel.target_date.between(f["start"], f["end"])]


def evaluate(model: Model, test: pd.DataFrame) -> dict:
    """Score as the submission will be made: prices use input parking, which is all that is known beforehand."""
    prices = compose_prices(predict_two_pass(model, test), test["input_car_space"])
    return score(pd.concat([test[["rent", "advert_car_space", "input_car_space"]], prices], axis=1))


def baseline_score(test: pd.DataFrame) -> dict:
    return score(test.assign(rent_with_car_space=test[["base_with", "base_without"]].max(axis=1),
                             rent_without_car_space=test["base_without"]))


def suburb_score(test: pd.DataFrame) -> dict:
    return score(test.assign(rent_with_car_space=test["base_suburb"], rent_without_car_space=test["base_suburb"]))


def report_methods(panel: pd.DataFrame, compared: dict, winner: str, winner_a: dict | None) -> list[dict]:
    """The report's method table. Losing methods hold Fold B only: they are never scored on Fold A."""
    folds = {"B": fold_rows(panel, "B")} | ({"A": fold_rows(panel, "A")} if winner_a else {})
    out = []
    if "base_suburb" in panel.columns:
        out.append(dict(key="suburb_median", label="Suburb median", folds={k: suburb_score(t) for k, t in folds.items()}))
    out.append(dict(key="baseline", label="Suburb x bedroom x parking median",
                    folds={k: baseline_score(t) for k, t in folds.items()}))
    out.append(dict(key="final", label=METHOD_LABELS[winner],
                    folds=dict(B=compared[winner]) | (dict(A=winner_a) if winner_a else {})))
    out += [dict(key=m, label=METHOD_LABELS[m], folds=dict(B=s)) for m, s in compared.items() if m != winner]
    return out


def pick_winner(results: dict) -> str:
    """Lowest median absolute error, then highest share within $100."""
    return min(results, key=lambda m: (results[m]["mae"], -results[m]["within100"]))


def _check_mode(panel: pd.DataFrame) -> None:
    """A mock panel must not reach the real log, however the file travelled."""
    if "data_mode" in panel.columns:
        here = "mock" if C.MOCK else "real"
        found = set(panel["data_mode"].unique())
        if found != {here}:
            raise ValueError(f"panel holds {sorted(found)} rows but this run is {here}")


def load_panel(cfg: dict) -> pd.DataFrame:
    panel = cfg["frame"] if cfg.get("frame") is not None else pd.read_parquet(cfg["features_path"])
    _check_mode(panel)
    attrs = dict(panel.attrs)
    panel = ensure_best_anchor(panel)
    panel = panel.assign(origin=pd.to_datetime(panel.origin), target_date=pd.to_datetime(panel.target_date))
    panel.attrs.update(attrs)
    return panel


def write_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def run_experiment(config: dict) -> dict:
    """Train and score from a cached panel, then write and log the result.

    config keys:
      frame or features_path   the panel, as a DataFrame or a parquet path (required)
      name, description        labels; name also sets the default output file
      n_origins                training origins per fold (default 12)
      methods                  subset of METHODS to compare on Fold B
      feature_cols             overrides the panel's list; drop_features removes from it
      lgb_params, num_boost_round, seed
      report_fold_a            default False. Ablations stay on Fold B; only the final run sets this
      out_path                 default out/<mode>/ablations/<name>.json
      log                      default True, appends to experiments.csv through log_experiment

    Fold A cannot affect any choice: the winner is settled on Fold B before Fold A is read, and only
    the winner is ever scored on it.
    """
    started = time.time()
    name = config.get("name", "experiment")
    panel = load_panel(config)
    cols, source = resolve_features(panel, config)
    n = config.get("n_origins", N_ORIGINS)
    methods = list(config.get("methods", METHODS))

    fold_b = fold_rows(panel, "B")
    train_b = train_rows(panel, C.FOLDS["B"]["cutoff"], n)
    compared = {m: evaluate(fit_model(train_b, cols, m, config), fold_b) for m in methods}
    winner = pick_winner(compared)

    target = "the ratio of rent to the best past-rent anchor" if winner == "log_ratio" else "weekly rent directly"
    result = dict(provenance=provenance(), sources=dict(nsw_bond_used=False),
                  method_summary=f"LightGBM with an L1 loss predicts {target}, trained on {n} rolling two-month "
                                 "windows, with the method chosen on Fold B.",
                  name=name, mode="mock" if C.MOCK else "real", winner=winner,
                  selection="lowest Fold B median abs error, then highest within-$100",
                  feature_source=source, features=cols, n_origins=n,
                  n_train=dict(B=len(train_b)), folds=dict(B=compared[winner]),
                  method_comparison=dict(B={**compared, "baseline_hier_median": baseline_score(fold_b)}),
                  fold_a_reported=bool(config.get("report_fold_a")))
    if config.get("report_fold_a"):
        fold_a = fold_rows(panel, "A")
        train_a = train_rows(panel, C.FOLDS["A"]["cutoff"], n)
        result["n_train"]["A"] = len(train_a)
        result["folds"]["A"] = evaluate(fit_model(train_a, cols, winner, config), fold_a)
        result["baseline_fold_a"] = baseline_score(fold_a)
    result["methods"] = report_methods(panel, compared, winner, result["folds"].get("A"))
    result["runtime_s"] = round(time.time() - started, 1)

    if config.get("log", True):
        result["run_id"] = log_experiment(
            config.get("description", name), f"{source}:{len(cols)} cols", f"lgbm_l1:{winner}:origins={n}",
            result["folds"].get("A", NO_SCORE), result["folds"]["B"])
    write_json(result, config.get("out_path") or C.OUT_DIR / "ablations" / f"{name}.json")
    return result


def shap_summary(model: Model, df: pd.DataFrame, n: int = 5000, seed: int = 7) -> dict:
    """Exact TreeSHAP from LightGBM, reduced to per-feature aggregates. No row-level values leave this function."""
    s = df.sample(min(n, len(df)), random_state=seed)
    contrib = model.booster.predict(_matrix(s, model.cols, model.categories), pred_contrib=True)
    vals = contrib[:, :-1]
    mean_abs = np.abs(vals).mean(axis=0)
    return dict(provenance=provenance(), method=model.method, n_rows=len(s),
                unit="log(rent / anchor)" if model.method == "log_ratio" else "$/week",
                top_features=[dict(feature=model.cols[i], mean_abs_shap=float(mean_abs[i]),
                                   label=FEATURE_LABELS.get(model.cols[i], model.cols[i].replace("_", " ").capitalize()))
                              for i in np.argsort(-mean_abs)])
