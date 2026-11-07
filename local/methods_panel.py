"""Cache a long panel for the method comparison. Fold A rows are dropped before anything is saved."""
import time

from src import config as C
from src import predict as P

N_ORIGINS = 30  # 24 training origins behind the earliest inner window, plus Fold A and Fold B


def main():
    started = time.time()
    panel = P.build_panel(N_ORIGINS)
    panel = panel[panel.origin <= C.FOLDS["B"]["cutoff"]].reset_index(drop=True)
    assert panel.target_date.max() <= C.FOLDS["B"]["end"]
    panel.attrs = {}
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(C.OUT_DIR / "methods_panel.parquet")
    print(len(panel), panel.origin.nunique(), panel.origin.min(), panel.target_date.max(), round(time.time() - started))
    print(panel.groupby("origin").size().to_string())
    print(list(panel.columns))


if __name__ == "__main__":
    main()
