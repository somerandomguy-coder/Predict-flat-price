"""Append new rows from a downloaded experiments_colab.csv to C.EXPERIMENTS_CSV.

Deduped by run_id and append-only: existing rows are never touched. In a real run, rows the
notebook marked as mock are skipped, because nothing from mock may reach experiments.csv.

Run: PYTHONPATH=. python scripts/merge_colab_results.py path/to/experiments_colab.csv
"""
import csv
import sys

from src import config as C
from src.validate import EXPERIMENT_COLS

MOCK_PREFIX = "MOCK:"


def read_rows(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        assert r.fieldnames == EXPERIMENT_COLS, f"{path}: columns are {r.fieldnames}, expected {EXPERIMENT_COLS}"
        return list(r)


def merge(src, dest=None) -> tuple:
    """Returns (added, skipped_duplicates, skipped_mock)."""
    dest = dest or C.EXPERIMENTS_CSV
    seen = {r["run_id"] for r in read_rows(dest)} if dest.exists() else set()
    added, dupes, mock = [], 0, 0
    for row in read_rows(src):
        if not C.MOCK and row["description"].startswith(MOCK_PREFIX):
            mock += 1
        elif row["run_id"] in seen:
            dupes += 1
        else:
            seen.add(row["run_id"])
            added.append(row)
    if added:
        new = not dest.exists()
        if not new:  # a file cut off mid-line would glue the first new row onto the last old one
            with open(dest, "rb") as f:
                f.seek(-1, 2)
                assert f.read(1) == b"\n", f"{dest} does not end in a newline"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, EXPERIMENT_COLS)
            if new:
                w.writeheader()
            w.writerows(added)
    return len(added), dupes, mock


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    a, d, m = merge(sys.argv[1])
    print(f"added {a}, skipped {d} duplicates, skipped {m} mock rows -> {C.EXPERIMENTS_CSV}")
