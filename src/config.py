"""Paths and dates shared by every lane. Set FLATS_MOCK=1 to run on synthetic data.

Mock mode writes only under out/mock/, so a synthetic result can never be mistaken
for a real submission or land in the real experiment log.
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MOCK = os.environ.get("FLATS_MOCK") == "1"
TEAM_SLUG = os.environ.get("TEAM_SLUG", "TEAM_SLUG")

DATA_DIR = Path(os.environ.get("FLATS_DATA_DIR", ROOT / "data" / ("mock" if MOCK else "")))
OUT_DIR = ROOT / "out" / ("mock" if MOCK else "real")
INPUT_CSV = DATA_DIR / "rents-input-flats-city-of-sydney.csv"
LEASES_CSV = DATA_DIR / "rents-past-leases-city-of-sydney.csv"
BLOCKS_CSV = DATA_DIR / "blocks-homes-and-people-2021-2026.csv"
MOCK_TRUTH_CSV = DATA_DIR / "_frozen_truth.csv"

SUBMISSION_CSV = OUT_DIR / f"rents_{TEAM_SLUG}.csv" if MOCK else ROOT / f"rents_{TEAM_SLUG}.csv"
EXPERIMENTS_CSV = OUT_DIR / "experiments.csv" if MOCK else ROOT / "experiments.csv"
PROFILE_MD = OUT_DIR / "data_profile.md" if MOCK else ROOT / "notes" / "data_profile.md"

AS_OF = pd.Timestamp("2026-07-31")
SCORE_START, SCORE_END = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-30")
SCORE_MID = pd.Timestamp("2026-08-31")

FOLDS = {
    "A": dict(cutoff=pd.Timestamp("2026-05-31"), start=pd.Timestamp("2026-06-01"), end=pd.Timestamp("2026-07-31")),
    "B": dict(cutoff=pd.Timestamp("2026-03-31"), start=pd.Timestamp("2026-04-01"), end=pd.Timestamp("2026-05-31")),
}

N_INPUT = 65_777
