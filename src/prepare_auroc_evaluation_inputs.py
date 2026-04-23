#!/usr/bin/env python3
"""
Prepare evaluation input aliases for the main-cohort minimal release.

Usage:
1) Copy train_multimodal_main_cohort_model.py outputs (proba_train/proba_val/proba_test.csv)
   to notebook-compatible aliases
   (proba_train_72/proba_val_72/proba_test_72.csv).
2) Optionally copy ecg_multi.csv
   to the legacy-compatible alias ecg_muti.csv.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROBA_ALIASES = {
    "proba_train.csv": "proba_train_72.csv",
    "proba_val.csv": "proba_val_72.csv",
    "proba_test.csv": "proba_test_72.csv",
}


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare main-cohort evaluation aliases.")
    parser.add_argument(
        "--pred_dir",
        type=Path,
        default=Path("data"),
        help="Directory containing proba_train.csv, proba_val.csv, and proba_test.csv.",
    )
    parser.add_argument(
        "--table_csv",
        type=Path,
        default=None,
        help="Optional path to ecg_multi.csv to generate legacy alias ecg_muti.csv.",
    )
    args = parser.parse_args()

    pred_dir = args.pred_dir.resolve()
    print(f"[INFO] prediction dir: {pred_dir}")

    copied = 0
    for src_name, alias_name in PROBA_ALIASES.items():
        src = pred_dir / src_name
        dst = pred_dir / alias_name
        if copy_if_exists(src, dst):
            copied += 1
            print(f"[OK] {src.name} -> {dst.name}")
        else:
            print(f"[SKIP] missing: {src}")

    if args.table_csv is not None:
        table_csv = args.table_csv.resolve()
        alias_csv = table_csv.with_name("ecg_muti.csv")
        if copy_if_exists(table_csv, alias_csv):
            print(f"[OK] {table_csv.name} -> {alias_csv.name}")
        else:
            print(f"[SKIP] missing: {table_csv}")

    if copied == 0:
        print("[WARN] no proba csv files copied; please run training first.")
    else:
        print("[DONE] evaluation aliases are ready.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
