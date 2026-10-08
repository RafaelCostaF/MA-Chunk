#!/usr/bin/env python3
"""Large data files are stored as row-chunked parts (<45 MB each) so the repository stays within
GitHub's file-size limits. `join` rebuilds every <file>.parquet from <file>.parquet.parts/;
`split` creates the parts (maintainers only).

Usage (from the repository root):
    python scripts/data/split_join.py join
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
LIMIT = 45 * 1024 * 1024


def split():
    for f in sorted(DATA.rglob("*.parquet")):
        if f.stat().st_size <= LIMIT or ".parts" in str(f):
            continue
        df = pd.read_parquet(f)
        n = int(np.ceil(f.stat().st_size / LIMIT)) + 1
        parts = f.parent / f"{f.name}.parts"
        parts.mkdir(exist_ok=True)
        for i, idx in enumerate(np.array_split(np.arange(len(df)), n)):
            df.iloc[idx].to_parquet(parts / f"part_{i:03d}.parquet", index=False)
        print(f"split {f.relative_to(DATA)} into {n} parts")


def join():
    for parts in sorted(DATA.rglob("*.parquet.parts")):
        target = parts.parent / parts.name[: -len(".parts")]
        if target.exists():
            print(f"[skip] {target.relative_to(DATA)} exists")
            continue
        df = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("part_*.parquet"))], ignore_index=True)
        df.to_parquet(target, index=False)
        print(f"joined {target.relative_to(DATA)} ({len(df)} rows)")


if __name__ == "__main__":
    {"split": split, "join": join}[sys.argv[1] if len(sys.argv) > 1 else "join"]()
