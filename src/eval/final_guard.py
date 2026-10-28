"""src/eval/final_guard.py -- one place that decides which cache/split an evaluation script may read.

Normal mode : reads the DEV cache, split 'val'; refuses a cache whose cache_index.json says cache_role == 'test'.
--final_test: reads the TEST cache (--test_cache_dir, must be role 'test'), split 'test'; the dev cache is only used
              for things that must come from calibration events. Every final run is appended to
              outputs/FINAL_TEST_LOG.txt so accidental repeats are visible.
"""
from __future__ import annotations

import datetime
import json
import os
import sys


def _role(path: str):
    idx = os.path.join(path, "cache_index.json")
    return json.load(open(idx)).get("cache_role") if os.path.exists(idx) else None


def resolve_eval_cache(args, script: str):
    if not getattr(args, "final_test", False):
        if _role(args.cache_dir) == "test":
            sys.exit("REFUSED: cache_dir is a TEST cache. Use --final_test --test_cache_dir for the final run.")
        return args.cache_dir, "val"
    if not getattr(args, "test_cache_dir", None):
        sys.exit("--final_test needs --test_cache_dir")
    if _role(args.test_cache_dir) != "test":
        sys.exit("REFUSED: --test_cache_dir is not marked cache_role == 'test'.")
    if _role(args.cache_dir) == "test":
        sys.exit("REFUSED: --cache_dir (dev) must not be the test cache.")
    os.makedirs("outputs", exist_ok=True)
    line = f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} {script} FINAL TEST\n"
    with open(os.path.join("outputs", "FINAL_TEST_LOG.txt"), "a") as f:
        f.write(line)
    print("=" * 70 + f"\nFINAL TEST RUN ({script}). Do not tune anything on these numbers.\n" + "=" * 70)
    return args.test_cache_dir, "test"
