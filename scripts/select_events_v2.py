"""
scripts/select_events_v2.py
Objective, reproducible choice of new 6-hour training/validation/calibration/test events.

Selection uses NOAA SPC storm reports (tornado/wind/hail) inside the study crop as an
INDEPENDENT proxy for convective activity. It never looks at imagery or model results, so
choosing test events this way cannot leak information from the model.

Pipeline
  1. fetch SPC daily report files (cached on disk; resumable) for the date range
  2. per day: count reports inside the crop, find the densest 6-hour UTC window
  3. bucket days into strata (high / medium / low / quiet) by report count
  4. seeded random picks per stratum with spacing + per-month constraints
  5. (optional) verify the NOAA bucket really has the scans, else take the next candidate
  6. assign splits with a fixed plan; write configs/events_v2.json (existing v1 events included)

If SPC cannot be reached, pass --dates 2025-05-18,2025-06-03,... (manual list) instead.

Example (Colab, needs internet)
  python scripts/select_events_v2.py --v1_manifest outputs/dataset_splits/splits_manifest.json \
      --out configs/events_v2.json --spc_cache /content/spc_cache --verify_s3
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TIME_FMT = "%Y-%m-%d %H:%M:%S"
STRATA = [("high", 40), ("medium", 10), ("low", 1), ("quiet", 0)]   # (name, min reports in crop)
# new events only (v1 events keep their existing splits): 12 = 7 train + 1 val + 2 cal + 2 test
SPLIT_PLAN: List[Tuple[str, List[str]]] = [
    ("test", ["high", "medium"]),
    ("cal", ["high", "medium"]),
    ("val", ["high"]),
    ("train", ["high", "medium", "medium", "low", "low", "quiet", "quiet"]),
]
KIND_BY_SECOND_COL = {"F_Scale": "tornado", "Speed": "wind", "Size": "hail"}


# ----------------------------------------------------------------------------- SPC parsing
def parse_spc_csv(text: str) -> List[Dict[str, Any]]:
    """SPC daily report file: three sections (tornado/wind/hail), each starting 'Time,...'."""
    reports, cols, kind = [], None, "other"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Time,"):
            cols = [c.strip() for c in line.split(",")]
            kind = KIND_BY_SECOND_COL.get(cols[1] if len(cols) > 1 else "", "other")
            continue
        if cols is None:
            continue
        try:
            parts = next(csv.reader([line]))
            row = dict(zip(cols, parts))
            hhmm = int(row["Time"])
            lat, lon = float(row["Lat"]), float(row["Lon"])
        except (KeyError, ValueError, StopIteration):
            continue
        if not (0 <= hhmm <= 2359 and hhmm % 100 < 60):
            continue
        reports.append({"hhmm": hhmm, "lat": lat, "lon": lon, "kind": kind})
    return reports


def report_datetime(day: dt.date, hhmm: int) -> dt.datetime:
    """Convective day = 12Z..12Z: reports before 12Z belong to the next calendar date."""
    h, m = divmod(hhmm, 100)
    base = dt.datetime(day.year, day.month, day.day, h, m)
    return base + dt.timedelta(days=1) if h < 12 else base


def in_bounds(r: Dict[str, Any], bounds: Tuple[float, float, float, float]) -> bool:
    lat_min, lat_max, lon_min, lon_max = bounds
    return lat_min <= r["lat"] <= lat_max and lon_min <= r["lon"] <= lon_max


def best_window(day: dt.date, reports: Sequence[Dict[str, Any]], hours: int = 6,
                default_start_hour: int = 19) -> Tuple[dt.datetime, int]:
    """Start of the densest `hours`-long UTC window (whole hours, 14Z..02Z+1) and its count."""
    t0 = dt.datetime(day.year, day.month, day.day, 14)
    times = [report_datetime(day, r["hhmm"]) for r in reports]
    best_s, best_n = dt.datetime(day.year, day.month, day.day, default_start_hour), 0
    for k in range(0, 13):                                  # starts 14Z .. 02Z(+1)
        s = t0 + dt.timedelta(hours=k)
        n = sum(1 for t in times if s <= t < s + dt.timedelta(hours=hours))
        if n > best_n:
            best_s, best_n = s, n
    return best_s, best_n


def stratum_of(n_reports: int) -> str:
    for name, lo in STRATA:
        if n_reports >= lo and (lo > 0 or n_reports == 0):
            return name
    return "quiet"


def day_summary(day: dt.date, text: str, bounds, hours: int = 6) -> Dict[str, Any]:
    reps = [r for r in parse_spc_csv(text) if in_bounds(r, bounds)]
    start, n_win = best_window(day, reps, hours)
    return {"date": day.isoformat(), "n_reports": len(reps), "n_in_window": n_win,
            "start": start.strftime(TIME_FMT),
            "end": (start + dt.timedelta(hours=hours) - dt.timedelta(seconds=1)).strftime(TIME_FMT),
            "stratum": stratum_of(len(reps))}


# ----------------------------------------------------------------------------- selection
def _month_key(d: dt.date) -> Tuple[int, int]:
    return (d.year, d.month)


def choose_candidates(
    table: Sequence[Dict[str, Any]],
    wanted: Dict[str, int],
    taken_dates: Sequence[dt.date],
    seed: int,
    min_sep_days: int = 5,
    max_per_month: int = 3,
    ok_fn: Callable[[Dict[str, Any]], bool] = lambda c: True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Seeded random picks per stratum honouring spacing and per-month caps."""
    rng = random.Random(seed)
    chosen: Dict[str, List[Dict[str, Any]]] = {k: [] for k in wanted}
    taken = list(taken_dates)
    month_count: Dict[Tuple[int, int], int] = {}
    for d in taken:
        month_count[_month_key(d)] = month_count.get(_month_key(d), 0) + 1
    for stratum in ("high", "medium", "low", "quiet"):          # rarest first
        if stratum not in wanted:
            continue
        pool = [c for c in table if c["stratum"] == stratum]
        rng.shuffle(pool)
        for c in pool:
            if len(chosen[stratum]) >= wanted[stratum]:
                break
            d = dt.date.fromisoformat(c["date"])
            if any(abs((d - t).days) < min_sep_days for t in taken):
                continue
            if month_count.get(_month_key(d), 0) >= max_per_month:
                continue
            if not ok_fn(c):
                continue
            chosen[stratum].append(c)
            taken.append(d)
            month_count[_month_key(d)] = month_count.get(_month_key(d), 0) + 1
    return chosen


def assign_splits(chosen: Dict[str, List[Dict[str, Any]]],
                  plan: Sequence[Tuple[str, List[str]]] = SPLIT_PLAN) -> List[Dict[str, Any]]:
    pools = {k: list(v) for k, v in chosen.items()}
    out = []
    for split, strata in plan:
        for s in strata:
            if not pools.get(s):
                raise RuntimeError(f"Not enough '{s}' candidates for split '{split}'. "
                                   f"Widen the date range or lower --min_sep_days.")
            c = pools[s].pop(0)
            out.append({"event_id": f"event_{c['start'][:10]}", "split": split, "start": c["start"],
                        "end": c["end"], "stratum": s, "spc_reports_in_crop": c["n_reports"],
                        "spc_reports_in_window": c["n_in_window"], "source": "spc_v2"})
    return out


def v1_events_from_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        m = json.load(f)
    spans: Dict[str, Dict[str, Any]] = {}
    for split, wins in m["splits"].items():
        for w in wins:
            times = [dt.datetime.strptime(x["datetime"].replace(" UTC", "").strip(), TIME_FMT)
                     for x in w["source_files"]["input"] + w["source_files"]["target"]]
            g = w["group_id"]
            lo, hi = min(times), max(times)
            if g in spans:
                lo, hi = min(lo, spans[g]["lo"]), max(hi, spans[g]["hi"])
            spans[g] = {"split": split, "lo": lo, "hi": hi}
    out = []
    for g, s in sorted(spans.items(), key=lambda kv: kv[1]["lo"]):
        start = s["lo"].replace(minute=0, second=0)
        out.append({"event_id": g, "split": s["split"], "start": start.strftime(TIME_FMT),
                    "end": (start + dt.timedelta(hours=6) - dt.timedelta(seconds=1)).strftime(TIME_FMT),
                    "stratum": "v1", "source": "v1"})
    return out


# ----------------------------------------------------------------------------- network
def fetch_spc_day(day: dt.date, cache_dir: str, timeout: float = 30.0) -> Optional[str]:
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"{day:%y%m%d}_rpts.csv")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return open(p, "r", errors="replace").read()
    for suffix in ("_rpts.csv", "_rpts_filtered.csv"):
        url = f"https://www.spc.noaa.gov/climo/reports/{day:%y%m%d}{suffix}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "meteor-net-insat3ds/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                text = r.read().decode("utf-8", errors="replace")
            if "Time," in text:
                with open(p, "w") as f:
                    f.write(text)
                return text
        except Exception:  # noqa: BLE001
            continue
    return None


def s3_has_scans(cand: Dict[str, Any], channels=(9, 13, 14, 15), need: int = 66) -> bool:
    from src.data.goes_s3 import group_scans, list_keys, list_window_keys
    s = dt.datetime.strptime(cand["start"], TIME_FMT)
    e = dt.datetime.strptime(cand["end"], TIME_FMT)
    try:
        return len(group_scans(list_window_keys(s, e, list_keys), channels, s, e)) >= need
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1_manifest", required=True)
    ap.add_argument("--out", default="configs/events_v2.json")
    ap.add_argument("--spc_cache", default="spc_cache")
    ap.add_argument("--date_from", default="2025-04-15")
    ap.add_argument("--date_to", default="2026-10-15")
    ap.add_argument("--lat_min", type=float, default=25.0)
    ap.add_argument("--lat_max", type=float, default=35.0)
    ap.add_argument("--lon_min", type=float, default=-95.0)
    ap.add_argument("--lon_max", type=float, default=-80.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min_sep_days", type=int, default=5)
    ap.add_argument("--max_per_month", type=int, default=3)
    ap.add_argument("--verify_s3", action="store_true")
    ap.add_argument("--dates", default="", help="manual comma list YYYY-MM-DD (skips SPC; 19Z-01Z window)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    bounds = (args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    v1 = v1_events_from_manifest(args.v1_manifest)
    taken = [dt.datetime.strptime(e["start"], TIME_FMT).date() for e in v1]

    if args.dates:
        table = []
        for s in args.dates.split(","):
            d = dt.date.fromisoformat(s.strip())
            st = dt.datetime(d.year, d.month, d.day, 19)
            table.append({"date": d.isoformat(), "n_reports": -1, "n_in_window": -1,
                          "start": st.strftime(TIME_FMT),
                          "end": (st + dt.timedelta(hours=6) - dt.timedelta(seconds=1)).strftime(TIME_FMT),
                          "stratum": "manual"})
        print("Manual mode: copy these windows into configs/events_v2.json (add event_id/split by hand):")
        print(json.dumps(table, indent=1))
        return

    d0, d1 = dt.date.fromisoformat(args.date_from), dt.date.fromisoformat(args.date_to)
    days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    print(f"Fetching SPC reports for {len(days)} days (cached in {args.spc_cache}) ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        texts = list(ex.map(lambda d: fetch_spc_day(d, args.spc_cache), days))
    ok = [(d, t) for d, t in zip(days, texts) if t]
    print(f"  got {len(ok)}/{len(days)} days")
    if len(ok) < 0.5 * len(days):
        sys.exit("Too many SPC days failed to download. Check internet, or use --dates.")
    table = [day_summary(d, t, bounds) for d, t in ok]
    by_s = {n: sum(1 for r in table if r["stratum"] == n) for n, _ in STRATA}
    print("  day counts by stratum:", by_s)

    wanted: Dict[str, int] = {}
    for _, strata in SPLIT_PLAN:
        for s in strata:
            wanted[s] = wanted.get(s, 0) + 1
    ok_fn = s3_has_scans if args.verify_s3 else (lambda c: True)
    chosen = choose_candidates(table, wanted, taken, args.seed, args.min_sep_days, args.max_per_month, ok_fn)
    new_events = assign_splits(chosen)

    allev = sorted(v1 + new_events, key=lambda e: e["start"])
    payload = {
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed, "crop": {"lat": [args.lat_min, args.lat_max], "lon": [args.lon_min, args.lon_max]},
        "selection": "SPC storm reports in crop (independent of imagery/model); strata high>=40, medium 10-39, low 1-9, quiet 0",
        "events": allev,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}: {len(allev)} events")
    print(f"{'event_id':<18}{'split':<7}{'stratum':<9}{'reports':>8}  window (UTC)")
    for e in allev:
        print(f"{e['event_id']:<18}{e['split']:<7}{e['stratum']:<9}{e.get('spc_reports_in_crop', '-'):>8}  {e['start']} -> {e['end']}")


if __name__ == "__main__":
    main()
