"""
scripts/extend_events.py -- add MORE validation/calibration events (round 2) to an existing events JSON.

Same objective rules as select_events_v2.py (SPC storm reports in the crop, spacing >= 5 days from EVERY existing
event including test, <= 3 per month, seeded random, optional S3 verification). Existing records are copied
unchanged; new records get "round": 2. Test events are never added or changed. The output is a new file, so
the old events_v2.json stays valid.

  python scripts/extend_events.py --events_in configs/events_v2.json --out configs/events_v3.json \
     --spc_cache /content/spc_cache --verify_s3
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)

import select_events_v2 as S  # noqa: E402

PLAN = [("val", ["high", "high", "medium", "low"]), ("cal", ["high", "medium", "medium"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events_in", required=True)
    ap.add_argument("--out", default="configs/events_v3.json")
    ap.add_argument("--spc_cache", default="spc_cache")
    ap.add_argument("--date_from", default="2025-04-15")
    ap.add_argument("--date_to", default="2026-10-15")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--min_sep_days", type=int, default=5)
    ap.add_argument("--max_per_month", type=int, default=3)
    ap.add_argument("--verify_s3", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    raw = json.load(open(args.events_in))
    events = raw["events"] if isinstance(raw, dict) else raw
    if any(e.get("round") == 2 for e in events):
        sys.exit("events_in already contains round-2 events; refusing to extend twice.")
    crop = raw.get("crop", {"lat": [25.0, 35.0], "lon": [-95.0, -80.0]}) if isinstance(raw, dict) else \
        {"lat": [25.0, 35.0], "lon": [-95.0, -80.0]}
    bounds = (crop["lat"][0], crop["lat"][1], crop["lon"][0], crop["lon"][1])
    taken = [dt.datetime.strptime(e["start"], S.TIME_FMT).date() for e in events]

    d0, d1 = dt.date.fromisoformat(args.date_from), dt.date.fromisoformat(args.date_to)
    days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        texts = list(ex.map(lambda d: S.fetch_spc_day(d, args.spc_cache), days))
    ok = [(d, t) for d, t in zip(days, texts) if t]
    print(f"SPC days available: {len(ok)}/{len(days)}")
    if len(ok) < 0.5 * len(days):
        sys.exit("Too many SPC days failed to download.")
    table = [S.day_summary(d, t, bounds) for d, t in ok]

    wanted = {}
    for _, strata in PLAN:
        for s in strata:
            wanted[s] = wanted.get(s, 0) + 1
    ok_fn = S.s3_has_scans if args.verify_s3 else (lambda c: True)
    chosen = S.choose_candidates(table, wanted, taken, args.seed, args.min_sep_days, args.max_per_month, ok_fn)
    new = S.assign_splits(chosen, PLAN)
    for e in new:
        e["round"] = 2
    allev = sorted(events + new, key=lambda e: e["start"])
    payload = dict(raw) if isinstance(raw, dict) else {}
    payload["events"] = allev
    payload["extension"] = {"seed": args.seed, "plan": PLAN,
                            "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}: {len(allev)} events ({len(new)} new)")
    print(f"{'event_id':<18}{'split':<7}{'stratum':<9}{'reports':>8}  window (UTC)")
    for e in allev:
        tag = " <- NEW" if e.get("round") == 2 else ""
        print(f"{e['event_id']:<18}{e['split']:<7}{e['stratum']:<9}{e.get('spc_reports_in_crop', '-'):>8}  {e['start']} -> {e['end']}{tag}")
    print("\nNew-event ids for build_cache_s3.py --only_events:", ",".join(e["event_id"] for e in new))


if __name__ == "__main__":
    main()
