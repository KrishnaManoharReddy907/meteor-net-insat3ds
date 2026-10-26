"""Tests for Step C: S3 helpers, event selector, cache builder (fake I/O), normalization fit."""
import datetime as dt
import json
import os
import sys
import tempfile
import unittest

# On Windows, open memory-mapped numpy arrays prevent tempfile rmtree cleanup
if os.name == "nt":
    _orig_td = tempfile.TemporaryDirectory
    class _SafeTD(_orig_td):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("ignore_cleanup_errors", True)
            super().__init__(*args, **kwargs)
    tempfile.TemporaryDirectory = _SafeTD

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import build_cache_s3 as bc  # noqa: E402
import fit_normalization_v2 as fn  # noqa: E402
import select_events_v2 as se  # noqa: E402
from src.data import goes_s3 as g3  # noqa: E402
from src.data.fast_cache import decode_kelvin, load_event_cache  # noqa: E402

CH = (9, 13, 14, 15)


def key(band, t, mode=6):
    s = t.strftime("%Y%j%H%M%S") + "3"
    e = (t + dt.timedelta(minutes=2, seconds=40)).strftime("%Y%j%H%M%S") + "9"
    c = (t + dt.timedelta(minutes=3)).strftime("%Y%j%H%M%S") + "4"
    return f"ABI-L2-CMIPC/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/OR_ABI-L2-CMIPC-M{mode}C{band:02d}_G19_s{s}_e{e}_c{c}.nc"


T0 = dt.datetime(2026, 5, 20, 18, 1, 17)


def all_keys(n=72, bands=CH, mode=6, skip=()):
    return [key(b, T0 + dt.timedelta(minutes=5 * i), mode) for i in range(n) if i not in skip for b in bands]


class TestS3Helpers(unittest.TestCase):
    def test_parse_key_and_regex(self):
        k = "ABI-L2-CMIPC/2026/001/02/OR_ABI-L2-CMIPC-M6C13_G19_s20260010201193_e20260010203579_c20260010204034.nc"
        p = g3.parse_key(k)
        self.assertEqual((p["mode"], p["band"]), (6, 13))
        self.assertEqual(p["datetime"], dt.datetime(2026, 1, 1, 2, 1, 19))
        self.assertIsNone(g3.parse_key("ABI-L2-CMIPF/x/OR_ABI-L2-CMIPF-M6C13_G19_s1.nc"))

    def test_hours_covering_crosses_midnight_and_year(self):
        h = g3.hours_covering(dt.datetime(2025, 12, 31, 22, 30), dt.datetime(2026, 1, 1, 1, 10))
        self.assertEqual(h, [(2025, 365, 22), (2025, 365, 23), (2026, 1, 0), (2026, 1, 1)])

    def test_listing_xml_pagination_fields(self):
        xml = (b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>true</IsTruncated>'
               b'<NextContinuationToken>abc</NextContinuationToken><Contents><Key>a.nc</Key></Contents>'
               b'<Contents><Key>b.nc</Key></Contents></ListBucketResult>')
        keys, trunc, tok = g3.parse_listing_xml(xml)
        self.assertEqual((keys, trunc, tok), (["a.nc", "b.nc"], True, "abc"))

    def test_list_keys_follows_continuation(self):
        pages = [
            b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>true</IsTruncated>'
            b'<NextContinuationToken>T2</NextContinuationToken><Contents><Key>k1</Key></Contents></ListBucketResult>',
            b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
            b'<Contents><Key>k2</Key></Contents></ListBucketResult>']
        calls = []

        def fake_get(url, timeout):
            calls.append(url)
            return pages[len(calls) - 1]
        self.assertEqual(g3.list_keys("p/", http_get=fake_get), ["k1", "k2"])
        self.assertIn("continuation-token=T2", calls[1])

    def test_group_scans_complete_only_window_and_modes(self):
        keys = all_keys(10, skip=(3,)) + [key(13, T0 + dt.timedelta(minutes=15))]   # extra dup band-13 file
        # remove band 14 from scan 5 -> incomplete
        t5 = T0 + dt.timedelta(minutes=25)
        keys = [k for k in keys if not (g3.parse_key(k)["band"] == 14 and g3.parse_key(k)["datetime"] == t5)]
        scans = g3.group_scans(keys, CH, T0, T0 + dt.timedelta(minutes=40))
        times = [s["datetime"] for s in scans]
        self.assertNotIn(T0 + dt.timedelta(minutes=15), times)    # scan 3 skipped entirely
        self.assertNotIn(t5, times)                               # incomplete scan dropped
        self.assertEqual(len(scans), 7)                           # 9 present -1 incomplete, window cuts nothing else
        mixed = g3.group_scans(all_keys(2, mode=3), CH)
        self.assertEqual(mixed[0]["modes"], [3])                  # other scan modes accepted


class TestSelector(unittest.TestCase):
    SPC = ("Time,F_Scale,Location,County,State,Lat,Lon,Comments\n"
           "2005,UNK,2 N A,B,AL,30.5,-88.0,tornado, with comma\n"
           "Time,Speed,Location,County,State,Lat,Lon,Comments\n"
           "2010,60,x,y,MS,31.0,-89.0,wind\n"
           "0130,55,x,y,LA,29.0,-91.0,after midnight\n"
           "1900,50,x,y,TX,32.0,-99.0,outside crop\n"
           "Time,Size,Location,County,State,Lat,Lon,Comments\n"
           "2020,100,x,y,GA,33.0,-84.0,hail\n"
           "garbage line\n")

    def test_parse_and_kinds(self):
        r = se.parse_spc_csv(self.SPC)
        self.assertEqual([x["kind"] for x in r], ["tornado", "wind", "wind", "wind", "hail"])
        self.assertEqual(len(r), 5)

    def test_convective_day_rollover_and_window(self):
        day = dt.date(2025, 5, 18)
        self.assertEqual(se.report_datetime(day, 130), dt.datetime(2025, 5, 19, 1, 30))
        self.assertEqual(se.report_datetime(day, 2010), dt.datetime(2025, 5, 18, 20, 10))
        s = se.day_summary(day, self.SPC, (25, 35, -95, -80))
        self.assertEqual(s["n_reports"], 4)                       # TX report excluded
        st = dt.datetime.strptime(s["start"], se.TIME_FMT)
        self.assertTrue(st <= dt.datetime(2025, 5, 18, 20, 5) < st + dt.timedelta(hours=6))
        self.assertEqual(s["stratum"], "low")
        q = se.day_summary(day, "Time,Speed,Location,County,State,Lat,Lon,Comments\n", (25, 35, -95, -80))
        self.assertEqual((q["stratum"], q["start"][11:13]), ("quiet", "19"))

    def test_strata_boundaries(self):
        self.assertEqual([se.stratum_of(n) for n in (0, 1, 9, 10, 39, 40, 400)],
                         ["quiet", "low", "low", "medium", "medium", "high", "high"])

    def _table(self):
        rng = np.random.default_rng(0)
        out = []
        for i in range(300):
            d = dt.date(2025, 4, 15) + dt.timedelta(days=i)
            n = int(rng.choice([0, 3, 15, 60]))
            out.append({"date": d.isoformat(), "n_reports": n, "n_in_window": n, "stratum": se.stratum_of(n),
                        "start": f"{d} 19:00:00", "end": f"{d + dt.timedelta(days=1)} 00:59:59"})
        return out

    def test_choose_and_assign_constraints_and_determinism(self):
        wanted = {}
        for _, ss in se.SPLIT_PLAN:
            for s in ss:
                wanted[s] = wanted.get(s, 0) + 1
        taken = [dt.date(2025, 5, 20)]
        a = se.choose_candidates(self._table(), wanted, taken, seed=1)
        b = se.choose_candidates(self._table(), wanted, taken, seed=1)
        self.assertEqual(a, b)
        ev = se.assign_splits(a)
        self.assertEqual(len(ev), 12)
        counts = {s: sum(1 for e in ev if e["split"] == s) for s in ("train", "val", "cal", "test")}
        self.assertEqual(counts, {"train": 7, "val": 1, "cal": 2, "test": 2})
        dates = sorted([dt.date.fromisoformat(e["event_id"][6:]) for e in ev] + taken)
        self.assertTrue(all((y - x).days >= 5 for x, y in zip(dates, dates[1:])))
        months = {}
        for d in dates:
            months[(d.year, d.month)] = months.get((d.year, d.month), 0) + 1
        self.assertLessEqual(max(months.values()), 3)
        for sp in ("val", "cal", "test"):                          # every eval split has a strong-convection event
            self.assertTrue(any(e["split"] == sp and e["stratum"] == "high" for e in ev))
        self.assertEqual(len({e["event_id"] for e in ev}), 12)

    def test_ok_fn_replaces_unavailable_days(self):
        wanted = {"high": 2}
        t = [c for c in self._table() if c["stratum"] == "high"]
        first = se.choose_candidates(t, wanted, [], seed=3)["high"]
        banned = {first[0]["date"]}
        second = se.choose_candidates(t, wanted, [], seed=3, ok_fn=lambda c: c["date"] not in banned)["high"]
        self.assertNotIn(first[0]["date"], [c["date"] for c in second])
        self.assertEqual(len(second), 2)

    def test_v1_manifest_conversion(self):
        def win(g, a, b):
            f = lambda t: {"datetime": t + " UTC"}
            return {"group_id": g, "source_files": {"input": [f(a)], "target": [f(b)]}}
        m = {"splits": {"train": [win("event_X", "2026-01-01 02:01:19", "2026-01-01 07:56:19")],
                        "test": [win("event_T", "2026-09-07 16:01:17", "2026-09-07 21:56:17")]}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(m, f)
        ev = se.v1_events_from_manifest(f.name)
        os.unlink(f.name)
        self.assertEqual([e["split"] for e in ev], ["train", "test"])
        self.assertEqual(ev[0]["start"], "2026-01-01 02:00:00")
        self.assertEqual(ev[0]["end"], "2026-01-01 07:59:59")


def fake_env(tmpdir, n_scans=72, skip=(), fail_key=None):
    keys = all_keys(n_scans, skip=skip)
    store = {"downloads": 0}

    def list_fn(prefix):
        return [k for k in keys if k.startswith(prefix)]

    def download_fn(k, dest):
        if fail_key and fail_key in k:
            raise IOError("boom")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(b"x" * 10)
        store["downloads"] += 1
        return 10

    def read_fn(paths):
        # frame value encodes scan time: 250 + minutes since T0 / 5 * 0.1 ; one NaN pixel in every frame
        t = g3.parse_key(os.path.basename(paths[13]))
        minutes = (t["datetime"] - T0).total_seconds() / 60.0
        fr = np.full((4, 20, 24), 250.0 + minutes / 5.0 * 0.1, dtype=np.float32)
        fr[:, 0, 0] = np.nan
        return fr
    return list_fn, download_fn, read_fn, store


def spec(split="train", eid="event_2026-05-20"):
    return {"event_id": eid, "split": split, "start_dt": T0.replace(minute=0, second=0),
            "end_dt": T0.replace(minute=0, second=0) + dt.timedelta(hours=6) - dt.timedelta(seconds=1),
            "start": "", "end": ""}


class TestBuilder(unittest.TestCase):
    CROP = {"crop_shape": [20, 24]}

    def test_build_event_end_to_end_with_fakes(self):
        with tempfile.TemporaryDirectory() as d:
            cache, work = os.path.join(d, "c"), os.path.join(d, "w")
            l, dl, rd, st = fake_env(d)
            r = bc.build_event(spec(), cache, work, CH, self.CROP, l, dl, rd, workers=4, log=lambda *_: None)
            self.assertEqual(r["status"], "built")
            self.assertEqual(st["downloads"], 72 * 4)
            arr, meta = load_event_cache(cache, "event_2026-05-20", verify_hash=True)
            self.assertEqual(arr.shape, (72, 4, 20, 24))
            k, valid = decode_kelvin(arr[10])
            self.assertFalse(valid[0, 0, 0])                           # NaN preserved as invalid
            self.assertAlmostEqual(float(k[0, 5, 5]), 250.0 + 10 * 0.1, places=2)   # scan 10 -> right frame
            self.assertFalse(os.path.exists(os.path.join(work, "event_2026-05-20")))  # raw deleted
            self.assertTrue(os.path.exists(os.path.join(cache, "event_2026-05-20.sources.json")))
            self.assertEqual(bc.read_index(cache)["events"]["event_2026-05-20"]["split"], "train")
            again = bc.build_event(spec(), cache, work, CH, self.CROP, l, dl, rd, log=lambda *_: None)
            self.assertEqual(again["status"], "exists")

    def test_insufficient_and_failed_downloads_do_not_write_cache(self):
        with tempfile.TemporaryDirectory() as d:
            cache, work = os.path.join(d, "c"), os.path.join(d, "w")
            l, dl, rd, _ = fake_env(d, n_scans=40)
            r = bc.build_event(spec(), cache, work, CH, self.CROP, l, dl, rd, log=lambda *_: None)
            self.assertEqual(r["status"], "insufficient")
            l, dl, rd, _ = fake_env(d, fail_key="C14")
            r = bc.build_event(spec(), cache, work, CH, self.CROP, l, dl, rd, log=lambda *_: None)
            self.assertEqual(r["status"], "download_failed")
            self.assertFalse(os.path.exists(bc.array_path(cache, "event_2026-05-20")))

    def test_dry_run_downloads_nothing_and_reports_irregular_steps(self):
        with tempfile.TemporaryDirectory() as d:
            l, dl, rd, st = fake_env(d, skip=(20, 21))
            r = bc.build_event(spec(), os.path.join(d, "c"), os.path.join(d, "w"), CH, self.CROP, l, dl, rd,
                               dry_run=True, log=lambda *_: None)
            self.assertEqual((r["status"], r["n_scans"], r["irregular_steps"]), ("dry_run_ok", 70, 1))
            self.assertEqual(st["downloads"], 0)

    def test_cache_role_guard(self):
        with tempfile.TemporaryDirectory() as d:
            dev = [{"event_id": "a", "split": "train"}, {"event_id": "b", "split": "cal"}]
            tst = [{"event_id": "t", "split": "test"}]
            self.assertEqual(bc.enforce_cache_role(d, dev), "dev")
            with self.assertRaises(SystemExit):
                bc.enforce_cache_role(d, tst)                           # test into a dev folder
            with self.assertRaises(SystemExit):
                bc.enforce_cache_role(os.path.join(d, "x"), dev + tst)  # mixed request
            self.assertEqual(bc.enforce_cache_role(os.path.join(d, "t"), tst), "test")
            with self.assertRaises(SystemExit):
                bc.enforce_cache_role(os.path.join(d, "t"), dev)

    def test_reindex_and_sync(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "c")
            l, dl, rd, _ = fake_env(d)
            bc.build_event(spec(), cache, os.path.join(d, "w"), CH, self.CROP, l, dl, rd, log=lambda *_: None)
            os.remove(bc.index_path(cache))
            ej = os.path.join(d, "ev.json")
            json.dump({"events": [{"event_id": "event_2026-05-20", "split": "val", "start": "", "end": ""}]}, open(ej, "w"))
            sys.argv = ["x", "--cache_dir", cache, "--events_json", ej, "--out", os.path.join(d, "s2.json")]
            with self.assertRaises(SystemExit):                          # no train events -> nothing to fit
                fn.main()

    def test_refuses_test_cache(self):
        with tempfile.TemporaryDirectory() as d:
            tc = os.path.join(d, "t")
            bc.enforce_cache_role(tc, [{"event_id": "t", "split": "test"}])
            ej = os.path.join(d, "ev.json")
            json.dump({"events": [{"event_id": "t", "split": "train", "start": "", "end": ""}]}, open(ej, "w"))
            sys.argv = ["x", "--cache_dir", tc, "--events_json", ej, "--out", os.path.join(d, "s.json")]
            with self.assertRaises(SystemExit):
                fn.main()


if __name__ == "__main__":
    unittest.main()
