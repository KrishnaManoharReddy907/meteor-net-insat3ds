"""Tests for src/data/fast_cache.py and src/data/fast_dataset.py (synthetic data only)."""
import datetime
import os
import tempfile
import unittest

import numpy as np

from src.data.fast_cache import (
    SENTINEL, decode_kelvin, encode_kelvin, load_event_cache, write_event_cache,
)
from src.data.fast_dataset import FastSequenceStore

# On Windows, open memory-mapped numpy arrays prevent tempfile rmtree cleanup
if os.name == "nt":
    _orig_td = tempfile.TemporaryDirectory
    class _SafeTD(_orig_td):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("ignore_cleanup_errors", True)
            super().__init__(*args, **kwargs)
    tempfile.TemporaryDirectory = _SafeTD

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

T0 = datetime.datetime(2026, 5, 20, 18, 1, 19)
H = W = 192          # -> patch grid: rows/cols at 0, 64 -> 2x2 = 4 patches of 128
CH = (9, 13, 14, 15)


def make_event(cache_dir, eid="ev", T=40, drop_index=None, invalid=None, cold_target=None, base=250.0):
    """Frame t has constant Kelvin = base + 0.5*t on all channels (so frame identity is readable)."""
    stamps = [T0 + datetime.timedelta(minutes=5 * i) for i in range(T)]
    if drop_index is not None:  # simulate a missing observation: remove the timestamp entirely
        stamps = [s for i, s in enumerate(stamps) if i != drop_index]
    n = len(stamps)
    orig = [i for i in range(T) if i != drop_index]

    def frame_fn(i):
        fr = np.full((4, H, W), base + 0.5 * orig[i], dtype=np.float32)
        if invalid and orig[i] == invalid[0]:
            fr[:, invalid[1]:invalid[1] + 2, invalid[2]:invalid[2] + 2] = np.nan
        if cold_target and orig[i] == cold_target[0]:
            fr[1, 0:40, 0:40] = 200.0   # B13 deep-convective block in patch 0
        return fr

    return write_event_cache(cache_dir, eid, stamps, CH, {"crop_shape": [H, W]}, frame_fn, (4, H, W))


class TestEncoding(unittest.TestCase):
    def test_roundtrip_and_sentinel(self):
        k = np.array([[180.123, 250.0, np.nan, 320.999]], dtype=np.float32)
        raw = encode_kelvin(k)
        dec, valid = decode_kelvin(raw)
        self.assertEqual(raw.dtype, np.uint16)
        self.assertEqual(raw[0, 2], SENTINEL)
        self.assertEqual(valid.tolist(), [[True, True, False, True]])
        self.assertTrue(np.all(np.abs(dec[valid] - k[valid]) <= 0.0051))

    def test_hash_verification_detects_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            make_event(d)
            load_event_cache(d, "ev", verify_hash=True)
            p = os.path.join(d, "ev.u16.npy")
            with open(p, "r+b") as f:
                f.seek(-4, os.SEEK_END)
                f.write(b"\x01\x02\x03\x04")
            with self.assertRaises(ValueError):
                load_event_cache(d, "ev", verify_hash=True)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            make_event(d)
            with self.assertRaises(FileExistsError):
                make_event(d)


class TestStore(unittest.TestCase):
    def test_gap_validation(self):
        with tempfile.TemporaryDirectory() as d:
            make_event(d)
            for bad in (15, 5, 25, 7):
                with self.assertRaises(ValueError):
                    FastSequenceStore(d, ["ev"], gaps_min=(bad,))

    def test_offsets_for_each_gap(self):
        """Frame t has K = 250 + 0.5 t, so decoded inputs/targets reveal exact frame indices."""
        with tempfile.TemporaryDirectory() as d:
            make_event(d, T=40)
            store = FastSequenceStore(d, ["ev"], gaps_min=(10, 20, 30), use_patch_stats_cache=False)
            for g in (10, 20, 30):
                ids = store.select(gap_min=g)
                self.assertGreater(len(ids), 0)
                b = store.gather(ids[:5])
                s = g // 5
                for j, i in enumerate(ids[:5]):
                    t = int(store.it_t[i])
                    inp = [decode_kelvin(b["raw_in"][j, k])[0].mean() for k in range(4)]
                    tgt = decode_kelvin(b["raw_tgt"][j])[0].mean()
                    exp_in = [250 + 0.5 * (t + off) for off in (-2 * s, -s, 0, s)]
                    self.assertTrue(np.allclose(inp, exp_in, atol=0.01), (g, inp, exp_in))
                    self.assertAlmostEqual(tgt, 250 + 0.5 * (t + s // 2), places=1)
            # G=30 must be the original [-60,-30,0,+30] -> +15 design: anchors t in [12, T-7)
            ids30 = store.select(gap_min=30)
            ts = np.unique(store.it_t[ids30])
            self.assertEqual((ts.min(), ts.max()), (12, 33))

    def test_strict_validity_excludes_patches_touching_invalid_pixel(self):
        with tempfile.TemporaryDirectory() as d:
            # invalid pixel in frame 20, inside patch 0 only (rows/cols 5..6)
            make_event(d, T=40, invalid=(20, 5, 5))
            store = FastSequenceStore(d, ["ev"], gaps_min=(10,), use_patch_stats_cache=False)
            s = 2
            # any sample whose 5 frames include frame 20 must not use patch 0
            for i in range(len(store)):
                t, p = int(store.it_t[i]), int(store.it_p[i])
                frames = {t - 2 * s, t - s, t, t + s, t + s // 2}
                if 20 in frames:
                    self.assertNotEqual(p, 0)
            # and patch 0 is still used when frame 20 is not involved
            self.assertTrue(((store.it_p == 0)).any())

    def test_missing_timestamp_blocks_spanning_windows(self):
        with tempfile.TemporaryDirectory() as d:
            make_event(d, T=40, drop_index=20)   # no observation at 5-min step 20
            store = FastSequenceStore(d, ["ev"], gaps_min=(10,), use_patch_stats_cache=False)
            self.assertGreater(len(store), 0)
            mp = load_event_cache(d, "ev")[1]
            from src.data.fast_cache import parse_iso
            ts = [parse_iso(x) for x in mp["timestamps"]]
            for i in range(len(store)):
                t, s = int(store.it_t[i]), int(store.it_s[i])
                for off in (-2 * s, -s, s, s // 2):
                    gap_s = (ts[t + off] - ts[t]).total_seconds()
                    self.assertAlmostEqual(gap_s, off * 300.0, delta=90)

    def test_convective_flag_uses_target_frame_only(self):
        with tempfile.TemporaryDirectory() as d:
            make_event(d, T=40, cold_target=(25, 0))
            store = FastSequenceStore(d, ["ev"], gaps_min=(10,), use_patch_stats_cache=False)
            for i in range(len(store)):
                t, s, p = int(store.it_t[i]), int(store.it_s[i]), int(store.it_p[i])
                expect = (t + s // 2 == 25) and p == 0
                self.assertEqual(bool(store.it_conv[i]), expect)
            self.assertTrue(store.it_conv.any())
            self.assertLess(store.it_tmin_k[store.it_conv].max(), 221.0)

    def test_sampling_is_deterministic_and_respects_convective_fraction(self):
        with tempfile.TemporaryDirectory() as d:
            make_event(d, T=60, cold_target=(25, 0), base=250.0)
            store = FastSequenceStore(d, ["ev"], gaps_min=(10, 20, 30), use_patch_stats_cache=False)
            a = store.sample_train_ids(np.random.default_rng(7), 16, convective_frac=0.5)
            b = store.sample_train_ids(np.random.default_rng(7), 16, convective_frac=0.5)
            self.assertTrue(np.array_equal(a, b))
            self.assertEqual(int(store.it_conv[a].sum()), 8)
            gaps = store.it_gap_min[store.sample_train_ids(np.random.default_rng(1), 4000)]
            self.assertGreater((gaps == 30).mean(), 0.40)   # gap-30 oversampled (~0.5)

    def test_patch_stats_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            make_event(d, T=40, cold_target=(25, 0))
            s1 = FastSequenceStore(d, ["ev"], gaps_min=(10, 30))
            s2 = FastSequenceStore(d, ["ev"], gaps_min=(10, 30))   # loads .npz
            self.assertTrue(np.array_equal(s1.it_conv, s2.it_conv))
            self.assertTrue(np.array_equal(s1.it_t, s2.it_t))


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTorchGlue(unittest.TestCase):
    def test_to_device_matches_numpy_decode(self):
        import torch
        from src.data.fast_dataset import to_device
        with tempfile.TemporaryDirectory() as d:
            make_event(d, T=40, invalid=(3, 5, 5))
            store = FastSequenceStore(d, ["ev"], gaps_min=(30,), use_patch_stats_cache=False)
            b = store.gather(store.select(gap_min=30)[:3])
            out = to_device(b, torch.device("cpu"))
            ref, valid = decode_kelvin(b["raw_in"])
            self.assertTrue(np.allclose(out["input_kelvin"].numpy(), ref, equal_nan=True))
            self.assertEqual(tuple(out["input_mask"].shape), (3, 4, 128, 128))
            self.assertTrue(bool(out["combined_mask"].all()))   # strict validity => all valid


if __name__ == "__main__":
    unittest.main()
