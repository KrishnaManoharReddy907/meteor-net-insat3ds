"""
src/data/goes_s3.py
Pure helpers for finding and downloading GOES-19 ABI-L2-CMIPC files from the public NOAA
bucket (anonymous HTTPS, no credentials). Network functions take injectable callables so the
parsing / grouping logic can be unit-tested without internet.

File name example:
    OR_ABI-L2-CMIPC-M6C13_G19_s20260010201193_e20260010203579_c20260010204034.nc
        mode=6  band=13  start = 2026 day-001 02:01:19 (+0.3 s)
All bands of one scan share the same start time to the second.
"""
from __future__ import annotations

import datetime
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

BASE_URL = "https://noaa-goes19.s3.amazonaws.com/"
PRODUCT = "ABI-L2-CMIPC"
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
KEY_RE = re.compile(
    r"OR_ABI-L2-CMIPC-M(?P<mode>\d)C(?P<band>\d{2})_G19_s(?P<s>\d{13})(?P<tenth>\d)"
    r"_e\d{14}_c\d{14}\.nc$"
)
USER_AGENT = "meteor-net-insat3ds/1.0"


def parse_key(key: str) -> Optional[Dict]:
    m = KEY_RE.search(key)
    if not m:
        return None
    return {
        "key": key,
        "mode": int(m.group("mode")),
        "band": int(m.group("band")),
        "datetime": datetime.datetime.strptime(m.group("s"), "%Y%j%H%M%S"),
    }


def prefix_for(year: int, doy: int, hour: int) -> str:
    return f"{PRODUCT}/{year}/{doy:03d}/{hour:02d}/"


def hours_covering(start: datetime.datetime, end: datetime.datetime) -> List[Tuple[int, int, int]]:
    """(year, day-of-year, hour) for every UTC hour touched by [start, end]."""
    if end < start:
        raise ValueError("end before start")
    t = start.replace(minute=0, second=0, microsecond=0)
    out = []
    while t <= end:
        out.append((t.year, t.timetuple().tm_yday, t.hour))
        t += datetime.timedelta(hours=1)
    return out


def parse_listing_xml(data: bytes) -> Tuple[List[str], bool, Optional[str]]:
    """S3 ListObjectsV2 response -> (keys, is_truncated, next_continuation_token)."""
    root = ET.fromstring(data)
    keys = [c.find(_NS + "Key").text for c in root.findall(_NS + "Contents")]
    trunc = (root.findtext(_NS + "IsTruncated") or "false").strip().lower() == "true"
    token = root.findtext(_NS + "NextContinuationToken")
    return keys, trunc, token


def _http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_keys(prefix: str, timeout: float = 30.0, retries: int = 4,
              http_get: Callable[[str, float], bytes] = _http_get) -> List[str]:
    """All keys under `prefix` (handles pagination, retries with backoff)."""
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        url = BASE_URL + "?" + urllib.parse.urlencode(q)
        for attempt in range(retries):
            try:
                data = http_get(url, timeout)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        page, trunc, token = parse_listing_xml(data)
        keys.extend(page)
        if not trunc or not token:
            return keys


def list_window_keys(start: datetime.datetime, end: datetime.datetime,
                     list_fn: Callable[[str], List[str]] = list_keys) -> List[str]:
    keys: List[str] = []
    for y, d, h in hours_covering(start, end):
        keys.extend(list_fn(prefix_for(y, d, h)))
    return keys


def group_scans(keys: Iterable[str], channels: Sequence[int],
                start: Optional[datetime.datetime] = None,
                end: Optional[datetime.datetime] = None) -> List[Dict]:
    """
    Group files into complete scans (all `channels` present with the same start second).
    start <= scan start <= end. If a band has duplicates for a scan (re-processed files) the
    lexicographically last key is used (later creation time). Sorted by time.
    """
    want = set(int(c) for c in channels)
    by_dt: Dict[datetime.datetime, Dict[int, str]] = {}
    modes: Dict[datetime.datetime, set] = {}
    for k in keys:
        p = parse_key(k)
        if p is None or p["band"] not in want:
            continue
        dt = p["datetime"]
        if (start is not None and dt < start) or (end is not None and dt > end):
            continue
        slot = by_dt.setdefault(dt, {})
        if p["band"] not in slot or k > slot[p["band"]]:
            slot[p["band"]] = k
        modes.setdefault(dt, set()).add(p["mode"])
    out = []
    for dt in sorted(by_dt):
        if set(by_dt[dt]) == want:
            out.append({"datetime": dt, "channels": by_dt[dt], "modes": sorted(modes[dt])})
    return out


def download_file(key: str, dest: str, timeout: float = 60.0, retries: int = 4) -> int:
    """Download to dest (atomic via .part). Reuses an existing non-empty file. Returns size."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return os.path.getsize(dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    url = BASE_URL + urllib.parse.quote(key)
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
                expected = int(r.headers.get("Content-Length", 0))
                n = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    n += len(chunk)
            if expected and n != expected:
                raise IOError(f"short read {n}/{expected} for {key}")
            os.replace(part, dest)
            return n
        except BaseException as e:  # noqa: BLE001
            last = e
            if os.path.exists(part):
                os.remove(part)
            time.sleep(2 ** attempt)
    raise IOError(f"download failed after {retries} tries: {key}: {last}")
