"""Capture Bahamas radar frames centered on Andros Island.

Source: official Bahamas Department of Meteorology composite,
served by the RainViewer API (personal/educational use, attribution
required). https://www.rainviewer.com/

Saves two versions per 10-minute frame:
  data/raw/    -> "Black and White" scheme: pixel encodes reflectivity.
                  Decode: dBZ = R/2 - 32  (red channel)
                  Same units (dBZ) as the official radar website legend.
  data/visual/ -> color scheme for quick visual inspection only.

Geometry: 512x512 px, Web Mercator zoom 7, centered on (24.45 N, -78.0 W).
Coverage: lon -79.406 to -76.594, lat 23.163 to 25.724 (~0.55 km/px).
Zoom 7 is the maximum the RainViewer API allows.

Output filenames are frozen (breaking them orphans the archive):
  data/{raw|visual}/YYYY/MM/DD/andros_YYYYMMDD_HHMMZ_{raw|visual}.png
"""

import os
import time

import requests
from datetime import datetime, timezone

API = "https://api.rainviewer.com/public/weather-maps.json"

LAT, LON = 24.45, -78.0
ZOOM = 7
SIZE = 512

# label -> (color scheme, smooth_snow options)
# raw: scheme 0 (dBZ grayscale), unsmoothed (0_0) so values are unaltered
# visual: scheme 4 (Universal Blue), smoothed (1_1), inspection only
SCHEMES = {"raw": ("0", "0_0"), "visual": ("4", "1_1")}


def fetch_meta(retries: int = 3, backoff_s: int = 10):
    """Fetch the frame index, retrying transient API/network errors."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(API, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"meta fetch attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(backoff_s)
    return None


def main() -> None:
    meta = fetch_meta()
    if meta is None:
        # transient outage: exit nonzero for visibility; a missed cycle is
        # recoverable on the next run while frames remain in the 2 h window
        print("could not reach RainViewer API; giving up this cycle")
        raise SystemExit(1)

    host = meta["host"]
    new_frames = 0
    failed = 0

    for frame in meta["radar"]["past"]:
        ts = frame["time"]
        dt = datetime.fromtimestamp(ts, timezone.utc)

        for kind, (color, opts) in SCHEMES.items():
            out_dir = os.path.join("data", kind, dt.strftime("%Y/%m/%d"))
            out = os.path.join(out_dir, dt.strftime(f"andros_%Y%m%d_%H%MZ_{kind}.png"))
            if os.path.exists(out):
                continue

            url = f"{host}{frame['path']}/{SIZE}/{ZOOM}/{LAT}/{LON}/{color}/{opts}.png"
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
            except Exception as e:
                # one bad frame must not kill the run: it stays absent on
                # disk and is retried next cycle while still in the window
                print(f"frame fetch failed ({out}): {e}")
                failed += 1
                continue

            os.makedirs(out_dir, exist_ok=True)
            with open(out, "wb") as f:
                f.write(r.content)
            new_frames += 1
            print("saved:", out)

    print(f"new frames: {new_frames} | failed fetches: {failed}")


if __name__ == "__main__":
    main()
