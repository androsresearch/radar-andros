"""Analyze captured radar frames (data/raw/ in dBZ, data/visual/ for display).

Usage:
  python analyze.py map data/visual/2026/07/21/andros_20260721_1420Z_visual.png
      -> writes a georeferenced PNG (coastline, graticule, Andros settlements)

  python analyze.py series
      -> walks every data/raw/**.png, decodes dBZ, and writes:
         analysis/series_andros.csv  (one row per 10-minute frame)
         analysis/series_andros.png  (time-series plot)

Geometry (must match capture.py): 512x512 px images, Web Mercator zoom 7,
centered on (24.45 N, -78.0 W). Coverage: lon -79.406 to -76.594,
lat 23.163 to 25.724 (~0.55 km/px).

Units: dBZ (reflectivity), same as the official radar website.
  dBZ = R/2 - 32   (red channel of raw PNG, RainViewer Black-and-White scheme)
Rain-rate estimate (Marshall-Palmer): Z = 200 * R^1.6

Coastline: Natural Earth 10m, fetched once from the official GitHub mirror
and cached in analysis/coast_andros.json.
"""

import glob
import json
import math
import os
import sys
import urllib.request

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# --- geometry (must match capture.py) ---
LAT_C, LON_C, ZOOM, SIZE = 24.45, -78.0, 7, 512
WORLD = 2 ** ZOOM

# approximate Andros Island (land) bounding box for the statistics
ANDROS = {"lat": (23.65, 25.25), "lon": (-78.55, -77.40)}

COAST_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_coastline.geojson"
)
COAST_CACHE = "analysis/coast_andros.json"

PLACES = {
    "Nicholls Town": (25.14, -78.00),
    "Fresh Creek": (24.73, -77.79),
    "Mangrove Cay": (24.23, -77.68),
    "Congo Town": (24.16, -77.55),
    "NASSAU": (25.06, -77.34),
}


def merc_y(lat):
    return (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * WORLD


def to_px(lon, lat):
    xc = (LON_C + 180) / 360 * WORLD
    yc = merc_y(LAT_C)
    return ((lon + 180) / 360 * WORLD - xc + 0.5) * SIZE, (merc_y(lat) - yc + 0.5) * SIZE


def px_to_lonlat(px, py):
    xc = (LON_C + 180) / 360 * WORLD
    yc = merc_y(LAT_C)
    lon = (xc + px / SIZE - 0.5) / WORLD * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (yc + py / SIZE - 0.5) / WORLD))))
    return lon, lat


def coastline():
    """Download (once) and clip the Natural Earth coastline to the scene."""
    if os.path.exists(COAST_CACHE):
        return json.load(open(COAST_CACHE))
    os.makedirs("analysis", exist_ok=True)
    print("downloading coastline (Natural Earth, one time only)...")
    raw = json.loads(urllib.request.urlopen(COAST_URL, timeout=120).read())
    lon0, lat0 = px_to_lonlat(0, 0)
    lon1, lat1 = px_to_lonlat(SIZE, SIZE)
    segments = []
    for feature in raw["features"]:
        g = feature["geometry"]
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for line in lines:
            pts = [(x, y) for x, y in line
                   if lon0 - 0.2 < x < lon1 + 0.2 and lat1 - 0.2 < y < lat0 + 0.2]
            if len(pts) > 1:
                segments.append(pts)
    json.dump(segments, open(COAST_CACHE, "w"))
    return segments


def decode_dbz(path_raw):
    """Raw PNG -> dBZ matrix (NaN where there is no echo)."""
    a = np.array(Image.open(path_raw).convert("RGBA"))
    dbz = np.where(a[..., 3] > 0, a[..., 0].astype(float) / 2 - 32, np.nan)
    dbz = np.where(dbz <= -31, np.nan, dbz)  # drop "no echo" floor values
    return dbz


def dbz_to_mmh(dbz):
    """Marshall-Palmer Z-R relation: Z = 200 * R^1.6 -> mm/h."""
    return (10 ** (dbz / 10) / 200) ** (1 / 1.6)


def draw_base(ax):
    for seg in coastline():
        pts = [to_px(x, y) for x, y in seg]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#e8d8a0", lw=1.4, zorder=2)
    for lon in np.arange(-79.5, -76.4, 0.5):
        p1, p2 = to_px(lon, 23.0), to_px(lon, 25.9)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", alpha=0.3, lw=0.7, zorder=1)
        ax.text(to_px(lon, 0)[0], 505, f"{abs(lon):.1f}°W", color="gray", fontsize=8, ha="center")
    for lat in np.arange(23.0, 26.0, 0.5):
        p1, p2 = to_px(-79.6, lat), to_px(-76.4, lat)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", alpha=0.3, lw=0.7, zorder=1)
        ax.text(6, to_px(0, lat)[1], f"{lat:.1f}°N", color="gray", fontsize=8, va="center")
    for name, (la, lo) in PLACES.items():
        x, y = to_px(lo, la)
        ax.plot(x, y, "o", color="#cc3333", ms=4, zorder=4)
        ax.text(x + 6, y - 4, name, color="#992222", fontsize=8, zorder=4)
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def cmd_map(path):
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.imshow(np.array(Image.open(path).convert("RGBA")), extent=[0, SIZE, SIZE, 0], zorder=3)
    draw_base(ax)
    name = os.path.basename(path)
    ax.set_title(f"Bahamas radar, georeferenced: {name}", fontsize=11)
    os.makedirs("analysis", exist_ok=True)
    out = os.path.join("analysis", name.replace(".png", "_georef.png"))
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print("saved:", out)


def cmd_series():
    # pixel mask of the Andros box
    x0, _ = to_px(ANDROS["lon"][0], LAT_C)
    x1, _ = to_px(ANDROS["lon"][1], LAT_C)
    _, y0 = to_px(LON_C, ANDROS["lat"][1])
    _, y1 = to_px(LON_C, ANDROS["lat"][0])
    xs = slice(int(x0), int(x1))
    ys = slice(int(y0), int(y1))

    frames = sorted(glob.glob("data/raw/**/*.png", recursive=True))
    if not frames:
        print("no frames found under data/raw/")
        return
    rows = []
    for f in frames:
        parts = os.path.basename(f).split("_")
        ts = parts[1] + "_" + parts[2]
        d = decode_dbz(f)
        box = d[ys, xs]
        echo = np.isfinite(box)
        if echo.any():
            dbz_max = float(np.nanmax(box))
            dbz_mean = float(np.nanmean(box))
            coverage = 100.0 * echo.sum() / echo.size
            rain_max = float(dbz_to_mmh(dbz_max))
        else:
            dbz_max = dbz_mean = rain_max = 0.0
            coverage = 0.0
        rows.append((ts, dbz_max, dbz_mean, coverage, rain_max))

    os.makedirs("analysis", exist_ok=True)
    with open("analysis/series_andros.csv", "w") as f:
        f.write("timestamp_utc,dbz_max,dbz_mean,echo_coverage_pct,rain_max_mmh\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.1f},{r[2]:.1f},{r[3]:.2f},{r[4]:.2f}\n")
    print(f"analysis/series_andros.csv ({len(rows)} frames)")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=110)
    t = range(len(rows))
    a1.plot(t, [r[1] for r in rows], "o-", ms=3, label="dBZ max")
    a1.plot(t, [r[2] for r in rows], "s-", ms=3, alpha=0.6, label="dBZ mean")
    a1.set_ylabel("dBZ")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.bar(t, [r[3] for r in rows], color="steelblue")
    a2.set_ylabel("echo coverage (%)")
    step = max(1, len(rows) // 15)
    a2.set_xticks(list(t)[::step])
    a2.set_xticklabels([rows[i][0] for i in t][::step], rotation=45, fontsize=7)
    a1.set_title("Andros Island box: radar time series (10-minute frames)")
    plt.tight_layout()
    plt.savefig("analysis/series_andros.png", dpi=110)
    print("analysis/series_andros.png")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "map":
        cmd_map(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "series":
        cmd_series()
    else:
        print(__doc__)
