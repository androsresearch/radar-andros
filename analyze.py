"""Analyze captured radar frames (data/raw/ in dBZ, data/visual/ for display).

Usage:
  python analyze.py series
      -> decodes every raw frame over the Andros box and writes:
         analysis/series_andros.csv  one row per 10-minute slot between the
             first and last captured frame; slots with no archived frame are
             kept as explicit rows with status=gap and empty statistics, so
             data loss is never read as zero rain
         analysis/series_andros.png  static plot, gaps shaded gray

  python analyze.py interactive
      -> reads the CSV (run 'series' first) and writes
         analysis/series_andros.html: three stacked panels (dBZ, mm/h,
         echo coverage) sharing one time axis, range slider for multi-day
         navigation, gaps shaded gray. Derived view of the CSV, not a new
         source.

  python analyze.py heatmap
      -> accumulates the Marshall-Palmer rain-rate estimate per pixel over
         all archived frames (10 min each) and writes
         analysis/heatmap_rain_mm.png: georeferenced map of estimated
         accumulated rainfall (mm) with coastline and settlements. Gaps are
         reported in the title (accumulation over missing slots is unknown).

  python analyze.py gif [YYYY-MM-DD]
      -> composes the visual frames (optionally one day only) over the
         georeferenced base with a per-frame UTC timestamp and writes
         analysis/radar_<range>.gif. If more than MAX_GIF_FRAMES frames
         match, they are subsampled evenly.

  python analyze.py map data/visual/.../andros_..._visual.png
      -> writes a georeferenced PNG of one frame (coastline, graticule,
         settlements)

Geometry (must match capture.py): 512x512 px, Web Mercator zoom 7, centered
on (24.45 N, -78.0 W). Coverage: lon -79.406 to -76.594, lat 23.163 to
25.724 (~0.55 km/px). This is NOT the whole Bahamas: it includes Andros,
New Providence (Nassau), the Berry Islands, Bimini and the northern Exumas;
Grand Bahama, Abaco and the southern islands fall outside the crop.

Units: dBZ (reflectivity), same as the official radar website.
  dBZ = R/2 - 32   (red channel of raw PNG, RainViewer Black-and-White scheme)
Rain-rate estimate (Marshall-Palmer): Z = 200 * R^1.6. Per-frame statistics
report dbz_max (raw, single-pixel, outlier-sensitive; reference only),
dbz_p95 (robust; drives the rain-rate column) and dbz_mean.

Coastline: Natural Earth 10m, fetched once from the official GitHub mirror
and cached in analysis/coast_andros.json.
"""

import glob
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import Image, ImageDraw

# --- geometry (must match capture.py) ---
LAT_C, LON_C, ZOOM, SIZE = 24.45, -78.0, 7, 512
WORLD = 2 ** ZOOM

# approximate Andros Island (land) bounding box for the statistics
ANDROS = {"lat": (23.65, 25.25), "lon": (-78.55, -77.40)}

FRAME_MINUTES = 10
MAX_GIF_FRAMES = 150

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

CSV_PATH = "analysis/series_andros.csv"
CSV_HEADER = ("timestamp_utc,status,dbz_max,dbz_p95,dbz_mean,"
              "echo_coverage_pct,rain_p95_mmh\n")


# ---------------------------------------------------------------- geometry --
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


# ------------------------------------------------------------------- data ---
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


def _ub_lut():
    """Official Universal Blue palette (RainViewer color table): RGB -> dBZ.

    Empirical finding (validated 2026-07-23 on this composite): the
    Black-and-White tiles served for the Bahamas composite do NOT follow the
    published BW encoding (interior-pixel cross-matching against the visual
    tiles shows e.g. BW value 255 rendering as the official 39-42 dBZ orange).
    The Universal Blue visual tiles DO match the published table exactly
    (color distance 0 on interior pixels), so quantitative decoding uses the
    UB palette lookup below. Smoothed tiles blend colors at echo edges; blended
    pixels beyond the match threshold are treated as no-data (NaN).
    """
    rain = {10:'cec087',11:'d2c48b',12:'d6c88f',13:'dacc93',14:'ded097',
            15:'88ddee',16:'6cd1eb',17:'51c5e8',18:'36bae5',19:'1baee2',
            20:'00a3e0',21:'009ad5',22:'0091ca',23:'0088bf',24:'007fb4',
            25:'0077aa',26:'0070a3',27:'00699c',28:'006295',29:'005b8e',
            30:'005588',31:'005180',32:'004e78',33:'004a70',34:'004768',
            35:'ffee00',36:'ffe000',37:'ffd200',38:'ffc500',39:'ffb700',
            40:'ffaa00',41:'ff9f00',42:'ff9500',43:'ff8b00',44:'ff8100',
            45:'ff4400',46:'f23600',47:'e62800',48:'d91b00',49:'cd0d00',
            50:'c10000',51:'a80000',52:'8f0000',53:'760000',54:'5d0000',
            55:'ffaaff',56:'ff9fff',57:'ff95ff',58:'ff8bff'}
    snow = {11:'b8f8ff',12:'b2f2ff',13:'abebff',14:'a5e5ff',15:'9fdfff',
            16:'98d8ff',17:'92d2ff',18:'8bcbff',19:'85c5ff',20:'7fbfff',
            22:'72b2ff',25:'5f9fff',30:'4f8fff',35:'3f7fff',40:'2f6fff',
            45:'1f5fff',50:'0f4fff',55:'003fff'}
    rgbs, dbzs = [], []
    for table in (rain, snow):
        for d, h in table.items():
            rgbs.append([int(h[i:i + 2], 16) for i in (0, 2, 4)])
            dbzs.append(float(d))
    return np.array(rgbs, dtype=int), np.array(dbzs)


_UB_RGB, _UB_DBZ = _ub_lut()
MATCH_DIST2 = 300  # max squared RGB distance for a palette match (~10/channel)


def decode_dbz(path_tile):
    """Radar tile PNG -> dBZ matrix (NaN where there is no precipitation echo).

    Structure of the BDM composite as served by RainViewer (verified
    2026-07-23 against the official color table, in-archive and live):

      * The `color` parameter has NO effect on this composite: schemes 0, 2
        and 4 return byte-identical tiles. The upstream image is passed
        through unmapped, which is why the documented Black-and-White
        encoding (dBZ = R/2 - 32) never applied here.
      * Fully opaque pixels (alpha 255) are radar echo, and match the
        official Universal Blue palette exactly (100% / 98.9% exact match on
        two independent archived frames).
      * Partially transparent pixels (alpha 73-190) are the basemap, a khaki
        land/bathymetry ramp with R>=G>B. They are NOT weak echo and must be
        excluded from every quantitative product.

    Filtering is therefore on alpha, which is exact by construction, rather
    than on color distance. Smoothed tiles (1_1) additionally blend palette
    colors at echo edges; blended opaque pixels beyond MATCH_DIST2 become
    NaN. Unsmoothed tiles (0_0) have far fewer of these.
    """
    a = np.array(Image.open(path_tile).convert("RGBA"))
    out = np.full(a.shape[:2], np.nan)
    m = a[..., 3] == 255           # radar echo only; excludes the basemap
    if not m.any():
        return out
    px = a[m][:, :3].astype(int)
    colors, inv = np.unique(px, axis=0, return_inverse=True)
    d2 = ((colors[:, None, :] - _UB_RGB[None, :, :]) ** 2).sum(axis=2)
    best = d2.argmin(axis=1)
    val = np.where(d2[np.arange(len(colors)), best] <= MATCH_DIST2,
                   _UB_DBZ[best], np.nan)
    out[m] = val[inv]
    return out


# Pixels at or above this reflectivity count as precipitation (NOAA: light
# rain begins around 20 dBZ; 10 dBZ keeps drizzle). Below it: cloud/clear-air.
RAIN_DBZ_MIN = 10.0

# Z-R conversion cap, standard operational practice to avoid hail/artifact
# contamination blowing up the exponential Marshall-Palmer relation.
DBZ_CAP = 55.0


def dbz_to_mmh(dbz):
    """Marshall-Palmer Z-R relation: Z = 200 * R^1.6 -> mm/h (dBZ capped)."""
    capped = np.minimum(dbz, DBZ_CAP)
    return (10 ** (capped / 10) / 200) ** (1 / 1.6)


def frame_time(path):
    """'.../andros_YYYYMMDD_HHMMZ_raw.png' -> datetime (UTC, naive)."""
    parts = os.path.basename(path).split("_")
    return datetime.strptime(parts[1] + parts[2][:4], "%Y%m%d%H%M")


# Frames archived before this date have data/raw/ in the old, unreliable
# Black-and-White encoding; for those the smoothed visual tile is the only
# decodable source. From this date on, data/raw/ is Universal Blue unsmoothed
# (exact palette colors, no edge blending) and is preferred.
RAW_IS_UB_FROM = datetime(2026, 7, 23)


def quantitative_frames():
    """Frames used for decoding, preferring the unsmoothed layer when valid.

    Returns a list of paths: data/raw/ (Universal Blue unsmoothed) for frames
    captured from RAW_IS_UB_FROM onward, data/visual/ otherwise.
    """
    out = []
    for v in sorted(glob.glob("data/visual/**/*.png", recursive=True)):
        t = frame_time(v)
        r = v.replace("data/visual/", "data/raw/").replace("_visual.png", "_raw.png")
        out.append(r if t >= RAW_IS_UB_FROM and os.path.exists(r) else v)
    return out


def expected_slots(times):
    """Every 10-minute slot between the first and last captured frame."""
    t, out = min(times), []
    while t <= max(times):
        out.append(t)
        t += timedelta(minutes=FRAME_MINUTES)
    return out


# -------------------------------------------------------------- base map ----
def draw_base(ax):
    for seg in coastline():
        pts = [to_px(x, y) for x, y in seg]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#8a7a3a", lw=1.4, zorder=4)
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
        ax.plot(x, y, "o", color="#cc3333", ms=4, zorder=5)
        ax.text(x + 6, y - 4, name, color="#992222", fontsize=8, zorder=5)
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)
    ax.set_xticks([])
    ax.set_yticks([])


# ------------------------------------------------------------------ series --
def cmd_series():
    frames = quantitative_frames()
    if not frames:
        print("no frames found under data/visual/")
        return
    by_time = {frame_time(f): f for f in frames}
    slots = expected_slots(list(by_time))

    x0, _ = to_px(ANDROS["lon"][0], LAT_C)
    x1, _ = to_px(ANDROS["lon"][1], LAT_C)
    _, y0 = to_px(LON_C, ANDROS["lat"][1])
    _, y1 = to_px(LON_C, ANDROS["lat"][0])
    xs, ys = slice(int(x0), int(x1)), slice(int(y0), int(y1))

    rows = []  # (dt, status, max, p95, mean, cov, rain)
    for t in slots:
        f = by_time.get(t)
        if f is None:
            rows.append((t, "gap", None, None, None, None, None))
            continue
        box = decode_dbz(f)[ys, xs]
        rainpx = np.isfinite(box) & (box >= RAIN_DBZ_MIN)
        if rainpx.any():
            vals = box[rainpx]
            dmax = float(vals.max())
            dp95 = float(np.percentile(vals, 95))
            dmean = float(vals.mean())
            cov = 100.0 * rainpx.sum() / rainpx.size
            rain = float(dbz_to_mmh(dp95))
        else:
            dmax = dp95 = dmean = rain = 0.0
            cov = 0.0
        rows.append((t, "ok", dmax, dp95, dmean, cov, rain))

    n_gap = sum(1 for r in rows if r[1] == "gap")
    os.makedirs("analysis", exist_ok=True)
    with open(CSV_PATH, "w") as f:
        f.write(CSV_HEADER)
        for t, st, dmax, dp95, dmean, cov, rain in rows:
            ts = t.strftime("%Y%m%d_%H%MZ")
            if st == "gap":
                f.write(f"{ts},gap,,,,,\n")
            else:
                f.write(f"{ts},ok,{dmax:.1f},{dp95:.1f},{dmean:.1f},{cov:.2f},{rain:.2f}\n")
    print(f"{CSV_PATH} ({len(rows)} slots, {n_gap} gaps)")

    # static plot with real datetimes and gap shading
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=110)
    ok = [r for r in rows if r[1] == "ok"]
    tt = [r[0] for r in ok]
    a1.plot(tt, [r[2] for r in ok], "o-", ms=2, alpha=0.5, label="dBZ max")
    a1.plot(tt, [r[3] for r in ok], "^-", ms=2, label="dBZ p95")
    a1.plot(tt, [r[4] for r in ok], "s-", ms=2, alpha=0.6, label="dBZ mean")
    a1.set_ylabel("dBZ")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.bar(tt, [r[5] for r in ok], width=0.006, color="steelblue")
    a2.set_ylabel("echo coverage (%)")
    half = timedelta(minutes=FRAME_MINUTES / 2)
    for r in rows:
        if r[1] == "gap":
            for ax in (a1, a2):
                ax.axvspan(r[0] - half, r[0] + half, color="0.85", zorder=0)
    a1.set_title(f"Andros Island box: radar time series "
                 f"(10-minute frames; gray bands are data gaps, n={n_gap})")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig("analysis/series_andros.png", dpi=110)
    print("analysis/series_andros.png")


def read_series_csv(path=CSV_PATH):
    """Load the series CSV. Returns list of dict rows; gaps keep None stats."""
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split(",")
            if len(c) != 7:
                continue
            t = datetime.strptime(c[0].replace("Z", ""), "%Y%m%d_%H%M")
            if c[1] == "gap":
                rows.append({"t": t, "status": "gap"})
            else:
                rows.append({"t": t, "status": "ok",
                             "dbz_max": float(c[2]), "dbz_p95": float(c[3]),
                             "dbz_mean": float(c[4]), "coverage": float(c[5]),
                             "rain": float(c[6])})
    return rows


# ------------------------------------------------------------- interactive --
def cmd_interactive():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} not found; run 'series' first")
        return
    rows = read_series_csv()
    ok = [r for r in rows if r["status"] == "ok"]
    gaps = [r["t"] for r in rows if r["status"] == "gap"]
    if not ok:
        print("CSV has no ok rows")
        return
    x = [r["t"] for r in ok]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Reflectivity (dBZ)", "Estimated rain rate (mm/h)",
                        "Echo coverage over Andros (%)"),
    )
    fig.add_trace(go.Scatter(x=x, y=[r["dbz_max"] for r in ok], name="dBZ max",
                             line=dict(color="#B5862B", width=1), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["dbz_p95"] for r in ok], name="dBZ p95",
                             line=dict(color="#185FA5", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["dbz_mean"] for r in ok], name="dBZ mean",
                             line=dict(color="#3B6D11", width=1), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[r["rain"] for r in ok], name="rain p95 (mm/h)",
                             line=dict(color="#185FA5", width=2), showlegend=False), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=[r["coverage"] for r in ok], name="coverage (%)",
                         marker_color="#4682B4", showlegend=False), row=3, col=1)

    # explicit data gaps: gray bands across all panels
    half = timedelta(minutes=FRAME_MINUTES / 2)
    for g in gaps:
        fig.add_vrect(x0=g - half, x1=g + half, fillcolor="gray",
                      opacity=0.25, line_width=0)
    # legend proxy so gaps are named in the legend
    fig.add_trace(go.Bar(x=[x[0]], y=[0], name="data gap (no frame)",
                         marker_color="gray", opacity=0.4), row=1, col=1)

    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=3, col=1)
    fig.update_yaxes(title_text="dBZ", row=1, col=1)
    fig.update_yaxes(title_text="mm/h", row=2, col=1)
    fig.update_yaxes(title_text="%", range=[0, 100], row=3, col=1)
    fig.update_layout(
        title=(f"Andros Island box: radar time series "
               f"(10-minute frames; gray bands are data gaps, n={len(gaps)})"),
        height=800, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    os.makedirs("analysis", exist_ok=True)
    out = "analysis/series_andros.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"{out} ({len(ok)} frames, {len(gaps)} gaps)")


# ---------------------------------------------------------------- heatmap ---
def cmd_heatmap():
    frames = quantitative_frames()
    if not frames:
        print("no frames found under data/visual/")
        return
    acc = np.zeros((SIZE, SIZE))
    for f in frames:
        d = decode_dbz(f)
        rainpx = np.isfinite(d) & (d >= RAIN_DBZ_MIN)
        mmh = np.where(rainpx, dbz_to_mmh(np.where(rainpx, d, 0.0)), 0.0)
        acc += mmh * (FRAME_MINUTES / 60.0)  # mm in this 10-min slot
    times = [frame_time(f) for f in frames]
    n_gap = len(expected_slots(times)) - len(times)

    fig, ax = plt.subplots(figsize=(9.5, 9), dpi=110)
    ax.set_facecolor("white")
    shown = np.ma.masked_less_equal(acc, 0.1)  # hide <0.1 mm for readability
    vmax = float(acc.max()) if acc.max() > 1 else 1.0
    im = ax.imshow(shown, extent=[0, SIZE, SIZE, 0], zorder=3, cmap="turbo",
                   norm=mcolors.LogNorm(vmin=0.1, vmax=vmax), alpha=0.85)
    draw_base(ax)
    # Andros statistics box for reference
    bx0, by0 = to_px(ANDROS["lon"][0], ANDROS["lat"][1])
    bx1, by1 = to_px(ANDROS["lon"][1], ANDROS["lat"][0])
    ax.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0],
            color="#333333", lw=1.0, ls="--", zorder=5)
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(f"estimated accumulated rainfall (mm, Marshall-Palmer, "
                 f"dBZ>={RAIN_DBZ_MIN:.0f}, cap {DBZ_CAP:.0f} dBZ, log scale)")
    t0, t1 = min(times), max(times)
    ax.set_title(f"Estimated accumulated rainfall: {t0:%Y-%m-%d %H:%M}Z to "
                 f"{t1:%Y-%m-%d %H:%M}Z\n{len(frames)} frames; {n_gap} missing "
                 f"slots not counted (accumulation there is unknown)", fontsize=10)
    plt.tight_layout()
    os.makedirs("analysis", exist_ok=True)
    out = "analysis/heatmap_rain_mm.png"
    plt.savefig(out, dpi=110)
    print(f"saved: {out}")


# -------------------------------------------------------------------- gif ---
def cmd_gif(day=None):
    vis = sorted(glob.glob("data/visual/**/*.png", recursive=True))
    if day:
        key = day.replace("-", "")
        vis = [f for f in vis if os.path.basename(f).split("_")[1] == key]
    if not vis:
        print("no visual frames match")
        return
    if len(vis) > MAX_GIF_FRAMES:
        idx = np.linspace(0, len(vis) - 1, MAX_GIF_FRAMES).astype(int)
        vis = [vis[i] for i in sorted(set(idx))]
        print(f"subsampled to {len(vis)} frames (cap {MAX_GIF_FRAMES})")

    # render the georeferenced base once
    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    draw_base(ax)
    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    base = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    base_img = Image.fromarray(base).convert("RGBA")
    W, H = base_img.size
    # geometry of the axes area inside the rendered figure
    ax_bbox = ax.get_position()
    px0, px1 = int(ax_bbox.x0 * W), int(ax_bbox.x1 * W)
    py0, py1 = int((1 - ax_bbox.y1) * H), int((1 - ax_bbox.y0) * H)

    out_frames = []
    for f in vis:
        radar = Image.open(f).convert("RGBA").resize((px1 - px0, py1 - py0))
        frame = Image.new("RGBA", (W, H), "white")
        frame.paste(radar, (px0, py0), radar)          # radar under the lines
        frame.alpha_composite(base_img)                # coastline+labels on top
        d = ImageDraw.Draw(frame)
        ts = frame_time(f).strftime("%Y-%m-%d %H:%M UTC")
        d.rectangle([px0, py0, px0 + 190, py0 + 22], fill=(255, 255, 255, 220))
        d.text((px0 + 6, py0 + 4), ts, fill=(20, 20, 20))
        out_frames.append(frame.convert("P", palette=Image.ADAPTIVE))

    os.makedirs("analysis", exist_ok=True)
    tag = day if day else f"{frame_time(vis[0]):%Y%m%d}-{frame_time(vis[-1]):%Y%m%d}"
    out = f"analysis/radar_{tag}.gif"
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:],
                       duration=150, loop=0, optimize=True)
    print(f"saved: {out} ({len(out_frames)} frames)")


# -------------------------------------------------------------------- map ---
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


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "map":
        cmd_map(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "series":
        cmd_series()
    elif len(sys.argv) >= 2 and sys.argv[1] == "interactive":
        cmd_interactive()
    elif len(sys.argv) >= 2 and sys.argv[1] == "heatmap":
        cmd_heatmap()
    elif len(sys.argv) >= 2 and sys.argv[1] == "gif":
        cmd_gif(sys.argv[2] if len(sys.argv) >= 3 else None)
    else:
        print(__doc__)
