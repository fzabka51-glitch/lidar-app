# =============================================================================
# app.py – LiDAR Archäologie Pro 2.5 (eine Datei, ersetzt importers/archaeology/exporters)
# Konvention: Raster intern "Nord oben" (Zeile 0 = Norden), alle Parameter in Metern.
# Neu in 2.1: größere Galerie, Einzelbild-Ansicht, Mausrad-Zoom, Bildhöhe/-auflösung.
# Neu in 2.2: Kontrast-Streckung für alle Modelle, kräftigere 3D-Schattierung.
# Neu in 2.3: Tab "Flutung" (Wasserstand, Uferlinie, Flutkurve, Strandlinien, 3D, Export).
# Neu in 2.4: Regionale Flutung auf der Karte (AWS Terrain Tiles), neues Design.
# Neu in 2.5: Wasser wird mit dem Relief zu einem Bild verschmolzen (sicher sichtbar),
#             kräftigere Wasserfarbe, Animation wieder mit Play/Pause/Schieberegler
#             (Bedienung unter dem Bild), GIF als optionaler Download.
# =============================================================================
from __future__ import annotations
import base64, html, io, json, math, os, re, warnings
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import ndimage

st.set_page_config(page_title="LiDAR Archäologie Pro", page_icon="🏛️", layout="wide")

try:
    import laspy
    LASPY_OK = True
except ImportError:
    LASPY_OK = False
try:
    from rasterio.io import MemoryFile
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    RASTERIO_OK = True
except ImportError:
    RASTERIO_OK = False
try:
    import pyproj
    PYPROJ_OK = True
except ImportError:
    PYPROJ_OK = False
try:
    from skimage.feature import blob_log, canny
    from skimage.transform import hough_circle, hough_circle_peaks, probabilistic_hough_line
    from skimage.measure import find_contours
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False
try:
    from plyfile import PlyData
    PLY_OK = True
except ImportError:
    PLY_OK = False
try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except ImportError:
    PIL_OK = False

APP_VERSION = "2.5"
MAX_PIXELS = 6_000_000  # Speicherschutz (~2450 x 2450 px) – bei viel RAM erhöhen
POINT_EXT = (".xyz", ".txt", ".csv", ".las", ".laz", ".ply")
RASTER_EXT = (".tif", ".tiff", ".geotiff", ".asc")
ESRI_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
LABEL_COLORS = {"auffällig": "#e41a1c", "möglich": "#ff7f00", "unklar": "#377eb8"}
CIRCLE_T = np.linspace(0, 2 * np.pi, 37)
GEOM_KEYS = ["r0", "c0", "r1", "c1", "rad_px"]
WATER_CMAP = LinearSegmentedColormap.from_list("wasser", ["#6ec1f5", "#1f78d1", "#08306b"])
FLOOD_MODES = ["Badewanne (alles unter dem Wasserstand)",
               "Zusammenhängend vom Kachelrand",
               "Zusammenhängend ab Startpunkt"]
REGION_MODES = ["Badewanne (alles unter dem Wasserstand)",
                "Zusammenhängend vom Regionsrand (z. B. Meer)",
                "Zusammenhängend ab Startpunkt (aus Tab Flutung)"]
DEFAULTS = dict(
    use_file_crs=True, epsg=25832, grid_res=1.0, agg_mode="Mittelwert", ground_only=True,
    morph_filter=False, morph_window_m=15.0, morph_threshold_m=1.5,
    hs_azimuth=315, hs_altitude=35, lrm_sigma_m=15.0,
    enable_svf=False, svf_dirs=8, svf_radius_m=15,
    stretch=(2.0, 98.0), hide_voids=True, z_exag=2.0, view_height=850, view_px=2000,
    shade_3d=0.7,
    det_lines=True, line_min_m=20.0, det_blobs=True, blob_d=(4.0, 30.0), blob_min_h=0.2,
    det_rings=False, ring_d=(10.0, 60.0),
    flood_mode=FLOOD_MODES[0], water_opacity=0.7, show_shore=True, flood_base="Hillshade",
    map_water_tile=False, map_water_region=False, region_km=10, region_mode=REGION_MODES[0],
    region_offset_on=True,
)
S = st.session_state
for _k, _v in DEFAULTS.items():
    S.setdefault(_k, _v)

# --- Design (ergänzt das Farbschema aus .streamlit/config.toml) ---------------
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1700px;}
.app-header {background: linear-gradient(135deg, #4a3216 0%, #8b5a2b 55%, #c89b5a 100%);
  color: #fff; padding: 18px 26px; border-radius: 14px; margin-bottom: 16px;
  box-shadow: 0 4px 14px rgba(74, 50, 22, .25);}
.app-header h1 {color: #fff !important; margin: 0; font-size: 1.9rem; padding: 0;}
.app-header p {margin: 4px 0 0; opacity: .92; font-size: 1rem;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e6dccb; border-radius: 12px;
  padding: 10px 14px; box-shadow: 0 1px 4px rgba(0, 0, 0, .05);}
div[data-testid="stMetricValue"] {font-size: 1.3rem;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background: #f1e8da; border-radius: 10px 10px 0 0; padding: 8px 18px;}
.stTabs [aria-selected="true"] {background: #8b5a2b !important;}
.stTabs [aria-selected="true"] p {color: #fff !important; font-weight: 600;}
section[data-testid="stSidebar"] {background: #f4ede2;}
div[data-testid="stExpander"] {border-radius: 10px; background: #fff;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Datenstrukturen & Hilfsfunktionen
# =============================================================================
@dataclass
class GeoGrid:
    z: np.ndarray
    void: np.ndarray
    x_min: float
    y_max: float
    res: float

    @property
    def h(self): return self.z.shape[0]
    @property
    def w(self): return self.z.shape[1]
    @property
    def x_max(self): return self.x_min + self.w * self.res
    @property
    def y_min(self): return self.y_max - self.h * self.res

    def pixel_to_world(self, row, col):
        row, col = np.asarray(row, float), np.asarray(col, float)
        return self.x_min + (col + 0.5) * self.res, self.y_max - (row + 0.5) * self.res

    def world_to_pixel(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (self.y_max - y) / self.res - 0.5, (x - self.x_min) / self.res - 0.5


@dataclass
class Layer:
    raw: np.ndarray
    cmap: str
    unit: str = ""
    diverging: bool = False
    fixed: tuple | None = None
    desc: str = ""


def file_ext(name):
    n = name.lower()
    return ".laz" if n.endswith(".copc.laz") else os.path.splitext(n)[1]


def _is_number(s):
    try:
        float(str(s).replace(",", "."))
        return True
    except ValueError:
        return False


def _num(v, nd=2):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), nd)


def _crs_str(crs):
    if crs is None:
        return None
    try:
        e = crs.to_epsg()
        if e:
            return f"EPSG:{e}"
    except Exception:
        pass
    try:
        return crs.to_wkt()
    except Exception:
        return None


def is_geographic(crs):
    try:
        return PYPROJ_OK and bool(crs) and pyproj.CRS.from_user_input(crs).is_geographic
    except Exception:
        return False


@st.cache_resource
def get_transformer(crs):
    return pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def to_latlon(x, y, crs):
    if not PYPROJ_OK or not crs:
        return None, None
    try:
        lon, lat = get_transformer(crs).transform(x, y)
        if not (np.all(np.isfinite(lat)) and np.all(np.abs(lat) <= 90)):
            return None, None
        return lat, lon
    except Exception:
        return None, None


def maps_link(lat, lon):
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"


def stride_ds(a, max_dim):
    s = max(1, math.ceil(max(a.shape) / max_dim))
    return a[::s, ::s], s


def block_nanmean(z, f):
    h, w = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(z[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3)).astype(np.float32)


def fmt_vol(v):
    if v >= 1e9:
        return f"{v / 1e9:.2f} km³"
    if v >= 1e6:
        return f"{v / 1e6:.2f} Mio. m³"
    return f"{v:,.0f} m³".replace(",", ".")


# =============================================================================
# Import
# =============================================================================
ALIASES = {"x": {"x", "east", "easting", "rechtswert", "e"},
           "y": {"y", "north", "northing", "hochwert", "n"},
           "z": {"z", "h", "height", "elevation", "hoehe", "höhe", "alt"},
           "classification": {"classification", "class", "klasse", "cls"}}


def load_xyz_text(b):
    """Robuster XYZ/TXT/CSV-Import: Trennzeichen, Kopfzeile, Dezimalkomma – float64!"""
    lines = [l for l in b[:65536].decode("utf-8", "ignore").splitlines()
             if l.strip() and not l.lstrip().startswith(("#", "//"))]
    if not lines:
        raise ValueError("Die Textdatei enthält keine lesbaren Daten.")
    first = lines[0].strip()
    sep = ";" if ";" in first else "," if "," in first else "\t" if "\t" in first else r"\s+"
    decimal = "," if sep == ";" and re.search(r"\d,\d", first) else "."
    tokens = [t for t in re.split(sep, first) if t.strip()]
    header = not all(_is_number(t) for t in tokens[:3])
    df = pd.read_csv(io.BytesIO(b), sep=sep, decimal=decimal, header=0 if header else None, comment="#")
    if not header:
        df.columns = [f"c{i}" for i in range(df.shape[1])]
    mapping, low = {}, {c: str(c).strip().lower().lstrip("/") for c in df.columns}
    for target, al in ALIASES.items():
        for c, lc in low.items():
            if lc in al and c not in mapping.values():
                mapping[target] = c
                break
    if not all(k in mapping for k in "xyz"):
        num = [c for c in df.columns if pd.to_numeric(df[c].head(200), errors="coerce").notna().mean() > 0.9]
        if len(num) < 3:
            raise ValueError("Keine drei numerischen Spalten (X, Y, Z) gefunden.")
        cls = mapping.get("classification")
        mapping = dict(zip("xyz", num[:3]))
        if cls is not None and cls not in num[:3]:
            mapping["classification"] = cls
    out = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in mapping.items()})
    return out.dropna(subset=["x", "y", "z"])


def load_ply(b):
    if not PLY_OK:
        raise ImportError("Für PLY wird 'plyfile' benötigt: pip install plyfile")
    v = PlyData.read(io.BytesIO(b))["vertex"]
    return pd.DataFrame({k: np.asarray(v[k], np.float64) for k in "xyz"})


def las_crs(header):
    try:
        crs = header.parse_crs()
        if crs is None:
            return None
        if crs.is_compound and crs.sub_crs_list:   # IGN: Lambert-93 + IGN69
            crs = crs.sub_crs_list[0]
        return _crs_str(crs)
    except Exception:
        return None


def las_bounds(b):
    with laspy.open(io.BytesIO(b)) as r:
        h = r.header
        return float(h.mins[0]), float(h.mins[1]), float(h.maxs[0]), float(h.maxs[1])


class GridAcc:
    """Aggregiert Punkte blockweise in ein Raster (Mittelwert oder Minimum), Nord oben."""

    def __init__(self, x0, y0, x1, y1, res, use_min):
        dx, dy = max(x1 - x0, res), max(y1 - y0, res)
        self.note = None
        if (dx // res + 1) * (dy // res + 1) > MAX_PIXELS:
            new = math.ceil(math.sqrt(dx * dy / MAX_PIXELS) * 105) / 100
            self.note = f"Auflösung automatisch von {res:.2f} m auf {new:.2f} m erhöht (Speicherschutz)."
            res = new
        self.res, self.x0, self.y1, self.use_min = res, x0, y1, use_min
        self.w, self.h = int(dx // res) + 1, int(dy // res) + 1
        self.acc = np.full(self.h * self.w, np.inf) if use_min else np.zeros(self.h * self.w)
        self.cnt = np.zeros(self.h * self.w, np.int32)
        self.n = 0

    def add(self, x, y, z):
        if len(x) == 0:
            return
        col = np.clip(((x - self.x0) / self.res).astype(np.int64), 0, self.w - 1)
        row = np.clip(((self.y1 - y) / self.res).astype(np.int64), 0, self.h - 1)
        idx = row * self.w + col
        order = np.argsort(idx, kind="stable")
        idx, z = idx[order], np.asarray(z, np.float64)[order]
        uniq, start, counts = np.unique(idx, return_index=True, return_counts=True)
        self.cnt[uniq] += counts.astype(np.int32)
        if self.use_min:
            self.acc[uniq] = np.minimum(self.acc[uniq], np.minimum.reduceat(z, start))
        else:
            self.acc[uniq] += np.add.reduceat(z, start)
        self.n += len(z)

    def result(self):
        ok = self.cnt > 0
        out = np.full(self.h * self.w, np.nan, np.float32)
        out[ok] = self.acc[ok] if self.use_min else self.acc[ok] / self.cnt[ok]
        return out.reshape(self.h, self.w)


def las_add(b, acc, ground_only):
    """Liest LAS/LAZ/COPC blockweise. Fällt bei fehlender Klassifikation auf alle Punkte zurück."""
    for use_filter in ([True, False] if ground_only else [False]):
        used = 0
        with laspy.open(io.BytesIO(b)) as r:
            crs = las_crs(r.header)
            for pts in r.chunk_iterator(2_000_000):
                x, y, z = (np.asarray(v, np.float64) for v in (pts.x, pts.y, pts.z))
                if use_filter:
                    m = np.asarray(pts.classification) == 2
                    x, y, z = x[m], y[m], z[m]
                acc.add(x, y, z)
                used += len(x)
        if used:
            return crs, use_filter != ground_only
    raise ValueError("Die LAS/LAZ-Datei enthält keine Punkte.")


def load_geotiff(b):
    if not RASTERIO_OK:
        raise ImportError("Für GeoTIFF wird 'rasterio' benötigt: pip install rasterio")
    with MemoryFile(b) as mem, mem.open() as src:
        t = src.transform
        if abs(t.b) > 1e-9 or abs(t.d) > 1e-9:
            raise ValueError("Rotierte Raster werden nicht unterstützt.")
        h, w = src.height, src.width
        f = max(1, math.ceil(math.sqrt(h * w / MAX_PIXELS)))
        oh, ow = max(1, h // f), max(1, w // f)
        arr = src.read(1, out_shape=(oh, ow), masked=True,
                       resampling=Resampling.average if f > 1 else Resampling.nearest)
        z = np.ma.filled(arr.astype(np.float32), np.nan)
        rx, ry = abs(t.a) * w / ow, abs(t.e) * h / oh
        if not math.isclose(rx, ry, rel_tol=0.02):
            raise ValueError(f"Nicht-quadratische Pixel ({rx:.3f} × {ry:.3f}) werden nicht unterstützt.")
        x_min = t.c if t.a > 0 else t.c + t.a * w
        if t.e < 0:
            y_max = t.f
        else:
            y_max, z = t.f + t.e * h, z[::-1]
        if t.a < 0:
            z = z[:, ::-1]
        crs = _crs_str(src.crs)
    z[z <= -9999] = np.nan
    notes = [f"Großes Raster ({h}×{w} px) auf {oh}×{ow} px reduziert (Faktor {f}). "
             "Für volle Auflösung einen kleineren Ausschnitt wählen."] if f > 1 else []
    return np.ascontiguousarray(z), float(x_min), float(y_max), float((rx + ry) / 2), crs, notes


def load_asc(b):
    lines = b.decode("utf-8", "ignore").splitlines()
    hdr, i = {}, 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 2 and not _is_number(parts[0]):
            hdr[parts[0].lower()] = float(parts[1].replace(",", "."))
            i += 1
        else:
            break
    nc, nr, cs = int(hdr["ncols"]), int(hdr["nrows"]), float(hdr["cellsize"])
    vals = np.array(" ".join(lines[i:]).split(), dtype=np.float32)
    if vals.size < nr * nc:
        raise ValueError("ASC-Datei unvollständig.")
    z = vals[: nr * nc].reshape(nr, nc)
    if "nodata_value" in hdr:
        z[z == hdr["nodata_value"]] = np.nan
    x_min = hdr["xllcenter"] - cs / 2 if "xllcenter" in hdr else hdr["xllcorner"]
    y_min = hdr["yllcenter"] - cs / 2 if "yllcenter" in hdr else hdr["yllcorner"]
    y_max, notes = y_min + nr * cs, []
    f = max(1, math.ceil(math.sqrt(nr * nc / MAX_PIXELS)))
    if f > 1:
        z, cs = block_nanmean(z, f), cs * f
        notes.append(f"Großes ASC-Raster um Faktor {f} reduziert.")
    return z, float(x_min), float(y_max), cs, None, notes


def fill_gaps(z):
    void = ~np.isfinite(z)
    if not void.any():
        return z.astype(np.float32)
    idx = ndimage.distance_transform_edt(void, return_distances=False, return_indices=True)
    filled = z[tuple(idx)].astype(np.float32)
    filled[void] = ndimage.uniform_filter(filled, size=5)[void]
    return filled


@st.cache_resource(max_entries=2, show_spinner="Lade und rasterisiere Daten …")
def load_grid(grid_key, _files):
    _, pc, morph = grid_key
    notes, n_points = [], None
    named = [(f.name, f.getvalue()) for f in _files]
    rasters = [(n, b) for n, b in named if file_ext(n) in RASTER_EXT]
    points = [(n, b) for n, b in named if file_ext(n) in POINT_EXT]
    if rasters:
        name, b = rasters[0]
        if len(named) > 1:
            notes.append(f"Nur '{name}' wird verwendet – Mosaike sind nur mit Punktwolken möglich.")
        z, x_min, y_max, res, crs, n = load_asc(b) if file_ext(name) == ".asc" else load_geotiff(b)
        notes += n + [f"Auflösung aus Datei: {res:.2f} m (Sidebar-Auflösung gilt nur für Punktwolken)."]
        if is_geographic(crs):
            raise ValueError("Raster liegt in geografischen Koordinaten (Grad). Bitte in ein "
                             "metrisches CRS (z. B. UTM) umprojizieren.")
        source = "Raster"
    else:
        res, agg, ground_only = pc
        parsed, crs = [], None
        for name, b in points:
            ext = file_ext(name)
            if ext in (".las", ".laz"):
                if not LASPY_OK:
                    raise ImportError("Für LAS/LAZ wird 'laspy[lazrs]' benötigt.")
                parsed.append(("las", name, b, las_bounds(b)))
            else:
                df = load_ply(b) if ext == ".ply" else load_xyz_text(b)
                if ground_only and "classification" in df and (df["classification"] == 2).any():
                    df = df[df["classification"] == 2]
                    notes.append(f"'{name}': nur Bodenpunkte (Klasse 2) verwendet.")
                parsed.append(("df", name, df, (df.x.min(), df.y.min(), df.x.max(), df.y.max())))
        bx = np.array([p[3] for p in parsed], float)
        acc = GridAcc(bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max(),
                      res, agg.startswith("Minimum"))
        if acc.note:
            notes.append(acc.note)
        for kind, name, payload, _ in parsed:
            if kind == "las":
                try:
                    c, fallback = las_add(payload, acc, ground_only)
                except Exception as e:
                    if "backend" in str(e).lower() or "lazrs" in str(e).lower():
                        raise RuntimeError("Kein LAZ-Backend gefunden: 'laspy[lazrs]' installieren.") from e
                    raise
                crs = crs or c
                if fallback:
                    notes.append(f"'{name}': keine Bodenklassifikation – alle Punkte verwendet "
                                 "(Tipp: Aggregation 'Minimum' oder Vegetationsfilter).")
            else:
                acc.add(payload.x.to_numpy(np.float64), payload.y.to_numpy(np.float64),
                        payload.z.to_numpy(np.float64))
        z, x_min, y_max, res, n_points = acc.result(), acc.x0, acc.y1, acc.res, acc.n
        if len(points) > 1:
            notes.append(f"{len(points)} Punktwolken zu einem Mosaik zusammengesetzt.")
        source = "Punktwolke"
    if min(z.shape) < 3:
        raise ValueError("Raster zu klein (mind. 3×3 Pixel).")
    void = ~np.isfinite(z)
    if void.all():
        raise ValueError("Keine gültigen Höhenwerte gefunden.")
    zf = fill_gaps(z)
    if void.any():
        notes.append(f"{void.mean() * 100:.1f} % der Zellen ohne Daten wurden interpoliert.")
    if morph:
        win = max(3, int(round(morph[0] / res)))
        nonground = (zf - ndimage.grey_opening(zf, size=(win, win))) > morph[1]
        if nonground.any():
            z2 = zf.copy()
            z2[nonground] = np.nan
            zf = fill_gaps(z2)
            notes.append(f"Vegetationsfilter: {nonground.mean() * 100:.1f} % der Zellen ersetzt.")
    big_void = ndimage.distance_transform_edt(void) > 2 if void.any() else void
    return dict(z=zf, void=big_void, x_min=float(x_min), y_max=float(y_max), res=float(res),
                crs=crs, notes=notes, source=source, n_points=n_points)


# =============================================================================
# Reliefmodelle (Rohwerte in physikalischen Einheiten)
# =============================================================================
def gradients(z, res):
    d_row, d_col = np.gradient(z, res)
    return d_col, -d_row  # dz/dx (Ost), dz/dy (Nord)


def hillshade(dzdx, dzdy, az, alt):
    az, alt = np.deg2rad(az), np.deg2rad(alt)
    lx, ly, lz = np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)
    return np.clip((-dzdx * lx - dzdy * ly + lz) / np.sqrt(dzdx**2 + dzdy**2 + 1), 0, 1).astype(np.float32)


def pct_range(a, lo, hi):
    v = stride_ds(a, 1000)[0]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    p = np.percentile(v, (lo, hi))
    return float(p[0]), float(p[1])


def norm01(a, lo=5, hi=95):
    l, h = pct_range(a, lo, hi)
    return np.full_like(a, 0.5) if h <= l else np.clip((a - l) / (h - l), 0, 1).astype(np.float32)


def horizon_models(z, res, n_dir, radius_m):
    """Sky-View-Factor, positive & negative Openness per Horizontscan."""
    r_px = max(2, int(round(radius_m / res)))
    steps = np.unique(np.round(np.geomspace(1, r_px, num=min(r_px, 20))).astype(int))
    pad, (h, w) = r_px + 1, z.shape
    zp = np.pad(z, pad, mode="edge")
    s_sin, s_pos, s_neg = (np.zeros(z.shape, np.float32) for _ in range(3))
    for k in range(n_dir):
        a = 2 * math.pi * k / n_dir
        mx, mn = np.full(z.shape, -np.inf, np.float32), np.full(z.shape, np.inf, np.float32)
        seen = set()
        for s in steps:
            dr, dc = int(round(-math.cos(a) * s)), int(round(math.sin(a) * s))
            if (dr, dc) in seen or (dr, dc) == (0, 0):
                continue
            seen.add((dr, dc))
            t = (zp[pad + dr:pad + dr + h, pad + dc:pad + dc + w] - z) / (math.hypot(dr, dc) * res)
            np.maximum(mx, t, out=mx)
            np.minimum(mn, t, out=mn)
        ma, mi = np.arctan(mx), np.arctan(mn)
        s_sin += np.sin(np.clip(ma, 0, None))
        s_pos += np.pi / 2 - ma
        s_neg += np.pi / 2 + mi
    return ((1 - s_sin / n_dir).astype(np.float32), np.degrees(s_pos / n_dir).astype(np.float32),
            np.degrees(s_neg / n_dir).astype(np.float32))


@st.cache_resource(max_entries=1, show_spinner="Berechne Reliefmodelle …")
def compute_models(models_key, _z, res, az, alt, sigma_m, svf_on, svf_dirs, svf_radius):
    z = _z
    dzdx, dzdy = gradients(z, res)
    hs = hillshade(dzdx, dzdy, az, alt)
    mds = np.mean([hillshade(dzdx, dzdy, a, alt) for a in (315, 45, 135, 225)], axis=0).astype(np.float32)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)
    aspect = ((np.degrees(np.arctan2(-dzdx, -dzdy)) + 360) % 360).astype(np.float32)
    curv = (-ndimage.laplace(z) / res**2).astype(np.float32)
    smooth = ndimage.gaussian_filter(z, max(1.0, sigma_m / res))
    lrm = (z - smooth).astype(np.float32)
    bg_slope = np.degrees(np.arctan(np.hypot(*gradients(smooth, res)))).astype(np.float32)
    radii = [r for r in (2, 5, 10, 20, 40) if r / res >= 1 and r <= min(z.shape) * res / 4] or [2 * res]
    msrm = np.mean([z - ndimage.gaussian_filter(z, r / res) for r in radii], axis=0).astype(np.float32)
    lrm_n = norm01(lrm)
    comp = np.clip(mds + (lrm_n - 0.5) * 0.3, 0, 1).astype(np.float32)
    if svf_on:
        svf, pos, neg = horizon_models(z, res, svf_dirs, svf_radius)
        parts = [hs, 1 - np.clip(slope / 50, 0, 1), np.clip((svf - 0.65) / 0.35, 0, 1),
                 np.clip((pos - 60) / 35, 0, 1)]
    else:
        parts = [hs, 1 - np.clip(slope / 50, 0, 1), lrm_n]
    L = {
        "Final Composite (Fusion)": Layer(comp, "gray", desc="MDS + Restrelief (Ursprungsversion)."),
        "VAT-ähnliche Fusion": Layer(np.mean(parts, axis=0).astype(np.float32), "gray",
                                     desc="Hillshade + Slope + " + ("SVF + Openness" if svf_on else "LRM")
                                     + " – angelehnt an VAT (Kokalj & Somrak)."),
        "Hillshade": Layer(hs, "gray", desc=f"Azimut {az}°, Sonnenhöhe {alt}°."),
        "MDS Composite": Layer(mds, "gray", desc="Mittel aus 4 Lichtrichtungen."),
        "Restrelief (LRM)": Layer(lrm, "RdBu_r", "m", True, desc=f"Höhe minus Glättung (σ={sigma_m:.0f} m). Rot = erhaben."),
        "Multi-Skalen-Relief": Layer(msrm, "RdBu_r", "m", True, desc="Mittleres Restrelief über 2–40 m."),
        "Hangneigung (Slope)": Layer(slope, "plasma", "°", desc="Neigung in Grad."),
        "Krümmung (Curvature)": Layer(curv, "RdYlGn", "1/m", True, desc="+ konvex (Kuppe), − konkav (Mulde)."),
        "Hangausrichtung (Aspect)": Layer(aspect, "twilight", "°", fixed=(0, 360), desc="Exposition ab Nord."),
    }
    if svf_on:
        L["Sky-View-Factor (SVF)"] = Layer(svf, "gray", "", desc="Anteil sichtbarer Himmel (0–1).")
        L["Positive Openness"] = Layer(pos, "gray", "°", desc="Betont Kuppen und Wälle.")
        L["Negative Openness"] = Layer(neg, "gray_r", "°", desc="Betont Gräben und Senken.")
    L["DTM (Höhe)"] = Layer(z, "terrain", "m", desc="Geländehöhe (gefüllt/gefiltert).")
    return L, {"lrm": lrm, "bg_slope": bg_slope}


# =============================================================================
# Darstellung
# =============================================================================
def plotly_cs(name):
    if name == "gray":
        return [[0, "black"], [1, "white"]]
    if name == "gray_r":
        return [[0, "white"], [1, "black"]]
    return {"terrain": "earth"}.get(name, name)


def vrange(L, stretch):
    if L.fixed:
        return L.fixed
    lo, hi = pct_range(L.raw, *stretch)
    if L.diverging:
        m = max(abs(lo), abs(hi), 1e-9)
        return -m, m
    return (lo, hi) if hi > lo else (lo, lo + 1e-6)


def display_array(L, grid, max_dim, hide):
    arr, s = stride_ds(L.raw, max_dim)
    arr = arr.astype(np.float32, copy=True)
    if hide and grid.void.any():
        arr[grid.void[::s, ::s]] = np.nan
    return arr, s


def ds_axes(grid, shape, s):
    return (grid.x_min + (np.arange(shape[1]) * s + 0.5) * grid.res,
            grid.y_max - (np.arange(shape[0]) * s + 0.5) * grid.res)


def to_rgba(arr, cmap, vmin, vmax):
    n = (arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(arr)
    return matplotlib.colormaps[cmap](np.ma.masked_invalid(np.clip(n, 0, 1)), bytes=True)


def png_bytes(rgba):
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return buf.getvalue()


def data_uri(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


def composite_rgba(base, top):
    """Legt 'top' (RGBA, mit Transparenz) über 'base' (RGBA) – Ergebnis ist ein deckendes Bild."""
    b = base.astype(np.float32) / 255.0
    t = top.astype(np.float32) / 255.0
    ba, ta = b[..., 3:4], t[..., 3:4]
    b_rgb = b[..., :3] * ba + (1.0 - ba)          # transparente Stellen -> weiß
    rgb = t[..., :3] * ta + b_rgb * (1.0 - ta)
    out = np.concatenate([rgb, np.ones_like(ta)], axis=-1)
    return (out * 255.0 + 0.5).astype(np.uint8)


PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False,
                 "toImageButtonOptions": {"format": "png", "scale": 2}}


def show_plot(fig, container=None):
    """Plotly-Diagramm in voller Breite, Mausrad-Zoom aktiv."""
    (container or st).plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def show_image(img, caption=None):
    """Bild in voller Spaltenbreite (kompatibel mit älteren und neueren Streamlit-Versionen)."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)


@st.cache_resource(max_entries=32)
def layer_png(key, _L, _grid, stretch, hide, max_dim):
    arr, _ = display_array(_L, _grid, max_dim, hide)
    return png_bytes(to_rgba(arr, _L.cmap, *vrange(_L, stretch)))


def img_extent(g):
    """Platziert ein Bild passgenau in Weltkoordinaten."""
    return dict(xref="x", yref="y", x=g.x_min, y=g.y_max, sizex=g.x_max - g.x_min,
                sizey=g.y_max - g.y_min, sizing="stretch", layer="below")


def apply_xy_layout(fig, title, height):
    fig.update_layout(
        height=int(height), title=title, margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan", uirevision=str(hash(grid_key)), plot_bgcolor="white",
        xaxis=dict(title="X (m)", range=[grid.x_min, grid.x_max], constrain="domain",
                   showgrid=False, zeroline=False),
        yaxis=dict(title="Y (m)", range=[grid.y_min, grid.y_max], scaleanchor="x", scaleratio=1,
                   showgrid=False, zeroline=False),
        legend=dict(orientation="h", y=-0.08))
    return fig


def feature_traces(feats, grid):
    traces = []
    for label, color in LABEL_COLORS.items():
        fl = [f for f in feats if f["Einstufung"] == label]
        if not fl:
            continue
        lx, ly, mx, my, txt = [], [], [], [], []
        for f in fl:
            if f["Typ"] == "Linie":
                x, y = grid.pixel_to_world([f["r0"], f["r1"]], [f["c0"], f["c1"]])
            else:
                x0, y0 = grid.pixel_to_world(f["r0"], f["c0"])
                rad = f["rad_px"] * grid.res
                x, y = x0 + rad * np.cos(CIRCLE_T), y0 + rad * np.sin(CIRCLE_T)
            lx += list(x) + [None]
            ly += list(y) + [None]
            mx.append(f["X"])
            my.append(f["Y"])
            h = f" · Höhe/Tiefe {f['Höhe_m']} m" if f["Höhe_m"] is not None else ""
            txt.append(f"#{f['ID']} {f['Typ']} · {f['Größe_m']} m{h}<br>{label}")
        traces.append(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=color, width=2),
                                 name=label, hoverinfo="skip"))
        traces.append(go.Scatter(x=mx, y=my, mode="markers", text=txt, showlegend=False,
                                 marker=dict(color=color, size=6, line=dict(color="white", width=1)),
                                 hovertemplate="%{text}<extra></extra>"))
    return traces


def layer_figure(name, L, grid, max_dim=None, feats=None, height=None, overlay=None, opacity=0.5):
    """Scharfes, zoombares Bild: PNG als Hintergrund + unsichtbares Raster für Messwerte."""
    max_dim = int(max_dim or S.view_px)
    height = int(height or S.view_height)
    fig = go.Figure()
    stack = [(L, 1.0)] + ([(overlay, opacity)] if overlay is not None else [])
    for layer, op in stack:
        png = layer_png((models_key, id(layer)), layer, grid, tuple(S.stretch), S.hide_voids, max_dim)
        fig.add_layout_image(source=data_uri(png), opacity=op, **img_extent(grid))
    hv, s = display_array(L, grid, 400, S.hide_voids)
    xs, ys = ds_axes(grid, hv.shape, s)
    u = f" {L.unit}" if L.unit else ""
    fig.add_trace(go.Heatmap(z=hv, x=xs, y=ys, opacity=0, showscale=False, showlegend=False,
                             hovertemplate=f"X %{{x:.1f}} · Y %{{y:.1f}}<br>≈ %{{z:.3f}}{u}<extra></extra>"))
    if L.unit:
        vmin, vmax = vrange(L, S.stretch)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(colorscale=plotly_cs(L.cmap), cmin=vmin, cmax=vmax,
                                             color=[vmin], showscale=True, colorbar=dict(title=L.unit))))
    for t in feature_traces(feats or [], grid):
        fig.add_trace(t)
    return apply_xy_layout(fig, name, height)


# =============================================================================
# Feature-Erkennung (heuristisch, keine archäologische Verifikation)
# =============================================================================
def classify(typ, bg, size, height=None):
    score = 2 if bg < 10 else (1 if bg < 20 else 0)
    if typ in ("Hügel", "Senke") and height is not None and 0.3 <= height <= 4.0:
        score += 1
    if typ == "Linie" and size and size >= 50:
        score += 1
    return "auffällig" if score >= 3 else "möglich" if score == 2 else "unklar"


def detect_features(lrm, bg_slope, grid, p):
    res, (H, W), feats = grid.res, lrm.shape, []
    lo, hi = pct_range(lrm, 1, 99)
    norm = np.clip(lrm / max(abs(lo), abs(hi), 1e-6), -1, 1).astype(np.float32)
    img01 = (norm + 1) / 2

    def add(typ, r0, c0, r1=None, c1=None, rad=None, size=None, height=None, vol=None):
        rm = r0 if r1 is None else (r0 + r1) / 2
        cm = c0 if c1 is None else (c0 + c1) / 2
        bg = float(bg_slope[int(np.clip(round(rm), 0, H - 1)), int(np.clip(round(cm), 0, W - 1))])
        x, y = grid.pixel_to_world(rm, cm)
        feats.append(dict(Typ=typ, r0=float(r0), c0=float(c0), r1=r1, c1=c1, rad_px=rad,
                          X=round(float(x), 2), Y=round(float(y), 2), Größe_m=_num(size, 1),
                          Höhe_m=_num(height), Volumen_m3=_num(vol, 1), Hang_bg_deg=_num(bg, 1),
                          Einstufung=classify(typ, bg, size, height)))

    if p["det_lines"]:
        f = max(1, math.ceil(max(H, W) / 2000))
        edges = canny(img01[::f, ::f], sigma=2.0, low_threshold=0.8, high_threshold=0.92, use_quantiles=True)
        ln = max(5, int(round(p["line_min_m"] / (res * f))))
        try:
            segs = probabilistic_hough_line(edges, threshold=10, line_length=ln, line_gap=3, rng=0)
        except TypeError:
            segs = probabilistic_hough_line(edges, threshold=10, line_length=ln, line_gap=3)
        for (xa, ya), (xb, yb) in sorted(segs, key=lambda s: -math.dist(*s))[:300]:
            add("Linie", ya * f, xa * f, float(yb * f), float(xb * f),
                size=math.dist((xa, ya), (xb, yb)) * f * res)

    if p["det_blobs"]:
        f = max(1, math.ceil(max(H, W) / 1500))
        dmin, dmax = p["blob_d"]
        smin = max(1.0, dmin / 2 / (res * f) / math.sqrt(2))
        smax = max(smin + 0.5, dmax / 2 / (res * f) / math.sqrt(2))
        for sign, typ in ((1, "Hügel"), (-1, "Senke")):
            cand = []
            for rb, cb, sb in blob_log(np.clip(sign * norm[::f, ::f], 0, 1), min_sigma=smin,
                                       max_sigma=smax, num_sigma=8, threshold=0.1, overlap=0.5):
                rad, r, c = sb * math.sqrt(2) * f, rb * f, cb * f
                rr = int(math.ceil(rad))
                r_lo, r_hi = max(0, int(r) - rr), min(H, int(r) + rr + 1)
                c_lo, c_hi = max(0, int(c) - rr), min(W, int(c) + rr + 1)
                yy, xx = np.ogrid[r_lo:r_hi, c_lo:c_hi]
                vals = sign * lrm[r_lo:r_hi, c_lo:c_hi][(yy - r) ** 2 + (xx - c) ** 2 <= rad**2]
                if vals.size and vals.max() >= p["blob_min_h"]:
                    cand.append((float(vals.max()), r, c, rad, float(np.clip(vals, 0, None).sum() * res**2)))
            for height, r, c, rad, vol in sorted(cand, reverse=True)[:400]:
                add(typ, r, c, rad=float(rad), size=2 * rad * res, height=height, vol=vol)

    if p["det_rings"]:
        f = max(1, math.ceil(max(H, W) / 1000))
        edges = canny(img01[::f, ::f], sigma=2.0, low_threshold=0.8, high_threshold=0.92, use_quantiles=True)
        rmin = max(3, int(round(p["ring_d"][0] / 2 / (res * f))))
        rmax = max(rmin + 1, int(round(p["ring_d"][1] / 2 / (res * f))))
        radii = np.unique(np.linspace(rmin, rmax, min(15, rmax - rmin + 1)).round().astype(int))
        acc, cxs, cys, rads = hough_circle_peaks(hough_circle(edges, radii), radii, total_num_peaks=40,
                                                 min_xdistance=rmin, min_ydistance=rmin)
        for a, x, y, rad in zip(acc, cxs, cys, rads):
            if a >= 0.35:
                add("Ring", y * f, x * f, rad=float(rad * f), size=2 * rad * f * res)

    for i, ft in enumerate(feats, 1):
        ft["ID"] = i
    return feats


def features_df(feats, crs):
    if not feats:
        return pd.DataFrame()
    df = pd.DataFrame(feats).drop(columns=GEOM_KEYS)
    lat, lon = to_latlon(df["X"].to_numpy(), df["Y"].to_numpy(), crs)
    if lat is not None:
        df["Lat"], df["Lon"] = np.round(lat, 6), np.round(lon, 6)
        df["Google Maps"] = [maps_link(a, b) for a, b in zip(lat, lon)]
    first = ["ID", "Typ", "Einstufung", "Größe_m", "Höhe_m", "Volumen_m3", "Hang_bg_deg", "X", "Y"]
    return df[first + [c for c in df.columns if c not in first]]


# =============================================================================
# Flutungs-Simulation (Badewannen-Modell, keine Strömungssimulation)
# =============================================================================
def flood_wet(z, level, mode, seed_rc=None):
    """Maske der überfluteten Zellen für einen Wasserstand."""
    below = z < level
    if mode == FLOOD_MODES[0] or not below.any():
        return below
    lab, _ = ndimage.label(below)
    if mode == FLOOD_MODES[1]:
        ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    else:
        r, c = seed_rc if seed_rc is not None else (-1, -1)
        ok = 0 <= r < z.shape[0] and 0 <= c < z.shape[1]
        ids = np.array([lab[r, c]]) if ok else np.array([0])
    ids = ids[ids > 0]
    return np.isin(lab, ids) if ids.size else np.zeros_like(below)


@st.cache_resource(max_entries=3, show_spinner="Berechne Überflutung …")
def compute_flood(flood_key, _z, level, mode, seed_rc):
    wet = flood_wet(_z, level, mode, seed_rc)
    depth = np.where(wet, level - _z, 0).astype(np.float32)
    return wet, depth


def _water_rgba(w, d, alpha, dref=None):
    """Wasser als RGBA: hellblau (flach) bis dunkelblau (tief), trockene Zellen transparent."""
    if dref is None:
        dref = float(np.percentile(d[w], 95)) if w.any() else 1.0
    t = np.clip(d / max(dref, 1e-3), 0, 1)
    rgba = WATER_CMAP(t, bytes=True)
    rgba[..., 3] = np.where(w, int(round(255 * alpha)), 0).astype(np.uint8)
    return rgba


def _water_png_raw(w, d, alpha):
    return png_bytes(_water_rgba(w, d, alpha))


@st.cache_resource(max_entries=12)
def water_png(key, _wet, _depth, max_dim, alpha):
    """Nur Wasser (transparent) – für die Karte."""
    w, s = stride_ds(_wet, max_dim)
    return _water_png_raw(w, _depth[::s, ::s], alpha)


@st.cache_resource(max_entries=6)
def flood_png(key, _L, _grid, _wet, _depth, stretch, hide, max_dim, alpha):
    """Relief + Wasser zu EINEM deckenden Bild verschmolzen."""
    base_arr, s = display_array(_L, _grid, max_dim, hide)
    base_rgba = to_rgba(base_arr, _L.cmap, *vrange(_L, stretch))
    water = _water_rgba(_wet[::s, ::s], _depth[::s, ::s], alpha)
    return png_bytes(composite_rgba(base_rgba, water))


@st.cache_resource(max_entries=3)
def shore_lines(flood_key, _wet, _grid, max_dim=1200):
    """Uferlinien als Liste von (x, y)-Arrays in Weltkoordinaten."""
    if not SKIMAGE_OK or not _wet.any() or _wet.all():
        return []
    w, s = stride_ds(_wet, max_dim)
    lines = []
    for c in sorted(find_contours(w.astype(np.float32), 0.5), key=len, reverse=True)[:400]:
        if len(c) < 6:
            continue
        x, y = _grid.pixel_to_world(c[:, 0] * s, c[:, 1] * s)
        lines.append((x, y))
    return lines


def shore_trace(lines):
    lx, ly = [], []
    for x, y in lines:
        lx += list(x) + [None]
        ly += list(y) + [None]
    return go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="#ffd400", width=1.6),
                      name="Uferlinie", hoverinfo="skip")


def flood_figure(title, base_name, wet, depth, lines, show_shore, height):
    fig = go.Figure()
    png = flood_png((models_key, flood_key, base_name), layers[base_name], grid, wet, depth,
                    tuple(S.stretch), bool(S.hide_voids), int(S.view_px), float(S.water_opacity))
    fig.add_layout_image(source=data_uri(png), opacity=1.0, **img_extent(grid))
    hz, s = stride_ds(grid.z, 300)
    hw, hd = wet[::s, ::s], depth[::s, ::s]
    txt = np.where(hw, np.char.mod("Wassertiefe %.2f m", hd), "trocken")
    xs, ys = ds_axes(grid, hz.shape, s)
    fig.add_trace(go.Heatmap(z=hz, x=xs, y=ys, text=txt.tolist(), opacity=0, showscale=False,
                             showlegend=False,
                             hovertemplate="X %{x:.1f} · Y %{y:.1f}<br>Höhe %{z:.2f} m<br>%{text}<extra></extra>"))
    if show_shore and lines:
        fig.add_trace(shore_trace(lines))
    return apply_xy_layout(fig, title, height)


@st.cache_resource(max_entries=2)
def hypsometry(key, _z, res, n=200):
    """Flutkurve (Badewanne): Fläche und Volumen in Abhängigkeit vom Wasserstand."""
    zs = np.sort(_z.ravel()).astype(np.float64)
    cums = np.concatenate([[0.0], np.cumsum(zs)])
    levels = np.linspace(zs[0], zs[-1], n)
    cnt = np.searchsorted(zs, levels, side="left")
    return pd.DataFrame({"wasserstand_m": levels, "flaeche_ha": cnt * res**2 / 1e4,
                         "anteil_prozent": cnt / zs.size * 100,
                         "volumen_m3": (levels * cnt - cums[cnt]) * res**2})


@st.cache_resource(max_entries=2, show_spinner="Suche Höhen mit gehäuften Geländekanten …")
def terrace_signal(key, _z, res, bin_m):
    """Relative Häufigkeit von Hangknicken und Ebenen je Höhenstufe (1 = Durchschnitt)."""
    zs = ndimage.gaussian_filter(_z, max(1.0, 2.0 / res))
    dzdx, dzdy = gradients(zs, res)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    curv = np.abs(ndimage.laplace(zs)) / res**2
    thr = np.percentile(stride_ds(curv, 1000)[0], 95)
    edges = (curv > thr) & (slope > 1.0)
    flat = slope < 2.0
    zmin_, zmax_ = float(_z.min()), float(_z.max())
    bin_m = max(bin_m, (zmax_ - zmin_) / 2000)
    bins = np.arange(math.floor(zmin_ / bin_m) * bin_m, zmax_ + bin_m, bin_m)
    if len(bins) < 3:
        bins = np.linspace(zmin_, zmax_ + 1e-6, 3)
    n_all = np.histogram(_z, bins)[0].astype(float)
    n_e = np.histogram(_z[edges], bins)[0]
    n_f = np.histogram(_z[flat], bins)[0]
    valid = n_all >= max(30, 0.0005 * _z.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_e = (n_e / n_all) / max(edges.mean(), 1e-9)
        rel_f = (n_f / n_all) / max(flat.mean(), 1e-9)
    rel_e[~valid] = np.nan
    rel_f[~valid] = np.nan
    return pd.DataFrame({"hoehe_m": (bins[:-1] + bins[1:]) / 2, "kantendichte": rel_e,
                         "ebenen_anteil": rel_f, "pixel": n_all})


def lines_geojson(lines, crs, wgs84, props):
    out = []
    for x, y in lines:
        if wgs84:
            lat, lon = to_latlon(np.asarray(x), np.asarray(y), crs)
            if lat is None:
                continue
            x, y = lon, lat
        out.append({"type": "Feature", "properties": props,
                    "geometry": {"type": "LineString",
                                 "coordinates": [[float(a), float(b)] for a, b in zip(x, y)]}})
    fc = {"type": "FeatureCollection", "features": out}
    if not wgs84 and crs and crs.startswith("EPSG:"):
        fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs[5:]}"}}
    return json.dumps(fc, ensure_ascii=False).encode("utf-8")


# --- Animation -----------------------------------------------------------------
def flood_frames(base_name, lo, hi, steps, mode, seed_rc, alpha, max_dim=600):
    """Liste von (Wasserstand, RGBA-Bild) – Relief und Wasser bereits verschmolzen."""
    L = layers[base_name]
    base_arr, s = display_array(L, grid, max_dim, S.hide_voids)
    base_rgba = to_rgba(base_arr, L.cmap, *vrange(L, S.stretch))
    zs = grid.z[::s, ::s]
    seed_s = (seed_rc[0] // s, seed_rc[1] // s) if seed_rc else None
    dref = max(hi - lo, 0.5)
    out = []
    for lv in np.linspace(lo, hi, steps):
        w = flood_wet(zs, lv, mode, seed_s)
        d = np.where(w, lv - zs, 0).astype(np.float32)
        out.append((float(lv), composite_rgba(base_rgba, _water_rgba(w, d, alpha, dref))))
    return out


def flood_anim_figure(frames, duration_ms):
    """Plotly-Animation mit Play/Pause und Schieberegler UNTER dem Bild."""
    ext = img_extent(grid)
    imgs = [dict(source=data_uri(png_bytes(rgba)), opacity=1.0, **ext) for _, rgba in frames]
    fr = [go.Frame(name=f"{lv:.3f}", layout=dict(images=[im], title=dict(text=f"Wasserstand {lv:.2f} m")))
          for (lv, _), im in zip(frames, imgs)]
    fig = go.Figure(data=[go.Scatter(x=[grid.x_min, grid.x_max], y=[grid.y_min, grid.y_max],
                                     mode="markers", opacity=0, hoverinfo="skip", showlegend=False)],
                    frames=fr)
    apply_xy_layout(fig, f"Wasserstand {frames[0][0]:.2f} m", S.view_height)
    play = dict(label="▶ Abspielen", method="animate",
                args=[None, dict(frame=dict(duration=int(duration_ms), redraw=True), fromcurrent=True,
                                 transition=dict(duration=0), mode="immediate")])
    pause = dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0),
                                    mode="immediate")])
    fig.update_layout(
        images=[imgs[0]],
        margin=dict(l=10, r=10, t=50, b=140),
        xaxis_title=None,
        updatemenus=[dict(type="buttons", direction="left", showactive=False,
                          x=0.0, xanchor="left", y=-0.03, yanchor="top",
                          pad=dict(r=10, t=10), buttons=[play, pause])],
        sliders=[dict(active=0, x=0.25, len=0.75, xanchor="left", y=-0.03, yanchor="top",
                      pad=dict(t=10, b=10),
                      currentvalue=dict(prefix="Wasserstand: ", suffix=" m", visible=True),
                      steps=[dict(method="animate", label=f"{lv:.1f}",
                                  args=[[f"{lv:.3f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                            transition=dict(duration=0))])
                             for lv, _ in frames])])
    return fig


def _gif_font(size):
    try:
        return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


def frames_to_gif(frames, duration_ms):
    """Animiertes GIF aus den Animationsbildern (mit eingeblendetem Wasserstand)."""
    pil_frames = []
    font = None
    for lv, rgba in frames:
        img = Image.fromarray(rgba, "RGBA").convert("RGB")
        if font is None:
            font = _gif_font(max(14, img.width // 30))
        draw = ImageDraw.Draw(img)
        txt = f"Wasserstand {lv:.2f} m"
        bx = draw.textbbox((0, 0), txt, font=font)
        pad = 8
        draw.rectangle([10, 10, 10 + bx[2] - bx[0] + 2 * pad, 10 + bx[3] - bx[1] + 2 * pad], fill=(40, 40, 40))
        draw.text((10 + pad - bx[0], 10 + pad - bx[1]), txt, fill=(255, 255, 255), font=font)
        pil_frames.append(img.quantize(colors=255))
    durations = [int(duration_ms)] * (len(pil_frames) - 1) + [int(duration_ms) * 4]
    buf = io.BytesIO()
    pil_frames[0].save(buf, format="GIF", save_all=True, append_images=pil_frames[1:],
                       duration=durations, loop=0)
    return buf.getvalue()


# =============================================================================
# Regionales Höhenmodell (AWS Terrain Tiles, Terrarium-Format) für die Karte
# =============================================================================
def lonlat_to_tile(lon, lat, z):
    n = 2.0 ** z
    lon = np.asarray(lon, float)
    lat = np.clip(np.asarray(lat, float), -85.0, 85.0)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x, y, z):
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


@st.cache_data(show_spinner=False, max_entries=200)
def fetch_terrarium(z, x, y):
    r = requests.get(TERRARIUM_URL.format(z=z, x=x, y=y), timeout=20)
    r.raise_for_status()
    a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float32)
    return (a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0).astype(np.float32)


@st.cache_data(show_spinner="Lade regionales Höhenmodell …", max_entries=4)
def regional_dem(lat, lon, radius_km, max_tiles=36):
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(math.cos(math.radians(lat)), 0.05))
    for z in range(13, 4, -1):
        x0, y0 = lonlat_to_tile(lon - dlon, lat + dlat, z)
        x1, y1 = lonlat_to_tile(lon + dlon, lat - dlat, z)
        tx0, ty0 = int(math.floor(float(x0))), int(math.floor(float(y0)))
        tx1, ty1 = int(math.floor(float(x1))), int(math.floor(float(y1)))
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= max_tiles:
            break
    rows = [np.hstack([fetch_terrarium(z, tx, ty) for tx in range(tx0, tx1 + 1)])
            for ty in range(ty0, ty1 + 1)]
    dem = np.vstack(rows)
    lon_w, lat_n = tile_to_lonlat(tx0, ty0, z)
    lon_e, lat_s = tile_to_lonlat(tx1 + 1, ty1 + 1, z)
    res_m = 156543.03 * math.cos(math.radians(lat)) / 2 ** z
    meta = dict(z=z, tx0=tx0, ty0=ty0, bounds=(lon_w, lat_n, lon_e, lat_s), res_m=res_m)
    return dem, meta


def dem_rowcol(meta, lon, lat):
    px, py = lonlat_to_tile(lon, lat, meta["z"])
    return (py - meta["ty0"]) * 256 - 0.5, (px - meta["tx0"]) * 256 - 0.5


@st.cache_resource(max_entries=4)
def regional_offset(key, _grid, crs, lat, lon, radius_km):
    """Median-Höhendifferenz LiDAR-Kachel minus Regionalmodell (im Bereich der Kachel)."""
    dem, meta = regional_dem(lat, lon, radius_km)
    zs, s = stride_ds(_grid.z, 40)
    xs, ys = ds_axes(_grid, zs.shape, s)
    X, Y = np.meshgrid(xs, ys)
    la, lo = to_latlon(X.ravel(), Y.ravel(), crs)
    if la is None:
        return 0.0
    r, c = dem_rowcol(meta, lo, la)
    d = ndimage.map_coordinates(dem, [r, c], order=1, mode="nearest")
    return float(np.median(zs.ravel() - d))


@st.cache_data(show_spinner="Berechne regionale Überflutung …", max_entries=6)
def regional_water(lat, lon, radius_km, level, offset, mode_idx, seed_ll):
    dem, meta = regional_dem(lat, lon, radius_km)
    zz = dem + offset
    seed = None
    if mode_idx == 2 and seed_ll:
        r, c = dem_rowcol(meta, seed_ll[0], seed_ll[1])
        seed = (int(round(float(r))), int(round(float(c))))
    wet = flood_wet(zz, level, FLOOD_MODES[mode_idx], seed)
    depth = np.where(wet, level - zz, 0).astype(np.float32)
    area_km2 = float(wet.sum()) * meta["res_m"] ** 2 / 1e6
    return _water_png_raw(wet, depth, 1.0), meta["bounds"], area_km2, meta["res_m"]


# =============================================================================
# Export
# =============================================================================
def geotiff_bytes(arr, grid, crs):
    data = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                      dtype="float32", crs=crs, nodata=-9999.0, compress="deflate",
                      transform=from_origin(grid.x_min, grid.y_max, grid.res, grid.res)) as dst:
            dst.write(data, 1)
        return mem.read()


def features_geojson(feats, grid, crs, wgs84):
    out = []
    for f in feats:
        if f["Typ"] == "Linie":
            xs, ys = grid.pixel_to_world([f["r0"], f["r1"]], [f["c0"], f["c1"]])
        else:
            cx, cy = grid.pixel_to_world(f["r0"], f["c0"])
            rad = f["rad_px"] * grid.res
            xs, ys = cx + rad * np.cos(CIRCLE_T), cy + rad * np.sin(CIRCLE_T)
        if wgs84:
            lat, lon = to_latlon(np.asarray(xs), np.asarray(ys), crs)
            xs, ys = lon, lat
        coords = [[float(a), float(b)] for a, b in zip(xs, ys)]
        geom = ({"type": "LineString", "coordinates": coords} if f["Typ"] == "Linie"
                else {"type": "Polygon", "coordinates": [coords]})
        props = {k: f[k] for k in ("ID", "Typ", "Einstufung", "Größe_m", "Höhe_m", "Volumen_m3", "Hang_bg_deg")}
        out.append({"type": "Feature", "geometry": geom, "properties": props})
    fc = {"type": "FeatureCollection", "features": out}
    if not wgs84 and crs and crs.startswith("EPSG:"):
        fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{crs[5:]}"}}
    return json.dumps(fc, ensure_ascii=False).encode("utf-8")


def lazy_download(label, spec, make, file_name, mime):
    """Erzeugt teure Exporte erst auf Knopfdruck und merkt sie sich."""
    sk = f"_dl_{label}"
    if st.button(label, key=f"_btn_{label}"):
        try:
            with st.spinner("Erzeuge Datei …"):
                S[sk] = (spec, make())
        except Exception as e:
            st.error(f"Export fehlgeschlagen: {e}")
    saved = S.get(sk)
    if saved and saved[0] == spec:
        st.download_button(f"⬇️ {file_name}", saved[1], file_name=file_name, mime=mime, key=f"_d_{label}")


def build_report(names, fdf):
    lat, lon = latlon
    loc = f"<a href='{maps_link(lat, lon)}'>{lat:.5f}, {lon:.5f}</a>" if lat is not None else "–"
    info = {"Dateien": ", ".join(f.name for f in files), "Quelle": data["source"],
            "Raster": f"{grid.w} × {grid.h} px", "Auflösung": f"{grid.res:.2f} m",
            "Fläche": f"{grid.w * grid.h * grid.res**2 / 1e4:.1f} ha", "CRS": html.escape(str(crs)),
            "Zentrum (WGS84)": loc}
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in info.items())
    prow = "".join(f"<tr><th>{k}</th><td>{html.escape(str(S[k]))}</td></tr>" for k in DEFAULTS)
    imgs = "".join(
        f"<div><h3>{html.escape(n)}</h3><img src='"
        f"{data_uri(layer_png((models_key, n), layers[n], grid, tuple(S.stretch), S.hide_voids, 1200))}'>"
        f"<p>{html.escape(layers[n].desc)}</p></div>" for n in names)
    ftab = fdf.to_html(index=False, render_links=True) if len(fdf) else "<p>Keine Features erkannt.</p>"
    css = ("body{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:20px}"
           "img{max-width:100%;border:1px solid #ccc}table{border-collapse:collapse}"
           "td,th{border:1px solid #ccc;padding:3px 8px;font-size:12px;text-align:left}"
           ".g{display:grid;grid-template-columns:1fr 1fr;gap:12px}")
    return (f"<html><head><meta charset='utf-8'><title>LiDAR-Bericht</title><style>{css}</style></head><body>"
            f"<h1>🏛️ LiDAR-Prospektionsbericht</h1><p>Erstellt am {datetime.now():%d.%m.%Y %H:%M}</p>"
            f"<h2>Datengrundlage</h2><table>{rows}</table><h2>Reliefmodelle</h2><div class='g'>{imgs}</div>"
            f"<h2>Erkannte Strukturen</h2>{ftab}<p><i>Automatische, regelbasierte Vorauswahl – "
            f"ersetzt keine archäologische Prüfung im Gelände.</i></p>"
            f"<h2>Parameter</h2><table>{prow}</table></body></html>").encode("utf-8")


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Steuerung")
    files = st.file_uploader("📂 Dateien (Punktwolke(n) oder ein Raster)", accept_multiple_files=True,
                             type=[e[1:] for e in POINT_EXT + RASTER_EXT],
                             help="Mehrere LAS/LAZ/XYZ-Kacheln werden zu einem Mosaik zusammengesetzt.")
    with st.expander("💾 Projekt laden"):
        proj = st.file_uploader("Projektdatei (.json)", type=["json"])
        if proj is not None and st.button("Parameter übernehmen"):
            try:
                for k, v in json.loads(proj.getvalue()).get("parameter", {}).items():
                    if k in DEFAULTS:
                        S[k] = tuple(v) if isinstance(DEFAULTS[k], tuple) else type(DEFAULTS[k])(v)
                st.success("Parameter übernommen.")
            except Exception as e:
                st.error(f"Projektdatei ungültig: {e}")

    with st.expander("🌍 Geo-Referenz"):
        st.checkbox("CRS aus Datei verwenden (falls vorhanden)", key="use_file_crs")
        st.number_input("EPSG-Code (Fallback)", min_value=1, max_value=999999, step=1, key="epsg",
                        help="25832 = UTM 32N · 25833 = UTM 33N · 2154 = Lambert-93 (IGN) · 2056 = CH LV95")
        if not PYPROJ_OK:
            st.warning("'pyproj' fehlt – Karte und Koordinaten deaktiviert.")

    with st.expander("☁️ Punktwolke → Raster"):
        st.number_input("Auflösung (m)", min_value=0.1, max_value=10.0, step=0.1, key="grid_res")
        st.selectbox("Aggregation pro Zelle", ["Mittelwert", "Minimum (robuster bei Vegetation)"], key="agg_mode")
        st.checkbox("Nur Bodenpunkte (Klasse 2)", key="ground_only")
        st.checkbox("Morphologischer Vegetationsfilter", key="morph_filter",
                    help="Entfernt Zellen, die deutlich über dem lokalen Minimum liegen (Bäume, Gebäude).")
        st.slider("Filterfenster (m)", 3.0, 50.0, step=1.0, key="morph_window_m", disabled=not S.morph_filter)
        st.slider("Filterschwelle (m)", 0.2, 5.0, step=0.1, key="morph_threshold_m", disabled=not S.morph_filter)

    with st.expander("🏔️ Reliefmodelle", expanded=True):
        st.slider("Licht-Azimut (°)", 0, 360, step=5, key="hs_azimuth",
                  help="Gilt für Hillshade und für die Lichtrichtung in 3D.")
        st.slider("Sonnenhöhe (°)", 5, 80, key="hs_altitude", help="Flaches Licht (10–25°) betont feine Wälle.")
        st.slider("LRM-Glättung σ (m)", 2.0, 60.0, step=1.0, key="lrm_sigma_m")
        st.checkbox("SVF & Openness berechnen (rechenintensiv)", key="enable_svf")
        st.slider("Richtungen", 4, 16, key="svf_dirs", disabled=not S.enable_svf)
        st.slider("Suchradius (m)", 5, 50, key="svf_radius_m", disabled=not S.enable_svf)

    with st.expander("🎨 Anzeige"):
        st.slider("Kontrast (Perzentile)", 0.0, 100.0, step=0.5, key="stretch",
                  help="Enger (z. B. 5–95) = kräftigerer Kontrast. Gilt für 2D, 3D, Karte und Bericht.")
        st.checkbox("Große Datenlücken ausblenden", key="hide_voids")
        st.slider("Z-Überhöhung (3D)", 0.5, 10.0, step=0.5, key="z_exag")
        st.slider("Bildhöhe (px)", 400, 1600, step=50, key="view_height",
                  help="Höhe der interaktiven Ansicht, der Karte und der 3D-Ansicht.")
        st.slider("Bildauflösung (px)", 800, 3000, step=100, key="view_px",
                  help="Höher = schärfer beim Hineinzoomen, aber langsamer.")

    with st.expander("🔍 Feature-Erkennung"):
        st.checkbox("Lineare Strukturen", key="det_lines")
        st.slider("Min. Linienlänge (m)", 5.0, 200.0, step=5.0, key="line_min_m", disabled=not S.det_lines)
        st.checkbox("Hügel & Senken", key="det_blobs")
        st.slider("Durchmesser (m)", 2.0, 80.0, step=1.0, key="blob_d", disabled=not S.det_blobs)
        st.slider("Min. Höhe/Tiefe (m)", 0.05, 2.0, step=0.05, key="blob_min_h", disabled=not S.det_blobs)
        st.checkbox("Ringstrukturen", key="det_rings")
        st.slider("Ring-Durchmesser (m)", 5.0, 150.0, step=5.0, key="ring_d", disabled=not S.det_rings)

    st.caption(f"LiDAR Archäologie Pro · Version {APP_VERSION}")


# =============================================================================
# Hauptbereich
# =============================================================================
st.markdown("""<div class="app-header"><h1>🏛️ LiDAR Archäologie Pro</h1>
<p>Reliefanalyse · Prospektion · Flutungssimulation</p></div>""", unsafe_allow_html=True)

if not files:
    cards = [("📂", "1 · Daten laden", "Links in der Sidebar eine LAS/LAZ-Punktwolke oder ein "
                                      "Höhenmodell (GeoTIFF, ASC) hochladen."),
             ("🏔️", "2 · Relief erkunden", "Reliefmodelle in 2D, 3D und auf der Karte vergleichen, "
                                         "Profile ziehen, Strukturen erkennen."),
             ("🌊", "3 · Flutung simulieren", "Wasserstand verändern, Uferlinien prüfen, Strandlinien suchen "
                                            "und Ergebnisse exportieren.")]
    for col, (icon, title, text) in zip(st.columns(3), cards):
        with col.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.write(text)
    st.info("⬅️ Starte mit einer Datei in der Sidebar.")
    st.stop()

file_key = tuple(sorted((f.name, f.size) for f in files))
use_raster = any(file_ext(f.name) in RASTER_EXT for f in files)
pc = None if use_raster else (float(S.grid_res), S.agg_mode, bool(S.ground_only))
morph = (float(S.morph_window_m), float(S.morph_threshold_m)) if S.morph_filter else None
grid_key = (file_key, pc, morph)

data = None
try:
    data = load_grid(grid_key, files)
except Exception as e:
    st.error(f"Fehler beim Laden: {e}")
    with st.expander("Technische Details"):
        st.exception(e)
if data is None:
    st.stop()

grid = GeoGrid(data["z"], data["void"], data["x_min"], data["y_max"], data["res"])
crs = data["crs"] if (S.use_file_crs and data["crs"]) else f"EPSG:{int(S.epsg)}"
latlon = to_latlon((grid.x_min + grid.x_max) / 2, (grid.y_min + grid.y_max) / 2, crs)

m = st.columns(5)
m[0].metric("Quelle", data["source"])
m[1].metric("Raster", f"{grid.w} × {grid.h}")
m[2].metric("Auflösung", f"{grid.res:.2f} m")
m[3].metric("Fläche", f"{grid.w * grid.h * grid.res**2 / 1e4:.1f} ha")
m[4].metric("CRS", crs if len(str(crs)) < 15 else "WKT")
if latlon[0] is not None:
    st.markdown(f"**📍 Zentrum:** [{latlon[0]:.5f}, {latlon[1]:.5f}]({maps_link(*latlon)})"
                + (" · CRS aus Datei" if crs == data["crs"] else " · CRS aus Sidebar"))
if data["notes"]:
    with st.expander(f"ℹ️ Hinweise zum Import ({len(data['notes'])})"):
        for n in data["notes"]:
            st.write("• " + n)

models_key = (grid_key, S.hs_azimuth, S.hs_altitude, float(S.lrm_sigma_m), S.enable_svf,
              S.svf_dirs if S.enable_svf else None, S.svf_radius_m if S.enable_svf else None)
DET_KEYS = ("det_lines", "line_min_m", "det_blobs", "blob_d", "blob_min_h", "det_rings", "ring_d")
det_params = {k: S[k] for k in DET_KEYS}
feat_key = (models_key, tuple((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in det_params.items()))


def current_features():
    d = S.get("features")
    return d["items"] if d and d["key"] == feat_key else []


def set_level(v):
    """Callback: Wasserstand setzen (auf Geländebereich begrenzt)."""
    S["water_level"] = float(min(max(v, S["_zmin"]), S["_zmax"]))


def shift_level(d):
    set_level(S["water_level"] + d)


try:
    layers, extras = compute_models(models_key, grid.z, grid.res, S.hs_azimuth, S.hs_altitude,
                                    float(S.lrm_sigma_m), S.enable_svf, S.svf_dirs, S.svf_radius_m)
    names = list(layers)
    feats = current_features()

    # --- Flutung vorbereiten (wird in Karte, Profil und Tab "Flutung" genutzt) ---
    zmin, zmax = float(grid.z.min()), float(grid.z.max())
    if zmax - zmin < 1e-3:
        zmax = zmin + 1.0
    S["_zmin"], S["_zmax"] = zmin, zmax
    wl = S.get("water_level")
    if wl is None or not (zmin <= wl <= zmax):
        S["water_level"] = float(min(max(round(zmin + 0.2 * (zmax - zmin), 2), zmin), zmax))
    cx0, cy0 = (grid.x_min + grid.x_max) / 2, (grid.y_min + grid.y_max) / 2
    if not (grid.x_min <= S.get("seed_x", -1e18) <= grid.x_max):
        S["seed_x"] = float(round(cx0, 1))
    if not (grid.y_min <= S.get("seed_y", -1e18) <= grid.y_max):
        S["seed_y"] = float(round(cy0, 1))
    _r, _c = grid.world_to_pixel(S.seed_x, S.seed_y)
    seed_rc = (int(round(float(_r))), int(round(float(_c))))
    flood_mode = S.flood_mode if S.flood_mode in FLOOD_MODES else FLOOD_MODES[0]
    flood_key = (grid_key, round(float(S.water_level), 3), flood_mode,
                 seed_rc if flood_mode == FLOOD_MODES[2] else None)
    wet, depth = compute_flood(flood_key, grid.z, float(S.water_level), flood_mode, seed_rc)

    t2d, t3d, tmap, tprof, tfeat, tflood, texp = st.tabs(
        ["🖼️ 2D", "🌐 3D", "🗺️ Karte", "📈 Profil", "🔍 Features", "🌊 Flutung", "📦 Export"])

    # ---------------- 2D ----------------
    with t2d:
        mode = st.radio("Ansicht", ["Galerie", "Einzelbild", "Interaktiv", "Überblendung"],
                        horizontal=True, key="mode2d")
        if mode == "Galerie":
            ncol = st.select_slider("Bilder pro Zeile", options=[1, 2, 3], value=2, key="gal_cols")
            cols = st.columns(ncol)
            for i, n in enumerate(names):
                with cols[i % ncol]:
                    show_image(layer_png((models_key, n), layers[n], grid, tuple(S.stretch),
                                         S.hide_voids, 1200), caption=n)
            st.caption("💡 Mit der Maus über ein Bild fahren → oben rechts erscheint das Vollbild-Symbol.")
        elif mode == "Einzelbild":
            sel = st.selectbox("Modell", names, key="sel_single")
            st.caption(layers[sel].desc)
            show_image(layer_png((models_key, sel), layers[sel], grid, tuple(S.stretch),
                                 S.hide_voids, int(S.view_px)), caption=sel)
        elif mode == "Interaktiv":
            sel = st.selectbox("Modell", names, key="sel2d")
            show_f = st.checkbox(f"Erkannte Features einblenden ({len(feats)})", value=bool(feats))
            st.caption(f"{layers[sel].desc}  ·  🖱️ Mausrad = Zoom · Ziehen = Verschieben · "
                       "Doppelklick = Gesamtansicht")
            show_plot(layer_figure(sel, layers[sel], grid, feats=feats if show_f else None))
        else:
            c1, c2, c3 = st.columns(3)
            a = c1.selectbox("Basis", names, index=names.index("Hillshade"), key="blend_a")
            b = c2.selectbox("Überlagerung", names, index=names.index("Restrelief (LRM)"), key="blend_b")
            op = c3.slider("Deckkraft", 0.0, 1.0, 0.5, 0.05, key="blend_op")
            st.caption("🖱️ Mausrad = Zoom · Ziehen = Verschieben · Doppelklick = Gesamtansicht")
            show_plot(layer_figure(f"{a} + {b}", layers[a], grid, overlay=layers[b], opacity=op))

    # ---------------- 3D ----------------
    with t3d:
        c1, c2, c3 = st.columns([2, 1, 1])
        tex = c1.selectbox("Textur", names, key="tex3d")
        npts = c2.select_slider("Detailgrad (Punkte)", [100_000, 250_000, 400_000, 600_000], value=250_000)
        shade = c3.slider("Schattierung", 0.0, 1.0, step=0.05, key="shade_3d",
                          help="0 = flach (nur Textur) · 1 = sehr plastisch. Lichtrichtung = Sidebar.")
        s = max(1, math.ceil(math.sqrt(grid.w * grid.h / npts)))
        zp, tp = grid.z[::s, ::s].copy(), layers[tex].raw[::s, ::s].astype(np.float32, copy=True)
        if S.hide_voids and grid.void.any():
            zp[grid.void[::s, ::s]] = np.nan
        xs, ys = ds_axes(grid, zp.shape, s)
        dx, dy = grid.w * grid.res, grid.h * grid.res
        zr = float(np.nanmax(zp) - np.nanmin(zp)) or 1.0
        vmin, vmax = vrange(layers[tex], S.stretch)
        az_r, alt_r = math.radians(S.hs_azimuth), math.radians(S.hs_altitude)
        light_pos = dict(x=1e5 * math.sin(az_r) * math.cos(alt_r),
                         y=1e5 * math.cos(az_r) * math.cos(alt_r),
                         z=1e5 * math.sin(alt_r))
        lighting = dict(ambient=0.9 - 0.6 * shade, diffuse=0.1 + 0.85 * shade,
                        specular=0.05, roughness=0.9, fresnel=0.05)
        fig3d = go.Figure(go.Surface(
            x=xs, y=ys, z=zp, surfacecolor=tp, colorscale=plotly_cs(layers[tex].cmap), cmin=vmin, cmax=vmax,
            showscale=bool(layers[tex].unit), lighting=lighting, lightposition=light_pos,
            hovertemplate="X %{x:.1f} · Y %{y:.1f}<br>Höhe %{z:.2f} m<extra></extra>"))
        fig3d.update_layout(height=int(S.view_height), margin=dict(l=0, r=0, t=30, b=0), title=f"3D: {tex}",
                            scene=dict(aspectmode="manual",
                                       aspectratio=dict(x=1, y=dy / dx, z=max(zr / dx * S.z_exag, 0.02)),
                                       xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Höhe (m)"))
        show_plot(fig3d)
        st.caption(f"Jeder {s}. Pixel dargestellt · Z-Überhöhung {S.z_exag}× · Kontrast {S.stretch[0]:.0f}–"
                   f"{S.stretch[1]:.0f} % (Sidebar). 💡 Für mehr Kontrast: Kontrast enger stellen (z. B. 5–95) "
                   "oder Schattierung erhöhen.")

    # ---------------- Karte ----------------
    with tmap:
        if latlon[0] is None:
            st.warning("Keine Umrechnung nach WGS84 möglich – 'pyproj' installieren und EPSG prüfen.")
        else:
            c1, c2, c3 = st.columns(3)
            mlayer = c1.selectbox("Overlay", names, key="map_layer")
            base = c2.radio("Hintergrund", ["OpenStreetMap", "Satellit (Esri)", "Ohne"], horizontal=True)
            op = c3.slider("Deckkraft Overlay", 0.0, 1.0, 0.7, 0.05, key="map_op")
            custom = st.text_input("Eigener XYZ/WMS-Layer (optional, z. B. historische Karte)",
                                   help="XYZ: https://…/{z}/{x}/{y}.png · WMS: GetMap-URL mit "
                                        "BBOX={bbox-epsg-3857}&SRS=EPSG:3857&WIDTH=256&HEIGHT=256")
            st.markdown(f"**🌊 Wasser auf der Karte** – Wasserstand **{S.water_level:.2f} m** "
                        "(einstellbar im Tab „Flutung“)")
            w1, w2 = st.columns(2)
            w1.checkbox("In der Kachel (LiDAR, genau)", key="map_water_tile")
            w2.checkbox("In der ganzen Region (grobes Höhenmodell)", key="map_water_region")

            region = None
            if S.map_water_region:
                r1, r2, r3 = st.columns(3)
                r1.slider("Regionsradius (km)", 2, 50, step=1, key="region_km")
                r2.selectbox("Ausbreitung", REGION_MODES, key="region_mode")
                r3.checkbox("Höhenversatz zur Kachel ausgleichen", key="region_offset_on",
                            help="Vergleicht das Regionalmodell mit deinen LiDAR-Daten im Bereich der Kachel "
                                 "und gleicht den mittleren Höhenunterschied aus.")
                if not (REQUESTS_OK and PIL_OK):
                    st.warning("Für das Regionalmodell werden 'requests' und 'Pillow' benötigt.")
                else:
                    try:
                        lat_c, lon_c = round(float(latlon[0]), 4), round(float(latlon[1]), 4)
                        rkm = int(S.region_km)
                        off = (regional_offset((grid_key, str(crs), lat_c, lon_c, rkm), grid, crs,
                                               lat_c, lon_c, rkm) if S.region_offset_on else 0.0)
                        rmode = S.region_mode if S.region_mode in REGION_MODES else REGION_MODES[0]
                        sll = to_latlon(S.seed_x, S.seed_y, crs)
                        seed_ll = ((round(float(sll[1]), 6), round(float(sll[0]), 6))
                                   if sll[0] is not None else None)
                        rpng, rb, rarea, rres = regional_water(lat_c, lon_c, rkm, round(float(S.water_level), 2),
                                                               round(off, 2), REGION_MODES.index(rmode), seed_ll)
                        region = dict(png=rpng, bounds=rb, area=rarea, res=rres, offset=off)
                    except Exception as e:
                        st.warning(f"Regionales Höhenmodell konnte nicht geladen werden: {e}")

            cx = [grid.x_min, grid.x_max, grid.x_max, grid.x_min]
            cy = [grid.y_max, grid.y_max, grid.y_min, grid.y_min]
            clat, clon = to_latlon(np.array(cx), np.array(cy), crs)
            corners = [[float(a), float(b)] for a, b in zip(clon, clat)]
            uri = data_uri(layer_png((models_key, mlayer), layers[mlayer], grid, tuple(S.stretch),
                                     S.hide_voids, int(S.view_px)))
            mlayers = []
            if base.startswith("Satellit"):
                mlayers.append(dict(sourcetype="raster", source=[ESRI_URL], below="traces",
                                    sourceattribution="Esri World Imagery"))
            if custom.strip():
                mlayers.append(dict(sourcetype="raster", source=[custom.strip()], below="traces"))
            if region:
                lw, ln_, le, ls_ = region["bounds"]
                mlayers.append(dict(sourcetype="image", source=data_uri(region["png"]),
                                    opacity=float(S.water_opacity), below="traces",
                                    coordinates=[[lw, ln_], [le, ln_], [le, ls_], [lw, ls_]]))
            mlayers.append(dict(sourcetype="image", source=uri, opacity=op, below="traces", coordinates=corners))
            if S.map_water_tile and wet.any():
                wuri = data_uri(water_png((flood_key, int(S.view_px), 1.0), wet, depth, int(S.view_px), 1.0))
                mlayers.append(dict(sourcetype="image", source=wuri, opacity=float(S.water_opacity),
                                    below="traces", coordinates=corners))
            use_new = hasattr(go, "Scattermap")
            Scatter = go.Scattermap if use_new else go.Scattermapbox
            fig_m = go.Figure(Scatter(lat=list(clat) + [clat[0]], lon=list(clon) + [clon[0]], mode="lines",
                                      name="LiDAR-Kachel", line=dict(color="#8b5a2b", width=2),
                                      hoverinfo="skip"))
            fdf_map = features_df(feats, crs)
            if len(fdf_map) and "Lat" in fdf_map:
                for label, color in LABEL_COLORS.items():
                    sub = fdf_map[fdf_map["Einstufung"] == label]
                    if len(sub):
                        fig_m.add_trace(Scatter(lat=sub["Lat"], lon=sub["Lon"], mode="markers", name=label,
                                                marker=dict(size=9, color=color),
                                                text=[f"#{i} {t} · {g} m" for i, t, g in
                                                      zip(sub["ID"], sub["Typ"], sub["Größe_m"])]))
            span = (region["bounds"][2] - region["bounds"][0]) if region else max(np.ptp(clon), 1e-6)
            zoom = float(np.clip(math.log2(360 / span) + (0.8 if region else 1.5), 1, 19))
            fig_m.update_layout(**{"map" if use_new else "mapbox": dict(
                style="white-bg" if base != "OpenStreetMap" else "open-street-map",
                center=dict(lat=latlon[0], lon=latlon[1]), zoom=zoom, layers=mlayers)},
                height=int(S.view_height), margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h"))
            show_plot(fig_m)
            if region:
                k1, k2, k3 = st.columns(3)
                k1.metric("Überflutet in der Region", f"{region['area']:,.1f} km²".replace(",", "."))
                k2.metric("Auflösung Regionalmodell", f"≈ {region['res']:.0f} m")
                k3.metric("Höhenversatz Kachel ↔ Region", f"{region['offset']:+.2f} m")
                st.caption("🌍 Regionalmodell: AWS Terrain Tiles (Mapzen, u. a. SRTM). Grobe Auflösung, "
                           "Höhenfehler von mehreren Metern möglich, teils mit Bewuchs und Bebauung. Deiche, "
                           "Dämme und kleine Täler werden nicht erfasst. Innerhalb der Kachel gilt das genauere "
                           "LiDAR-Wasser (Häkchen „In der Kachel“).")
            st.caption("🖱️ Mausrad = Zoom · Ziehen = Verschieben. Das Overlay wird über seine Eckpunkte "
                       "eingepasst (für Kacheln bis einige km ausreichend genau).")

    # ---------------- Profil ----------------
    with tprof:
        st.caption("Höhenprofil entlang einer Linie – z. B. um Wall/Graben oder Hohlwege zu prüfen.")
        ym = (grid.y_min + grid.y_max) / 2
        c = st.columns(4)
        x0 = c[0].number_input("Start X", value=float(grid.x_min + grid.res), format="%.1f")
        y0 = c[1].number_input("Start Y", value=float(ym), format="%.1f")
        x1 = c[2].number_input("Ende X", value=float(grid.x_max - grid.res), format="%.1f")
        y1 = c[3].number_input("Ende Y", value=float(ym), format="%.1f")
        show_w = st.checkbox(f"🌊 Wasserstand einzeichnen ({S.water_level:.2f} m, aus Tab „Flutung“)",
                             value=True, key="prof_water")
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 2 * grid.res:
            st.warning("Profillinie zu kurz.")
        else:
            n = int(min(5000, max(2, 2 * length / grid.res)))
            px, py = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
            rr, cc = grid.world_to_pixel(px, py)
            zs = ndimage.map_coordinates(grid.z, [rr, cc], order=1, mode="nearest")
            ls = ndimage.map_coordinates(extras["lrm"], [rr, cc], order=1, mode="nearest")
            dist = np.linspace(0, length, n)
            a, b = st.columns([1, 2])
            ov = layer_figure("Lage", layers["Hillshade"], grid, max_dim=1000, height=450)
            ov.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines+markers", name="Profil",
                                    line=dict(color="red", width=3)))
            show_plot(ov, a)
            pf = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                               subplot_titles=("Höhe (m)", "Restrelief (m)"))
            pf.add_trace(go.Scatter(x=dist, y=zs, mode="lines", name="Höhe", line=dict(color="saddlebrown")),
                         row=1, col=1)
            if show_w:
                ws = ndimage.map_coordinates(wet.astype(np.uint8), [rr, cc], order=0, mode="nearest")
                yw = np.where(ws > 0, float(S.water_level), np.nan)
                pf.add_trace(go.Scatter(x=dist, y=yw, mode="lines", fill="tonexty", name="Wasser",
                                        line=dict(color="royalblue", width=1),
                                        fillcolor="rgba(65,105,225,0.35)"), row=1, col=1)
                pf.add_hline(y=float(S.water_level), line_dash="dash", line_color="royalblue", row=1, col=1)
            pf.add_trace(go.Scatter(x=dist, y=ls, mode="lines", name="LRM", line=dict(color="firebrick")), row=2, col=1)
            pf.add_hline(y=0, line_dash="dot", row=2, col=1)
            pf.update_layout(height=450, showlegend=False, margin=dict(t=30, b=10), xaxis2_title="Distanz (m)")
            show_plot(pf, b)
            st.download_button("⬇️ Profil als CSV", pd.DataFrame(
                {"distanz_m": dist, "x": px, "y": py, "hoehe_m": zs, "lrm_m": ls}).to_csv(index=False).encode(),
                file_name="profil.csv", mime="text/csv")

    # ---------------- Features ----------------
    with tfeat:
        if not SKIMAGE_OK:
            st.error("Für die Feature-Erkennung wird 'scikit-image' benötigt: pip install scikit-image")
        else:
            st.caption("Erkennung auf dem Restrelief (LRM). Einstellungen in der Sidebar.")
            if st.button("🔍 Erkennung starten", type="primary"):
                with st.spinner("Suche Strukturen …"):
                    S["features"] = {"key": feat_key,
                                     "items": detect_features(extras["lrm"], extras["bg_slope"], grid, det_params)}
                feats = current_features()
            if not feats:
                st.info("Parameter geändert – bitte Erkennung neu starten." if S.get("features")
                        else "Noch keine Erkennung durchgeführt.")
            else:
                types = pd.Series([f["Typ"] for f in feats]).value_counts()
                mc = st.columns(len(types) + 1)
                mc[0].metric("Gesamt", len(feats))
                for i, (t, cnt) in enumerate(types.items(), 1):
                    mc[i].metric(t, int(cnt))
                fbase = st.selectbox("Hintergrund", names, index=names.index("Restrelief (LRM)"), key="feat_base")
                show_plot(layer_figure(f"Features auf {fbase}", layers[fbase], grid, feats=feats))
                st.dataframe(features_df(feats, crs), use_container_width=True, hide_index=True,
                             column_config={"Google Maps": st.column_config.LinkColumn(display_text="Öffnen")})
                st.caption("🔴 auffällig · 🟠 möglich · 🔵 unklar – regelbasierte Vorauswahl (Größe, Höhe, "
                           "lokale Hangneigung), keine archäologische Verifikation.")

    # ---------------- Flutung ----------------
    with tflood:
        st.caption("🌊 Badewannen-Modell: Alles unterhalb des Wasserstands steht unter Wasser. "
                   "Keine Strömung, keine Zeit – nur die Frage: *Wo stünde Wasser bei Höhe X?*")
        rng = zmax - zmin
        wl_step = 0.01 if rng < 5 else 0.05 if rng < 50 else 0.1 if rng < 500 else 0.5
        c1, c2 = st.columns([3, 2])
        c1.slider("Wasserstand (m, Höhenbezug wie in den Daten)", min_value=zmin, max_value=zmax,
                  step=wl_step, key="water_level", format="%.2f")
        with c2:
            st.write("Feineinstellung")
            bcols = st.columns(4)
            for i, (lbl, d) in enumerate([("−1 m", -1.0), ("−0,1", -0.1), ("+0,1", 0.1), ("+1 m", 1.0)]):
                bcols[i].button(lbl, on_click=shift_level, args=(d,), key=f"wl_btn_{i}",
                                use_container_width=True)
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        c1.radio("Modus", FLOOD_MODES, key="flood_mode",
                 help="Badewanne: jede tiefe Stelle wird blau. Vom Kachelrand: Wasser kommt nur von außen "
                      "herein (abgeschlossene Mulden bleiben trocken). Ab Startpunkt: Wasser breitet sich "
                      "nur von einem Punkt aus (z. B. Flussbett).")
        c2.selectbox("Hintergrund", names, key="flood_base")
        c3.slider("Deckkraft Wasser", 0.1, 1.0, step=0.05, key="water_opacity")
        c4.checkbox("Uferlinie", key="show_shore")
        if flood_mode == FLOOD_MODES[2]:
            s1, s2 = st.columns(2)
            s1.number_input("Startpunkt X", key="seed_x", format="%.1f")
            s2.number_input("Startpunkt Y", key="seed_y", format="%.1f")
            st.caption("💡 Koordinaten im Tab 2D → „Interaktiv“ beim Überfahren ablesen.")

        n_wet = int(wet.sum())
        area_ha = n_wet * grid.res**2 / 1e4
        vol = float(depth.sum(dtype=np.float64)) * grid.res**2
        mc = st.columns(5)
        mc[0].metric("Wasserstand", f"{S.water_level:.2f} m", f"{S.water_level - zmin:.2f} m über Tiefpunkt",
                     delta_color="off")
        mc[1].metric("Überflutete Fläche", f"{area_ha:,.1f} ha".replace(",", "."),
                     f"{n_wet / wet.size * 100:.1f} % der Kachel", delta_color="off")
        mc[2].metric("Wasservolumen", fmt_vol(vol))
        mc[3].metric("Max. Tiefe", f"{float(depth.max()):.2f} m")
        mc[4].metric("Mittlere Tiefe", f"{float(depth[wet].mean()) if n_wet else 0:.2f} m")
        if n_wet == 0 and flood_mode == FLOOD_MODES[2]:
            st.info("Der Startpunkt liegt über dem Wasserstand – nichts geflutet. Wasserstand erhöhen "
                    "oder Startpunkt in eine Senke legen.")

        lines = shore_lines(flood_key, wet, grid)
        show_plot(flood_figure(f"Wasserstand {S.water_level:.2f} m", S.flood_base, wet, depth,
                               lines, S.show_shore, S.view_height))
        st.caption("🖱️ Mausrad = Zoom · beim Überfahren: Höhe und Wassertiefe · Hellblau = flach, "
                   "Dunkelblau = tief. 💡 Hintergrund „Restrelief (LRM)“ + Uferlinie zeigt, ob die Uferlinie an "
                   "Geländekanten (mögliche Terrassen) entlangläuft. Wasser über die ganze Region: Tab „Karte“.")

        with st.expander("📈 Flutkurve – Fläche und Volumen je Wasserstand"):
            hyp = hypsometry(grid_key, grid.z, grid.res)
            fh = go.Figure()
            fh.add_trace(go.Scatter(x=hyp["wasserstand_m"], y=hyp["flaeche_ha"], name="Fläche (ha)",
                                    line=dict(color="royalblue")))
            fh.add_trace(go.Scatter(x=hyp["wasserstand_m"], y=hyp["volumen_m3"], name="Volumen (m³)",
                                    yaxis="y2", line=dict(color="darkorange", dash="dot")))
            fh.add_vline(x=float(S.water_level), line_dash="dash", line_color="gray")
            fh.update_layout(height=380, margin=dict(t=30, b=10), xaxis_title="Wasserstand (m)",
                             yaxis=dict(title="Fläche (ha)"),
                             yaxis2=dict(title="Volumen (m³)", overlaying="y", side="right"),
                             legend=dict(orientation="h", y=1.12))
            show_plot(fh)
            st.caption("Badewannen-Modell für die ganze Kachel. **Steile Stufen** in der Flächenkurve zeigen "
                       "Höhen, ab denen große ebene Flächen volllaufen – z. B. Talböden oder Terrassen.")

        with st.expander("🏖️ Strandlinien-Suche – auf welchen Höhen häufen sich Geländekanten?"):
            st.caption("Stand Wasser lange auf einer Höhe, entstehen oft Uferterrassen oder Brandungskanten – "
                       "alle auf **derselben Höhe**. Hier wird pro Höhenstufe gezählt, wie oft Hangknicke und "
                       "ebene Flächen vorkommen, verglichen mit dem Durchschnitt (1,0).")
            t1, t2 = st.columns([1, 3])
            run_terr = t1.checkbox("Analyse berechnen", key="terr_on")
            bin_m = t2.number_input("Höhenschritt (m)", min_value=0.1, max_value=50.0, value=1.0, step=0.5,
                                    key="terr_bin")
            if run_terr:
                tdf = terrace_signal((grid_key, float(bin_m)), grid.z, grid.res, float(bin_m))
                ft = go.Figure()
                ft.add_trace(go.Bar(x=tdf["hoehe_m"], y=tdf["kantendichte"], name="Hangknicke (relativ)",
                                    marker_color="darkorange"))
                ft.add_trace(go.Scatter(x=tdf["hoehe_m"], y=tdf["ebenen_anteil"], name="Ebene Flächen (relativ)",
                                        mode="lines", line=dict(color="seagreen")))
                ft.add_hline(y=1.0, line_dash="dot", line_color="gray")
                ft.add_vline(x=float(S.water_level), line_dash="dash", line_color="royalblue")
                ft.update_layout(height=380, margin=dict(t=30, b=10), xaxis_title="Höhe (m)",
                                 yaxis_title="× Durchschnitt", legend=dict(orientation="h", y=1.12), bargap=0)
                show_plot(ft)
                e = tdf["kantendichte"].to_numpy()
                peaks = [i for i in range(1, len(e) - 1)
                         if np.isfinite(e[i]) and e[i] >= 1.3 and e[i] >= np.nanmax(e[max(0, i - 2):i + 3])]
                top = sorted(peaks, key=lambda i: -e[i])[:6]
                if top:
                    st.write("**Auffällige Höhen** – Klick setzt den Wasserstand:")
                    pc_ = st.columns(len(top))
                    for j, i in enumerate(top):
                        h = float(tdf["hoehe_m"].iloc[i])
                        pc_[j].button(f"{h:.1f} m · ×{e[i]:.1f}", on_click=set_level, args=(h,),
                                      key=f"terr_btn_{j}", use_container_width=True)
                else:
                    st.info("Keine deutlich gehäuften Höhen gefunden (Schwelle ×1,3).")
                st.caption("⚠️ Nur ein Hinweis: Auch Ackerterrassen, Straßen, Bahndämme oder Gesteinsschichten "
                           "erzeugen Kanten auf gleicher Höhe. Prüfen mit Uferlinie + Restrelief, Profil und "
                           "Luftbild. Aussagekräftig wird es, wenn dieselbe Höhe in **mehreren Kacheln** auftaucht.")

        with st.expander("🎬 Animation – Wasser steigen lassen"):
            a1, a2, a3, a4 = st.columns(4)
            lo = a1.number_input("Von (m)", value=float(round(zmin, 2)), format="%.2f", key="anim_lo")
            hi = a2.number_input("Bis (m)", value=float(round(zmin + 0.5 * rng, 2)), format="%.2f",
                                 key="anim_hi")
            steps = a3.slider("Bilder", 6, 30, 15, key="anim_steps")
            speed = a4.select_slider("Tempo", ["langsam", "mittel", "schnell"], value="mittel",
                                     key="anim_speed")
            dur = {"langsam": 700, "mittel": 400, "schnell": 200}[speed]
            spec = (models_key, S.flood_base, round(lo, 3), round(hi, 3), steps, dur, flood_mode,
                    seed_rc if flood_mode == FLOOD_MODES[2] else None, float(S.water_opacity),
                    tuple(S.stretch))
            if st.button("🎬 Animation erzeugen"):
                if hi <= lo:
                    st.warning("„Bis“ muss größer sein als „Von“.")
                else:
                    with st.spinner("Erzeuge Animation …"):
                        frames = flood_frames(S.flood_base, lo, hi, steps, flood_mode, seed_rc,
                                              float(S.water_opacity))
                        S["_flood_anim"] = (spec, flood_anim_figure(frames, dur), frames)
            anim = S.get("_flood_anim")
            if anim and anim[0] == spec:
                show_plot(anim[1])
                st.caption("▶ Abspielen / ⏸ Pause und der Schieberegler befinden sich **unter dem Bild**. "
                           "Mit dem Schieberegler kannst du auch Bild für Bild durchgehen.")
                if PIL_OK:
                    lazy_download("GIF für Präsentationen erzeugen", spec,
                                  lambda: frames_to_gif(anim[2], dur), "flutung.gif", "image/gif")

        with st.expander("🌐 3D-Ansicht mit Wasser"):
            if st.checkbox("3D-Ansicht berechnen", key="flood_3d_on"):
                s3 = max(1, math.ceil(math.sqrt(grid.w * grid.h / 200_000)))
                zp3 = grid.z[::s3, ::s3]
                tp3 = layers[S.flood_base].raw[::s3, ::s3]
                wz = np.where(wet[::s3, ::s3], float(S.water_level), np.nan)
                xs3, ys3 = ds_axes(grid, zp3.shape, s3)
                v0, v1 = vrange(layers[S.flood_base], S.stretch)
                dx3, dy3 = grid.w * grid.res, grid.h * grid.res
                zr3 = float(zp3.max() - zp3.min()) or 1.0
                f3 = go.Figure([
                    go.Surface(x=xs3, y=ys3, z=zp3, surfacecolor=tp3, cmin=v0, cmax=v1, showscale=False,
                               colorscale=plotly_cs(layers[S.flood_base].cmap),
                               lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.9, specular=0.05),
                               hovertemplate="Höhe %{z:.2f} m<extra></extra>"),
                    go.Surface(x=xs3, y=ys3, z=wz, surfacecolor=np.zeros_like(wz), showscale=False,
                               colorscale=[[0, "#2b6cb0"], [1, "#2b6cb0"]], opacity=0.75,
                               lighting=dict(ambient=0.7, diffuse=0.3, specular=0.6, roughness=0.3),
                               hovertemplate="Wasserstand %{z:.2f} m<extra></extra>")])
                f3.update_layout(height=int(S.view_height), margin=dict(l=0, r=0, t=30, b=0),
                                 title=f"3D: Wasserstand {S.water_level:.2f} m",
                                 scene=dict(aspectmode="manual",
                                            aspectratio=dict(x=1, y=dy3 / dx3, z=max(zr3 / dx3 * S.z_exag, 0.02)),
                                            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Höhe (m)"))
                show_plot(f3)

        with st.expander("📦 Export Flutung"):
            e1, e2, e3 = st.columns(3)
            with e1:
                if RASTERIO_OK:
                    lazy_download("Wassertiefe als GeoTIFF erzeugen", flood_key,
                                  lambda: geotiff_bytes(np.where(wet, depth, np.nan), grid, crs),
                                  f"wassertiefe_{S.water_level:.1f}m.tif", "image/tiff")
                else:
                    st.caption("GeoTIFF benötigt 'rasterio'.")
            with e2:
                if lines:
                    props = {"wasserstand_m": round(float(S.water_level), 2), "modus": flood_mode}
                    if latlon[0] is not None:
                        st.download_button("⬇️ Uferlinie GeoJSON (WGS84)", lines_geojson(lines, crs, True, props),
                                           f"uferlinie_{S.water_level:.1f}m_wgs84.geojson", "application/geo+json")
                    st.download_button("⬇️ Uferlinie GeoJSON (Projekt-CRS)", lines_geojson(lines, crs, False, props),
                                       f"uferlinie_{S.water_level:.1f}m.geojson", "application/geo+json")
                else:
                    st.caption("Keine Uferlinie vorhanden (oder 'scikit-image' fehlt).")
            with e3:
                st.download_button("⬇️ Flutkurve (CSV)",
                                   hypsometry(grid_key, grid.z, grid.res).to_csv(index=False).encode("utf-8"),
                                   "flutkurve.csv", "text/csv")

        st.info("**Grenzen des Modells:** Das heutige Gelände ist nicht das damalige (Erosion, Ablagerung, "
                "Landwirtschaft, Bebauung). Höhenbezug prüfen (m ü. NHN vs. Ellipsoidhöhe – Unterschied in "
                "Deutschland rund 40–50 m). Eine einzelne Kachel zeigt keine regionale Flut. Ein blaues Bild ist "
                "kein Beleg – aussagekräftig sind nur Geländespuren (Terrassen, Strandwälle, Ablagerungen), die "
                "zum Wasserstand passen.")

    # ---------------- Export ----------------
    with texp:
        st.subheader("Raster (GeoTIFF)")
        if not RASTERIO_OK:
            st.warning("GeoTIFF-Export benötigt 'rasterio'.")
        else:
            c1, c2 = st.columns(2)
            ename = c1.selectbox("Layer", names, key="exp_layer")
            emode = c2.radio("Werte", ["Rohwerte (physikalische Einheit)", "Visualisierung (0–1)"], key="exp_mode")
            EL = layers[ename]

            def make_tif():
                arr = EL.raw
                if emode.startswith("Vis"):
                    lo_, hi_ = vrange(EL, S.stretch)
                    arr = np.clip((arr - lo_) / (hi_ - lo_), 0, 1)
                return geotiff_bytes(arr, grid, crs)

            lazy_download("GeoTIFF erzeugen", (models_key, ename, emode, tuple(S.stretch)), make_tif,
                          re.sub(r"[^\w-]+", "_", ename).strip("_") + ".tif", "image/tiff")

        st.subheader("Erkannte Features")
        if not feats:
            st.info("Zuerst im Tab '🔍 Features' die Erkennung starten.")
        else:
            c1, c2, c3 = st.columns(3)
            if latlon[0] is not None:
                c1.download_button("⬇️ GeoJSON (WGS84)", features_geojson(feats, grid, crs, True),
                                   "features_wgs84.geojson", "application/geo+json")
            c2.download_button("⬇️ GeoJSON (Projekt-CRS)", features_geojson(feats, grid, crs, False),
                               "features.geojson", "application/geo+json")
            c3.download_button("⬇️ Tabelle (CSV)", features_df(feats, crs).to_csv(index=False).encode("utf-8"),
                               "features.csv", "text/csv")

        st.subheader("Bericht & Projekt")
        rep = st.multiselect("Modelle im Bericht", names,
                             default=[n for n in ("VAT-ähnliche Fusion", "Restrelief (LRM)", "Hillshade",
                                                  "Hangneigung (Slope)") if n in names])
        lazy_download("HTML-Bericht erzeugen", (feat_key, tuple(rep), tuple(S.stretch)),
                      lambda: build_report(rep, features_df(feats, crs)), "lidar_bericht.html", "text/html")
        st.download_button("💾 Projekt speichern (.json)", json.dumps(
            {"app": f"LiDAR Archäologie Pro {APP_VERSION}", "dateien": [f.name for f in files], "crs": crs,
             "parameter": {k: S[k] for k in DEFAULTS}}, ensure_ascii=False, indent=2).encode("utf-8"),
            "lidar_projekt.json", "application/json")

except Exception as e:
    st.error(f"Fehler bei der Analyse: {e}")
    with st.expander("Technische Details"):
        st.exception(e)

