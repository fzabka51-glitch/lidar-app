# =============================================================================
# app.py – LiDAR Archäologie Pro 2.2 (eine Datei, ersetzt importers/archaeology/exporters)
# Konvention: Raster intern "Nord oben" (Zeile 0 = Norden), alle Parameter in Metern.
# Neu in 2.1: größere Galerie, Einzelbild-Ansicht, Mausrad-Zoom, scharfe interaktive
#             Ansicht, einstellbare Bildhöhe und Bildauflösung.
# Neu in 2.2: Kontrast-Streckung für alle Modelle, kräftigere 3D-Schattierung mit
#             Lichtrichtung aus der Sidebar, Regler "Schattierung" im 3D-Tab.
# =============================================================================
from __future__ import annotations
import base64, html, io, json, math, os, re, warnings
from dataclasses import dataclass
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False
try:
    from plyfile import PlyData
    PLY_OK = True
except ImportError:
    PLY_OK = False

MAX_PIXELS = 6_000_000  # Speicherschutz (~2450 x 2450 px) – bei viel RAM erhöhen
POINT_EXT = (".xyz", ".txt", ".csv", ".las", ".laz", ".ply")
RASTER_EXT = (".tif", ".tiff", ".geotiff", ".asc")
ESRI_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
LABEL_COLORS = {"auffällig": "#e41a1c", "möglich": "#ff7f00", "unklar": "#377eb8"}
CIRCLE_T = np.linspace(0, 2 * np.pi, 37)
GEOM_KEYS = ["r0", "c0", "r1", "c1", "rad_px"]
DEFAULTS = dict(
    use_file_crs=True, epsg=25832, grid_res=1.0, agg_mode="Mittelwert", ground_only=True,
    morph_filter=False, morph_window_m=15.0, morph_threshold_m=1.5,
    hs_azimuth=315, hs_altitude=35, lrm_sigma_m=15.0,
    enable_svf=False, svf_dirs=8, svf_radius_m=15,
    stretch=(2.0, 98.0), hide_voids=True, z_exag=2.0, view_height=850, view_px=2000,
    shade_3d=0.7,
    det_lines=True, line_min_m=20.0, det_blobs=True, blob_d=(4.0, 30.0), blob_min_h=0.2,
    det_rings=False, ring_d=(10.0, 60.0),
)
S = st.session_state
for _k, _v in DEFAULTS.items():
    S.setdefault(_k, _v)


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
    # Graustufen-Modelle ohne feste Skala -> werden über den Kontrastregler gestreckt
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

    # 1) Hochaufgelöste Bilder (Basis + optional Überlagerung) passgenau in Weltkoordinaten
    extent = dict(xref="x", yref="y", x=grid.x_min, y=grid.y_max,
                  sizex=grid.x_max - grid.x_min, sizey=grid.y_max - grid.y_min,
                  sizing="stretch", layer="below")
    stack = [(L, 1.0)] + ([(overlay, opacity)] if overlay is not None else [])
    for layer, op in stack:
        png = layer_png((models_key, id(layer)), layer, grid, tuple(S.stretch), S.hide_voids, max_dim)
        fig.add_layout_image(source="data:image/png;base64," + base64.b64encode(png).decode(),
                             opacity=op, **extent)

    # 2) Unsichtbares, grobes Raster nur für die Messwert-Anzeige beim Überfahren
    hv, s = display_array(L, grid, 400, S.hide_voids)
    xs, ys = ds_axes(grid, hv.shape, s)
    u = f" {L.unit}" if L.unit else ""
    fig.add_trace(go.Heatmap(z=hv, x=xs, y=ys, opacity=0, showscale=False, showlegend=False,
                             hovertemplate=f"X %{{x:.1f}} · Y %{{y:.1f}}<br>≈ %{{z:.3f}}{u}<extra></extra>"))

    # 3) Farbskala (nur bei Modellen mit Einheit)
    if L.unit:
        vmin, vmax = vrange(L, S.stretch)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(colorscale=plotly_cs(L.cmap), cmin=vmin, cmax=vmax,
                                             color=[vmin], showscale=True, colorbar=dict(title=L.unit))))

    # 4) Erkannte Features
    for t in feature_traces(feats or [], grid):
        fig.add_trace(t)

    fig.update_layout(
        height=height, title=name, margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan", uirevision=str(hash(grid_key)), plot_bgcolor="white",
        xaxis=dict(title="X (m)", range=[grid.x_min, grid.x_max], constrain="domain",
                   showgrid=False, zeroline=False),
        yaxis=dict(title="Y (m)", range=[grid.y_min, grid.y_max], scaleanchor="x", scaleratio=1,
                   showgrid=False, zeroline=False),
        legend=dict(orientation="h", y=-0.08))
    return fig


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
        f"<div><h3>{html.escape(n)}</h3><img src='data:image/png;base64,"
        f"{base64.b64encode(layer_png((models_key, n), layers[n], grid, tuple(S.stretch), S.hide_voids, 1200)).decode()}'>"
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
    st.header("⚙️ Parameter")
    files = st.file_uploader("Dateien (Punktwolke(n) oder ein Raster)", accept_multiple_files=True,
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

    st.subheader("🌍 Geo-Referenz")
    st.checkbox("CRS aus Datei verwenden (falls vorhanden)", key="use_file_crs")
    st.number_input("EPSG-Code (Fallback)", min_value=1, max_value=999999, step=1, key="epsg",
                    help="25832 = UTM 32N · 25833 = UTM 33N · 2154 = Lambert-93 (IGN) · 2056 = CH LV95")
    if not PYPROJ_OK:
        st.warning("'pyproj' fehlt – Karte und Koordinaten deaktiviert.")

    st.subheader("☁️ Punktwolke → Raster")
    st.number_input("Auflösung (m)", min_value=0.1, max_value=10.0, step=0.1, key="grid_res")
    st.selectbox("Aggregation pro Zelle", ["Mittelwert", "Minimum (robuster bei Vegetation)"], key="agg_mode")
    st.checkbox("Nur Bodenpunkte (Klasse 2)", key="ground_only")
    st.checkbox("Morphologischer Vegetationsfilter", key="morph_filter",
                help="Entfernt Zellen, die deutlich über dem lokalen Minimum liegen (Bäume, Gebäude).")
    st.slider("Filterfenster (m)", 3.0, 50.0, step=1.0, key="morph_window_m", disabled=not S.morph_filter)
    st.slider("Filterschwelle (m)", 0.2, 5.0, step=0.1, key="morph_threshold_m", disabled=not S.morph_filter)

    st.subheader("🏔️ Reliefmodelle")
    st.slider("Licht-Azimut (°)", 0, 360, step=5, key="hs_azimuth",
              help="Gilt für Hillshade und für die Lichtrichtung in 3D.")
    st.slider("Sonnenhöhe (°)", 5, 80, key="hs_altitude", help="Flaches Licht (10–25°) betont feine Wälle.")
    st.slider("LRM-Glättung σ (m)", 2.0, 60.0, step=1.0, key="lrm_sigma_m")
    st.checkbox("SVF & Openness berechnen (rechenintensiv)", key="enable_svf")
    st.slider("Richtungen", 4, 16, key="svf_dirs", disabled=not S.enable_svf)
    st.slider("Suchradius (m)", 5, 50, key="svf_radius_m", disabled=not S.enable_svf)

    st.subheader("🎨 Anzeige")
    st.slider("Kontrast (Perzentile)", 0.0, 100.0, step=0.5, key="stretch",
              help="Enger (z. B. 5–95) = kräftigerer Kontrast. Gilt für 2D, 3D, Karte und Bericht.")
    st.checkbox("Große Datenlücken ausblenden", key="hide_voids")
    st.slider("Z-Überhöhung (3D)", 0.5, 10.0, step=0.5, key="z_exag")
    st.slider("Bildhöhe (px)", 400, 1600, step=50, key="view_height",
              help="Höhe der interaktiven Ansicht, der Karte und der 3D-Ansicht.")
    st.slider("Bildauflösung (px)", 800, 3000, step=100, key="view_px",
              help="Höher = schärfer beim Hineinzoomen, aber langsamer.")

    st.subheader("🔍 Feature-Erkennung")
    st.checkbox("Lineare Strukturen", key="det_lines")
    st.slider("Min. Linienlänge (m)", 5.0, 200.0, step=5.0, key="line_min_m", disabled=not S.det_lines)
    st.checkbox("Hügel & Senken", key="det_blobs")
    st.slider("Durchmesser (m)", 2.0, 80.0, step=1.0, key="blob_d", disabled=not S.det_blobs)
    st.slider("Min. Höhe/Tiefe (m)", 0.05, 2.0, step=0.05, key="blob_min_h", disabled=not S.det_blobs)
    st.checkbox("Ringstrukturen", key="det_rings")
    st.slider("Ring-Durchmesser (m)", 5.0, 150.0, step=5.0, key="ring_d", disabled=not S.det_rings)


# =============================================================================
# Hauptbereich
# =============================================================================
st.title("🏛️ LiDAR Archäologie Pro")
if not files:
    st.info("Bitte eine oder mehrere Punktwolken (.las, .laz, .xyz, .txt, .csv, .ply) "
            "oder ein Raster (.tif, .asc) hochladen.")
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


try:
    layers, extras = compute_models(models_key, grid.z, grid.res, S.hs_azimuth, S.hs_altitude,
                                    float(S.lrm_sigma_m), S.enable_svf, S.svf_dirs, S.svf_radius_m)
    names = list(layers)
    feats = current_features()
    t2d, t3d, tmap, tprof, tfeat, texp = st.tabs(
        ["🖼️ 2D", "🌐 3D", "🗺️ Karte", "📈 Profil", "🔍 Features", "📦 Export"])

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
        # Lichtrichtung aus Sidebar (Azimut/Sonnenhöhe), Stärke aus Regler "Schattierung"
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
            cx = [grid.x_min, grid.x_max, grid.x_max, grid.x_min]
            cy = [grid.y_max, grid.y_max, grid.y_min, grid.y_min]
            clat, clon = to_latlon(np.array(cx), np.array(cy), crs)
            uri = "data:image/png;base64," + base64.b64encode(
                layer_png((models_key, mlayer), layers[mlayer], grid, tuple(S.stretch),
                          S.hide_voids, int(S.view_px))).decode()
            mlayers = []
            if base.startswith("Satellit"):
                mlayers.append(dict(sourcetype="raster", source=[ESRI_URL], below="traces",
                                    sourceattribution="Esri World Imagery"))
            if custom.strip():
                mlayers.append(dict(sourcetype="raster", source=[custom.strip()], below="traces"))
            mlayers.append(dict(sourcetype="image", source=uri, opacity=op, below="traces",
                                coordinates=[[float(a), float(b)] for a, b in zip(clon, clat)]))
            use_new = hasattr(go, "Scattermap")
            Scatter = go.Scattermap if use_new else go.Scattermapbox
            fig_m = go.Figure(Scatter(lat=[latlon[0]], lon=[latlon[1]], mode="markers", name="Zentrum",
                                      marker=dict(size=8, color="black")))
            fdf_map = features_df(feats, crs)
            if len(fdf_map) and "Lat" in fdf_map:
                for label, color in LABEL_COLORS.items():
                    sub = fdf_map[fdf_map["Einstufung"] == label]
                    if len(sub):
                        fig_m.add_trace(Scatter(lat=sub["Lat"], lon=sub["Lon"], mode="markers", name=label,
                                                marker=dict(size=9, color=color),
                                                text=[f"#{i} {t} · {g} m" for i, t, g in
                                                      zip(sub["ID"], sub["Typ"], sub["Größe_m"])]))
            zoom = float(np.clip(math.log2(360 / max(np.ptp(clon), 1e-6)) + 1.5, 1, 19))
            fig_m.update_layout(**{"map" if use_new else "mapbox": dict(
                style="white-bg" if base != "OpenStreetMap" else "open-street-map",
                center=dict(lat=latlon[0], lon=latlon[1]), zoom=zoom, layers=mlayers)},
                height=int(S.view_height), margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h"))
            show_plot(fig_m)
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
            pf.add_trace(go.Scatter(x=dist, y=zs, mode="lines", name="Höhe"), row=1, col=1)
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
                    lo, hi = vrange(EL, S.stretch)
                    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
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
            {"app": "LiDAR Archäologie Pro 2.2", "dateien": [f.name for f in files], "crs": crs,
             "parameter": {k: S[k] for k in DEFAULTS}}, ensure_ascii=False, indent=2).encode("utf-8"),
            "lidar_projekt.json", "application/json")

except Exception as e:
    st.error(f"Fehler bei der Analyse: {e}")
    with st.expander("Technische Details"):
        st.exception(e)

