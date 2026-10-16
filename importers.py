

"""
importers.py
============
Zentrales Modul für den Datenimport der LiDAR Archäologie Pro App.

Unterstützt neben dem ursprünglichen .xyz/.txt-Format zusätzlich:
    - .las / .laz  (via laspy, inkl. IGN LiDAR HD .copc.laz)
    - .ply         (via plyfile)
    - .csv         (generisches x,y,z)
    - .asc         (ESRI ASCII Grid)
    - .tif / .tiff / .geotiff (via rasterio)

Punktwolken-Loader liefern ein pandas.DataFrame mit den Spalten
['x', 'y', 'z'].
    - .xyz/.txt/.csv/.ply: float32 (wie bisher)
    - .las/.laz: x/y als float64 (wichtig bei Lambert-93, sonst Rundungsfehler),
      z als float32. Die Punkte sind bereits zu Zellmittelwerten aggregiert.
      Das erkannte EPSG steht in df.attrs["epsg"] (oder None).

Raster-Loader liefern ein Dictionary mit:
    {
        "array": np.ndarray (2D, float32),
        "transform": affine.Affine,
        "crs": rasterio.crs.CRS | None,
        "res": float,          # mittlere Pixelauflösung in m
        "bounds": (minx, miny, maxx, maxy)
    }
"""

from __future__ import annotations

import io
import os
from typing import Optional, Union, BinaryIO

import numpy as np
import pandas as pd

# --- Optionale Abhängigkeiten sauber kapseln ---------------------------

try:
    import laspy
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False

try:
    from plyfile import PlyData
    PLYFILE_AVAILABLE = True
except ImportError:
    PLYFILE_AVAILABLE = False

try:
    import rasterio
    from rasterio.io import MemoryFile
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


MAX_POINTS = 2_000_000       # Limit für xyz/txt/csv/ply (wie bisher)
MAX_CELLS = 20_000_000       # Obergrenze Zellen bei LAS/LAZ (ca. 240 MB RAM)
GROUND_CLASS = 2             # ASPRS-Klasse "Ground"


# --- Punktwolken-Loader --------------------------------------------------

def load_xyz_txt(file: Union[str, BinaryIO], max_points: int = MAX_POINTS) -> pd.DataFrame:
    """Lädt eine .xyz/.txt/.csv Datei (ursprüngliches Verhalten der App).

    Args:
        file: Pfad oder file-like Objekt (z. B. st.file_uploader Ergebnis).
        max_points: Obergrenze für die Anzahl geladener Punkte.

    Returns:
        DataFrame mit Spalten x, y, z (float32).
    """
    df = pd.read_csv(file, sep=None, engine="python", header=None,
                      names=["x", "y", "z"], dtype=np.float32)
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    return df


def _extract_epsg(header) -> Optional[int]:
    """Liest den horizontalen EPSG-Code aus dem LAS/LAZ-Header (oder None)."""
    try:
        crs = header.parse_crs()          # braucht pyproj
    except Exception:
        return None
    if crs is None:
        return None
    try:
        # IGN: Compound-CRS (Lambert-93 + IGN69) -> horizontalen Teil nehmen
        if crs.is_compound and crs.sub_crs_list:
            crs = crs.sub_crs_list[0]
        epsg = crs.to_epsg()
        return int(epsg) if epsg else None
    except Exception:
        return None


def load_las_laz(file: Union[str, BinaryIO], max_points: int = MAX_POINTS,
                  chunk_size: int = 2_000_000, ground_only: bool = True,
                  cell_size: float = 0.5) -> pd.DataFrame:
    """Lädt .las/.laz/.copc.laz blockweise, speicherschonend.

    Jeder Block wird direkt in ein Zellraster aggregiert (Mittelwert pro Zelle),
    es liegt also nie die komplette Punktwolke im RAM. Koordinaten bleiben
    float64, damit Lambert-93 (Y ~ 6.8 Mio. m) nicht gerundet wird.

    Args:
        file: Pfad oder file-like Objekt einer .las/.laz Datei.
        max_points: Nur für Signatur-Kompatibilität, bei LAS/LAZ ersetzt die
            Zell-Aggregation das Zufalls-Sampling.
        chunk_size: Anzahl Punkte pro Lese-Chunk.
        ground_only: nur Klasse 2 (Boden) verwenden.
        cell_size: Zellgrösse der Aggregation in Metern.

    Returns:
        DataFrame x, y, z (Zellmittelwerte). df.attrs["epsg"] enthält das
        erkannte EPSG oder None.
    """
    if not LASPY_AVAILABLE:
        raise ImportError("Das Paket 'laspy' wird für .las/.laz Dateien benötigt "
                           "(pip install 'laspy[lazrs]').")

    n_seen = 0
    n_used = 0

    try:
        with laspy.open(file) as reader:
            hdr = reader.header
            x0, y0 = float(hdr.mins[0]), float(hdr.mins[1])
            x1, y1 = float(hdr.maxs[0]), float(hdr.maxs[1])

            nx = int(np.ceil((x1 - x0) / cell_size)) + 1
            ny = int(np.ceil((y1 - y0) / cell_size)) + 1
            if nx * ny > MAX_CELLS:
                raise ValueError(
                    f"Raster zu gross ({nx}x{ny} Zellen). Bitte in der Sidebar eine "
                    f"gröbere Auflösung wählen oder die Datei vorher zuschneiden."
                )

            sum_z = np.zeros(nx * ny, dtype=np.float64)
            cnt = np.zeros(nx * ny, dtype=np.int32)

            for pts in reader.chunk_iterator(chunk_size):
                n_seen += len(pts)
                mask = np.ones(len(pts), dtype=bool)
                if ground_only:
                    mask &= (np.asarray(pts.classification) == GROUND_CLASS)
                if not mask.any():
                    continue

                x = np.asarray(pts.x)[mask]
                y = np.asarray(pts.y)[mask]
                z = np.asarray(pts.z)[mask]

                ix = np.clip(((x - x0) / cell_size).astype(np.int64), 0, nx - 1)
                iy = np.clip(((y - y0) / cell_size).astype(np.int64), 0, ny - 1)
                idx = iy * nx + ix

                # Speicher proportional zum Chunk, nicht zur Rastergrösse
                uniq, inv = np.unique(idx, return_inverse=True)
                inv = inv.ravel()
                sum_z[uniq] += np.bincount(inv, weights=z, minlength=len(uniq))
                cnt[uniq] += np.bincount(inv, minlength=len(uniq)).astype(np.int32)
                n_used += len(z)

            epsg = _extract_epsg(hdr)

    except ValueError:
        raise
    except Exception as e:
        if "backend" in str(e).lower():
            raise RuntimeError(
                "Kein LAZ-Backend gefunden. Bitte 'laspy[lazrs]>=2.4' in die "
                "requirements.txt eintragen und die App neu starten."
            ) from e
        raise

    if n_used == 0:
        if ground_only and n_seen > 0:
            raise ValueError(
                f"Keine Bodenpunkte (Klasse {GROUND_CLASS}) gefunden ({n_seen:,} Punkte "
                f"gelesen). Ist die Datei unklassifiziert? Dann in der Sidebar "
                f"'Nur Bodenpunkte' ausschalten."
            )
        raise ValueError("Die Datei enthält keine Punkte.")

    cells = np.nonzero(cnt)[0]
    df = pd.DataFrame({
        "x": x0 + (cells % nx + 0.5) * cell_size,
        "y": y0 + (cells // nx + 0.5) * cell_size,
        "z": (sum_z[cells] / cnt[cells]).astype(np.float32),
    })
    df.attrs["epsg"] = epsg
    return df


def load_ply(file: Union[str, BinaryIO], max_points: int = MAX_POINTS) -> pd.DataFrame:
    """Lädt eine .ply Punktwolke (ASCII oder binär) via plyfile.

    Args:
        file: Pfad oder file-like Objekt einer .ply Datei.
        max_points: Obergrenze für die Anzahl geladener Punkte.

    Returns:
        DataFrame mit Spalten x, y, z (float32).

    Raises:
        ImportError: falls plyfile nicht installiert ist.
    """
    if not PLYFILE_AVAILABLE:
        raise ImportError("Das Paket 'plyfile' wird für .ply Dateien benötigt "
                           "(pip install plyfile).")

    ply = PlyData.read(file)
    vertex = ply["vertex"]
    df = pd.DataFrame({
        "x": np.asarray(vertex["x"], dtype=np.float32),
        "y": np.asarray(vertex["y"], dtype=np.float32),
        "z": np.asarray(vertex["z"], dtype=np.float32),
    })
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    return df


def load_asc(file: Union[str, BinaryIO]) -> dict:
    """Lädt ein ESRI ASCII Grid (.asc) als Raster (kein Punktwolken-Format).

    Args:
        file: Pfad oder file-like Objekt einer .asc Datei.

    Returns:
        Dict im selben Format wie `load_geotiff` (array, transform, crs, res, bounds).
    """
    if hasattr(file, "read"):
        text = file.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        lines = text.splitlines()
    else:
        with open(file, "r") as f:
            lines = f.read().splitlines()

    header = {}
    i = 0
    for i, line in enumerate(lines[:6]):
        key, val = line.split()
        header[key.lower()] = float(val)

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header.get("xllcorner", header.get("xllcenter", 0.0))
    yll = header.get("yllcorner", header.get("yllcenter", 0.0))
    cellsize = header["cellsize"]
    nodata = header.get("nodata_value", -9999.0)

    data = np.loadtxt(lines[6:], dtype=np.float32).reshape(nrows, ncols)
    data = np.where(data == nodata, np.nan, data)

    try:
        from affine import Affine
        transform = Affine(cellsize, 0, xll, 0, -cellsize, yll + nrows * cellsize)
    except ImportError:
        transform = None

    return {
        "array": data,
        "transform": transform,
        "crs": None,
        "res": cellsize,
        "bounds": (xll, yll, xll + ncols * cellsize, yll + nrows * cellsize),
    }


def load_geotiff(file: Union[str, BinaryIO]) -> dict:
    """Lädt ein GeoTIFF / TIF Höhenraster via rasterio.

    Args:
        file: Pfad oder file-like Objekt einer .tif/.tiff Datei.

    Returns:
        Dict mit array, transform, crs, res, bounds.

    Raises:
        ImportError: falls rasterio nicht installiert ist.
    """
    if not RASTERIO_AVAILABLE:
        raise ImportError("Das Paket 'rasterio' wird für GeoTIFF Dateien benötigt "
                           "(pip install rasterio).")

    if hasattr(file, "read"):
        raw = file.read()
        src_ctx = MemoryFile(raw).open()
    else:
        src_ctx = rasterio.open(file)

    with src_ctx as src:
        array = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            array = np.where(array == nodata, np.nan, array)
        transform = src.transform
        crs = src.crs
        res = abs(transform.a)
        bounds = src.bounds

    return {
        "array": array,
        "transform": transform,
        "crs": crs,
        "res": res,
        "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
    }


# --- Dispatcher ------------------------------------------------------------

POINT_CLOUD_EXTENSIONS = {".xyz", ".txt", ".csv", ".las", ".laz", ".ply"}
RASTER_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".asc"}


def get_extension(filename: str) -> str:
    """Liefert die Dateiendung in Kleinbuchstaben inkl. Punkt."""
    return os.path.splitext(filename)[1].lower()


def load_point_cloud(file: Union[str, BinaryIO], filename: str,
                      max_points: int = MAX_POINTS,
                      ground_only: bool = True,
                      cell_size: float = 0.5) -> pd.DataFrame:
    """Dispatcher: lädt eine Punktwolken-Datei anhand ihrer Endung.

    Args:
        file: Pfad oder file-like Objekt.
        filename: Originaldateiname (für die Endungserkennung, z. B. von
            st.file_uploader.name).
        max_points: Obergrenze für die Anzahl geladener Punkte (xyz/ply).
        ground_only: nur für LAS/LAZ: nur Bodenpunkte (Klasse 2) verwenden.
        cell_size: nur für LAS/LAZ: Zellgrösse der Aggregation in Metern.

    Returns:
        DataFrame mit mindestens den Spalten x, y, z.

    Raises:
        ValueError: bei nicht unterstützter Dateiendung.
    """
    ext = get_extension(filename)
    if ext in (".xyz", ".txt", ".csv"):
        return load_xyz_txt(file, max_points)
    elif ext == ".las" or ext == ".laz":
        return load_las_laz(file, max_points, ground_only=ground_only,
                             cell_size=cell_size)
    elif ext == ".ply":
        return load_ply(file, max_points)
    else:
        raise ValueError(f"Nicht unterstütztes Punktwolken-Format: {ext}")


def load_raster(file: Union[str, BinaryIO], filename: str) -> dict:
    """Dispatcher: lädt eine Raster-Datei (GeoTIFF/ASC) anhand ihrer Endung.

    Args:
        file: Pfad oder file-like Objekt.
        filename: Originaldateiname (für die Endungserkennung).

    Returns:
        Dict mit array, transform, crs, res, bounds.

    Raises:
        ValueError: bei nicht unterstützter Dateiendung.
    """
    ext = get_extension(filename)
    if ext in (".tif", ".tiff", ".geotiff"):
        return load_geotiff(file)
    elif ext == ".asc":
        return load_asc(file)
    else:
        raise ValueError(f"Nicht unterstütztes Raster-Format: {ext}")


def is_point_cloud(filename: str) -> bool:
    """True, wenn die Endung ein Punktwolken-Format ist."""
    return get_extension(filename) in POINT_CLOUD_EXTENSIONS


def is_raster(filename: str) -> bool:
    """True, wenn die Endung ein Raster-Format ist."""
    return get_extension(filename) in RASTER_EXTENSIONS
