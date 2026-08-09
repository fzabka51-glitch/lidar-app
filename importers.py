"""
importers.py
============
Zentrales Modul für den Datenimport der LiDAR Archäologie Pro App.

Unterstützt neben dem ursprünglichen .xyz/.txt-Format zusätzlich:
    - .las / .laz  (via laspy)
    - .ply         (via plyfile)
    - .csv         (generisches x,y,z)
    - .asc         (ESRI ASCII Grid)
    - .tif / .tiff / .geotiff (via rasterio)

Alle Punktwolken-Loader liefern ein pandas.DataFrame mit den Spalten
['x', 'y', 'z'] (float32) zurück, optional ergänzt um 'classification'
und 'intensity', falls in der Quelldatei vorhanden.

Raster-Loader liefern ein Dictionary mit:
    {
        "array": np.ndarray (2D, float32),
        "transform": affine.Affine,
        "crs": rasterio.crs.CRS | None,
        "res": float,          # mittlere Pixelauflösung in m
        "bounds": (minx, miny, maxx, maxy)
    }

Design-Prinzip: Diese Datei ist rein additiv und verändert keine
bestehende Logik in app.py. Der bisherige .xyz-Workflow bleibt davon
unberührt - `load_xyz_txt` bildet exakt das ursprüngliche Verhalten ab.
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


MAX_POINTS = 2_000_000  # konsistent mit dem bisherigen Limit in app.py


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


def load_las_laz(file: Union[str, BinaryIO], max_points: int = MAX_POINTS,
                  chunk_size: int = 5_000_000) -> pd.DataFrame:
    """Lädt eine .las oder .laz Datei mittels laspy (chunked / speicherschonend).

    Für sehr große Dateien wird per Chunked-Reader gelesen und bei Bedarf
    zufällig auf `max_points` heruntergesampelt, damit der Speicher der
    Streamlit-App nicht überläuft.

    Args:
        file: Pfad oder file-like Objekt einer .las/.laz Datei.
        max_points: Obergrenze für die Anzahl geladener Punkte.
        chunk_size: Anzahl Punkte pro Lese-Chunk (Streaming).

    Returns:
        DataFrame mit Spalten x, y, z sowie optional classification, intensity.

    Raises:
        ImportError: falls laspy nicht installiert ist.
    """
    if not LASPY_AVAILABLE:
        raise ImportError("Das Paket 'laspy' wird für .las/.laz Dateien benötigt "
                           "(pip install laspy[lazrs]).")

    xs, ys, zs, classes, intens = [], [], [], [], []
    total = 0

    with laspy.open(file) as reader:
        header_count = reader.header.point_count
        # Reservoir-artiges Downsampling beim Streamen großer Dateien
        keep_ratio = min(1.0, max_points / max(header_count, 1))

        for points in reader.chunk_iterator(chunk_size):
            n = len(points.x)
            if keep_ratio < 1.0:
                mask = np.random.rand(n) < keep_ratio
            else:
                mask = np.ones(n, dtype=bool)

            xs.append(np.asarray(points.x)[mask])
            ys.append(np.asarray(points.y)[mask])
            zs.append(np.asarray(points.z)[mask])

            if hasattr(points, "classification"):
                classes.append(np.asarray(points.classification)[mask])
            if hasattr(points, "intensity"):
                intens.append(np.asarray(points.intensity)[mask])

            total += mask.sum()

    df = pd.DataFrame({
        "x": np.concatenate(xs).astype(np.float32),
        "y": np.concatenate(ys).astype(np.float32),
        "z": np.concatenate(zs).astype(np.float32),
    })
    if classes:
        df["classification"] = np.concatenate(classes)
    if intens:
        df["intensity"] = np.concatenate(intens)

    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    return df.reset_index(drop=True)


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


MAX_RASTER_PIXELS = 16_000_000  # ca. 4000x4000 - schützt vor Speicherabsturz bei großen DGM-Kacheln


def load_geotiff(file: Union[str, BinaryIO], max_pixels: int = MAX_RASTER_PIXELS) -> dict:
    """Lädt ein GeoTIFF / TIF Höhenraster via rasterio.

    Sehr große Kacheln (z. B. DGM25/DGM1-Exporte mit 25 cm - 1 m Auflösung
    über eine größere Fläche) werden automatisch per Decimated Read
    (Resampling während des Einlesens, nicht danach) auf `max_pixels`
    heruntergerechnet. Das verhindert, dass das komplette hochauflösende
    Array unnötig in den Arbeitsspeicher geladen wird und die App
    (insbesondere in speicherbegrenzten Cloud-Umgebungen) abstürzt.

    Args:
        file: Pfad oder file-like Objekt einer .tif/.tiff Datei.
        max_pixels: Obergrenze für Breite*Höhe des geladenen Arrays.
            Bei Überschreitung wird automatisch downgesampelt.

    Returns:
        Dict mit array, transform, crs, res, bounds. Enthält zusätzlich
        "downsampled": bool und bei Downsampling "original_shape".

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
        orig_height, orig_width = src.height, src.width
        orig_pixels = orig_height * orig_width
        downsampled = orig_pixels > max_pixels

        if downsampled:
            scale = (max_pixels / orig_pixels) ** 0.5
            out_height = max(1, int(orig_height * scale))
            out_width = max(1, int(orig_width * scale))
            array = src.read(
                1,
                out_shape=(out_height, out_width),
                resampling=rasterio.enums.Resampling.average,
            ).astype(np.float32)
            # Transform an die neue (gröbere) Auflösung anpassen
            transform = src.transform * src.transform.scale(
                (orig_width / out_width), (orig_height / out_height)
            )
        else:
            array = src.read(1).astype(np.float32)
            transform = src.transform

        nodata = src.nodata
        if nodata is not None:
            array = np.where(array == nodata, np.nan, array)
        crs = src.crs
        res = abs(transform.a)
        bounds = src.bounds

    result = {
        "array": array,
        "transform": transform,
        "crs": crs,
        "res": res,
        "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
        "downsampled": downsampled,
    }
    if downsampled:
        result["original_shape"] = (orig_height, orig_width)
    return result


# --- Dispatcher ------------------------------------------------------------

POINT_CLOUD_EXTENSIONS = {".xyz", ".txt", ".csv", ".las", ".laz", ".ply"}
RASTER_EXTENSIONS = {".tif", ".tiff", ".geotiff", ".asc"}


def get_extension(filename: str) -> str:
    """Liefert die Dateiendung in Kleinbuchstaben inkl. Punkt."""
    return os.path.splitext(filename)[1].lower()


def load_point_cloud(file: Union[str, BinaryIO], filename: str,
                      max_points: int = MAX_POINTS) -> pd.DataFrame:
    """Dispatcher: lädt eine Punktwolken-Datei anhand ihrer Endung.

    Args:
        file: Pfad oder file-like Objekt.
        filename: Originaldateiname (für die Endungserkennung, z. B. von
            st.file_uploader.name).
        max_points: Obergrenze für die Anzahl geladener Punkte.

    Returns:
        DataFrame mit mindestens den Spalten x, y, z.

    Raises:
        ValueError: bei nicht unterstützter Dateiendung.
    """
    ext = get_extension(filename)
    if ext in (".xyz", ".txt", ".csv"):
        return load_xyz_txt(file, max_points)
    elif ext == ".las" or ext == ".laz":
        return load_las_laz(file, max_points)
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
