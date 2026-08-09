"""
exporters.py
============
GIS-Export-Funktionen für die LiDAR Archäologie Pro App.

Unterstützt:
    - GeoTIFF (Raster-Export, z. B. LRM, SVF, Slope) via rasterio
    - Shapefile (.shp) via geopandas/shapely
    - GeoJSON via geopandas/shapely

Alle Funktionen erwarten bereits georeferenzierte Eingaben (transform +
CRS) und schreiben nach dem angegebenen Pfad; sie geben zur weiteren
Verwendung in Streamlit (Download-Button) zusätzlich die Roh-Bytes zurück,
wo das sinnvoll ist.
"""

from __future__ import annotations

import io
import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine, from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


def _require_rasterio():
    if not RASTERIO_AVAILABLE:
        raise ImportError("Das Paket 'rasterio' wird für den GeoTIFF-Export benötigt "
                           "(pip install rasterio).")


def _require_geopandas():
    if not GEOPANDAS_AVAILABLE:
        raise ImportError("Die Pakete 'geopandas' und 'shapely' werden für den "
                           "Shapefile/GeoJSON-Export benötigt "
                           "(pip install geopandas shapely).")


def export_geotiff(array: np.ndarray, out_path: str, transform: "Affine",
                    crs: Optional[str] = None, nodata: Optional[float] = None) -> str:
    """Exportiert ein 2D-Raster als GeoTIFF.

    Args:
        array: 2D numpy Array (z. B. LRM, SVF, Slope, DTM).
        out_path: Zielpfad der .tif Datei.
        transform: Affine-Transformation (Pixel -> Weltkoordinaten), z. B.
            via `rasterio.transform.from_bounds` oder aus einem geladenen
            Raster übernommen.
        crs: Koordinatenreferenzsystem, z. B. "EPSG:25832". Optional.
        nodata: Nodata-Wert, der für NaN-Zellen geschrieben wird.

    Returns:
        Der geschriebene Pfad (out_path).
    """
    _require_rasterio()
    data = array.astype(np.float32)
    if nodata is not None:
        data = np.where(np.isnan(data), nodata, data)

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)

    return out_path


def build_transform_from_bounds(min_x: float, min_y: float, max_x: float, max_y: float,
                                 width: int, height: int) -> "Affine":
    """Erzeugt eine Affine-Transformation aus den Welt-Bounds und der Rastergröße.

    Praktisches Hilfsmittel, wenn ein Raster (z. B. `gz` aus der
    Datashader-Rasterisierung in app.py) noch keine explizite Transform hat.

    Args:
        min_x, min_y, max_x, max_y: Georäumliche Ausdehnung des Rasters.
        width: Breite des Rasters in Pixeln.
        height: Höhe des Rasters in Pixeln.

    Returns:
        rasterio.transform.Affine Objekt.
    """
    _require_rasterio()
    return from_bounds(min_x, min_y, max_x, max_y, width, height)


def export_linear_features_shapefile(features: Sequence, out_path: str, transform: "Affine",
                                      crs: Optional[str] = None,
                                      attributes: Optional[List[dict]] = None) -> str:
    """Exportiert erkannte lineare Features (archaeology.LinearFeature) als Shapefile.

    Args:
        features: Liste von LinearFeature-Objekten (aus archaeology.detect_linear_features).
        out_path: Zielpfad der .shp Datei.
        transform: Affine-Transformation zur Umrechnung Pixel -> Weltkoordinaten.
        crs: Koordinatenreferenzsystem, z. B. "EPSG:25832".
        attributes: Optionale Liste von Attribut-Dicts (gleiche Länge wie features),
            z. B. Klassifikationsergebnisse aus classify_feature_origin.

    Returns:
        Der geschriebene Pfad (out_path).
    """
    _require_geopandas()
    geoms = []
    for f in features:
        (r0, c0), (r1, c1) = f.start, f.end
        x0, y0 = transform * (c0, r0)
        x1, y1 = transform * (c1, r1)
        geoms.append(LineString([(x0, y0), (x1, y1)]))

    data = attributes if attributes is not None else [
        {"length_px": f.length_px, "orientation_deg": f.orientation_deg} for f in features
    ]
    gdf = gpd.GeoDataFrame(data, geometry=geoms, crs=crs)
    gdf.to_file(out_path)
    return out_path


def export_circular_features_shapefile(features: Sequence, out_path: str, transform: "Affine",
                                        crs: Optional[str] = None,
                                        attributes: Optional[List[dict]] = None) -> str:
    """Exportiert erkannte zirkuläre Features (archaeology.CircularFeature) als Shapefile.

    Die Kreise werden als Punktgeometrie (Zentrum) mit Radius-Attribut
    exportiert, alternativ als Polygon-Approximation über `as_polygon=True`
    ließe sich das leicht erweitern.

    Args:
        features: Liste von CircularFeature-Objekten (aus archaeology.detect_circular_features).
        out_path: Zielpfad der .shp Datei.
        transform: Affine-Transformation zur Umrechnung Pixel -> Weltkoordinaten.
        crs: Koordinatenreferenzsystem, z. B. "EPSG:25832".
        attributes: Optionale Liste von Attribut-Dicts (gleiche Länge wie features).

    Returns:
        Der geschriebene Pfad (out_path).
    """
    _require_geopandas()
    geoms = []
    radii_m = []
    for f in features:
        r, c = f.center
        x, y = transform * (c, r)
        geoms.append(Point(x, y))
        # Pixelradius grob in Meter umrechnen (Annahme: quadratische Pixel)
        radii_m.append(abs(f.radius_px * transform.a))

    data = attributes if attributes is not None else [
        {"radius_px": f.radius_px, "radius_m": rm, "strength": f.strength}
        for f, rm in zip(features, radii_m)
    ]
    gdf = gpd.GeoDataFrame(data, geometry=geoms, crs=crs)
    gdf.to_file(out_path)
    return out_path


def export_geojson(features: Sequence, out_path: str, transform: "Affine",
                    feature_type: str = "linear", crs: Optional[str] = None,
                    attributes: Optional[List[dict]] = None) -> str:
    """Exportiert lineare oder zirkuläre Features als GeoJSON.

    Args:
        features: Liste von LinearFeature- oder CircularFeature-Objekten.
        out_path: Zielpfad der .geojson Datei.
        transform: Affine-Transformation zur Umrechnung Pixel -> Weltkoordinaten.
        feature_type: "linear" oder "circular" - bestimmt die Geometrieerzeugung.
        crs: Koordinatenreferenzsystem, z. B. "EPSG:4326" (für GeoJSON empfohlen).
        attributes: Optionale Liste von Attribut-Dicts.

    Returns:
        Der geschriebene Pfad (out_path).
    """
    _require_geopandas()
    if feature_type == "linear":
        geoms = []
        for f in features:
            (r0, c0), (r1, c1) = f.start, f.end
            x0, y0 = transform * (c0, r0)
            x1, y1 = transform * (c1, r1)
            geoms.append(LineString([(x0, y0), (x1, y1)]))
        default_attrs = [{"length_px": f.length_px, "orientation_deg": f.orientation_deg}
                          for f in features]
    elif feature_type == "circular":
        geoms = []
        default_attrs = []
        for f in features:
            r, c = f.center
            x, y = transform * (c, r)
            geoms.append(Point(x, y))
            default_attrs.append({"radius_px": f.radius_px, "strength": f.strength})
    else:
        raise ValueError("feature_type muss 'linear' oder 'circular' sein.")

    data = attributes if attributes is not None else default_attrs
    gdf = gpd.GeoDataFrame(data, geometry=geoms, crs=crs)
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def geotiff_bytes(array: np.ndarray, transform: "Affine", crs: Optional[str] = None,
                   nodata: Optional[float] = None) -> bytes:
    """Erzeugt GeoTIFF-Rohbytes im Speicher (praktisch für st.download_button).

    Args:
        array: 2D numpy Array.
        transform: Affine-Transformation.
        crs: Koordinatenreferenzsystem.
        nodata: Nodata-Wert.

    Returns:
        Bytes-Objekt mit dem GeoTIFF-Inhalt.
    """
    _require_rasterio()
    data = array.astype(np.float32)
    if nodata is not None:
        data = np.where(np.isnan(data), nodata, data)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=data.shape[0], width=data.shape[1],
            count=1, dtype=data.dtype, crs=crs, transform=transform, nodata=nodata,
        ) as dst:
            dst.write(data, 1)
        return memfile.read()
