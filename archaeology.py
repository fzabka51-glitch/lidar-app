"""
archaeology.py
==============
Modulare Berechnungs-Pipelines für die archäologische Geländeanalyse.

Enthält:
    - Relief-Modelle: Hillshade, Multi-Directional-Shading, LRM, Slope,
      Aspect, Curvature (die ursprünglichen Funktionen aus app.py wurden
      hierher übernommen und unverändert in ihrem Verhalten belassen).
    - Erweiterte archäologische Visualisierungen: Sky-View-Factor (SVF),
      Positive/Negative Openness.
    - DTM/nDSM/CHM Ableitung aus klassifizierten oder unklassifizierten
      Punktwolken.
    - Automatische Feature-Erkennung: lineare Strukturen (Wälle, Gräben,
      Hohlwege) via Kantenerkennung + Hough-Transformation, zirkuläre
      Strukturen (Ringwälle, Grabhügel) via Hough-Circle-Transformation.
    - Eine einfache regelbasierte Klassifikation anthropogen/natürlich.

Alle Funktionen arbeiten auf 2D numpy-Arrays (gerasterte Höhenmodelle)
und sind unabhängig von Streamlit, damit sie sich auch in Notebooks oder
Batch-Skripten wiederverwenden lassen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import laplace, gaussian_filter, uniform_filter, grey_opening

try:
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line, hough_circle, hough_circle_peaks
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Basis-Reliefmodelle (unverändert aus app.py übernommen)
# ---------------------------------------------------------------------------

def calculate_hillshade(data: np.ndarray, azimuth: float = 315, angle_altitude: float = 45,
                         res: float = 1.0) -> np.ndarray:
    """Berechnet ein Schummerungsbild (Hillshade) aus einer bestimmten Richtung.

    Args:
        data: Höhenraster (2D).
        azimuth: Sonnenazimut in Grad (0-360).
        angle_altitude: Sonnenhöhe in Grad (0-90).
        res: Rasterauflösung in Metern.

    Returns:
        2D Array mit Werten in [0, 1].
    """
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(altitude_rad) * np.cos(slope)) + \
            (np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)


def calculate_multi_hillshade(data: np.ndarray, res: float = 1.0) -> np.ndarray:
    """Multi-Directional Shading (MDS) aus 4 Himmelsrichtungen.

    Args:
        data: Höhenraster (2D).
        res: Rasterauflösung in Metern.

    Returns:
        2D Array mit Werten in [0, 1].
    """
    h1 = calculate_hillshade(data, 315, 45, res)
    h2 = calculate_hillshade(data, 45, 45, res)
    h3 = calculate_hillshade(data, 135, 45, res)
    h4 = calculate_hillshade(data, 225, 45, res)
    return (h1 + h2 + h3 + h4) / 4.0


def calculate_lrm(data: np.ndarray, sigma: float = 15) -> np.ndarray:
    """Local Relief Model (LRM) / Residual Topography.

    Args:
        data: Höhenraster (2D).
        sigma: Glättungsradius für das Trendmodell (Gauß-Filter).

    Returns:
        Auf [0, 1] normalisiertes Residual-Raster.
    """
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p_low, p_high = np.percentile(residual, (5, 95))
    res_clipped = np.clip(residual, p_low, p_high)
    res_min, res_max = res_clipped.min(), res_clipped.max()
    if res_max > res_min:
        return (res_clipped - res_min) / (res_max - res_min)
    return np.full_like(residual, 0.5)


def calculate_slope(data: np.ndarray, res: float = 1.0) -> np.ndarray:
    """Berechnet die Hangneigung (Slope) in Grad.

    Args:
        data: Höhenraster (2D).
        res: Rasterauflösung in Metern.

    Returns:
        2D Array der Hangneigung in Grad, auf das 98. Perzentil geclippt.
    """
    gy, gx = np.gradient(data, res, res)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))
    p_high = np.nanpercentile(slope_deg, 98)
    return np.clip(slope_deg, 0, p_high)


def calculate_aspect(data: np.ndarray, res: float = 1.0) -> np.ndarray:
    """Berechnet die Hangausrichtung (Aspect) in Grad (0-360, 0 = Nord).

    Args:
        data: Höhenraster (2D).
        res: Rasterauflösung in Metern.

    Returns:
        2D Array der Hangausrichtung in Grad.
    """
    gy, gx = np.gradient(data, res, res)
    aspect_rad = np.arctan2(gy, -gx)
    aspect_deg = np.rad2deg(aspect_rad)
    aspect_deg = np.mod(90.0 - aspect_deg, 360.0)
    return aspect_deg.astype(np.float32)


def calculate_curvature(data: np.ndarray) -> np.ndarray:
    """Berechnet die lokale Krümmung (Laplace-Operator).

    Args:
        data: Höhenraster (2D).

    Returns:
        Auf [0, 1] normalisierte Krümmung.
    """
    curv = -laplace(data)
    p_low, p_high = np.percentile(curv, (2, 98))
    curv_clipped = np.clip(curv, p_low, p_high)
    c_min, c_max = curv_clipped.min(), curv_clipped.max()
    if c_max > c_min:
        return (curv_clipped - c_min) / (c_max - c_min)
    return np.full_like(curv, 0.5)


# ---------------------------------------------------------------------------
# 2. Erweiterte archäologische Visualisierungen: SVF & Openness
# ---------------------------------------------------------------------------

def _directional_horizon_angle(data: np.ndarray, direction_rad: float, res: float,
                                max_radius_px: int) -> np.ndarray:
    """Ermittelt für jede Zelle den maximalen Horizontwinkel entlang einer Richtung.

    Interne Hilfsfunktion für SVF/Openness. Nutzt eine diskrete Abtastung
    entlang der Blickrichtung über zunehmende Radien (klassischer Ansatz
    für Sky-View-Factor auf gerasterten DEMs).

    Args:
        data: Höhenraster (2D).
        direction_rad: Blickrichtung in Radiant.
        res: Rasterauflösung in Metern.
        max_radius_px: Maximaler Suchradius in Pixeln.

    Returns:
        2D Array mit dem maximalen Elevationswinkel (Radiant) je Zelle.
    """
    rows, cols = data.shape
    dy = np.sin(direction_rad)
    dx = np.cos(direction_rad)

    max_angle = np.zeros_like(data, dtype=np.float32)
    yy, xx = np.mgrid[0:rows, 0:cols]

    # Radien in geometrisch wachsenden Schritten abtasten (Performance)
    radii = np.unique(np.round(np.geomspace(1, max_radius_px, num=min(max_radius_px, 16))).astype(int))

    for r in radii:
        sy = np.clip(np.round(yy + dy * r).astype(int), 0, rows - 1)
        sx = np.clip(np.round(xx + dx * r).astype(int), 0, cols - 1)
        dz = data[sy, sx] - data
        dist = r * res
        angle = np.arctan2(dz, dist)
        max_angle = np.maximum(max_angle, angle)

    return max_angle


def calculate_sky_view_factor(data: np.ndarray, res: float = 1.0, n_directions: int = 8,
                               search_radius_m: float = 20.0) -> np.ndarray:
    """Berechnet den Sky-View-Factor (SVF) - Standard-Tool der LiDAR-Archäologie.

    Der SVF misst den Anteil des sichtbaren Himmels je Zelle und ist
    besonders gut geeignet, um flache, konkave Strukturen wie Gräben und
    Hohlwege sichtbar zu machen.

    Hinweis: Diese Implementierung ist eine performante Rasterapproximation
    (diskrete Richtungs- und Radienabtastung) und kein Ersatz für
    spezialisierte Tools wie RVT, liefert für Prospektionszwecke aber
    vergleichbare, gut interpretierbare Ergebnisse.

    Args:
        data: Höhenraster (2D).
        res: Rasterauflösung in Metern.
        n_directions: Anzahl der abgetasteten Himmelsrichtungen (8-16 üblich).
        search_radius_m: Suchradius in Metern.

    Returns:
        Auf [0, 1] normalisiertes SVF-Raster (1 = voller Himmel sichtbar).
    """
    max_radius_px = max(1, int(search_radius_m / res))
    directions = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    horizon_sum = np.zeros_like(data, dtype=np.float32)
    for d in directions:
        angle = _directional_horizon_angle(data, d, res, max_radius_px)
        horizon_sum += np.sin(np.clip(angle, 0, np.pi / 2))

    svf = 1.0 - (horizon_sum / n_directions)
    return np.clip(svf, 0, 1).astype(np.float32)


def calculate_openness(data: np.ndarray, res: float = 1.0, n_directions: int = 8,
                        search_radius_m: float = 20.0,
                        mode: Literal["positive", "negative"] = "positive") -> np.ndarray:
    """Berechnet positive oder negative Openness.

    Positive Openness betont erhabene Strukturen (Grabhügel, Wälle),
    negative Openness betont Vertiefungen (Gräben, Hohlwege) - beide sind
    Standardwerkzeuge der archäologischen LiDAR-Prospektion.

    Args:
        data: Höhenraster (2D).
        res: Rasterauflösung in Metern.
        n_directions: Anzahl der abgetasteten Himmelsrichtungen.
        search_radius_m: Suchradius in Metern.
        mode: "positive" (Blick nach oben) oder "negative" (Blick nach unten,
            invertiertes DEM).

    Returns:
        Auf [0, 1] normalisiertes Openness-Raster.
    """
    surface = data if mode == "positive" else -data
    max_radius_px = max(1, int(search_radius_m / res))
    directions = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

    zenith_sum = np.zeros_like(surface, dtype=np.float32)
    for d in directions:
        angle = _directional_horizon_angle(surface, d, res, max_radius_px)
        zenith_angle = (np.pi / 2) - np.clip(angle, 0, np.pi / 2)
        zenith_sum += zenith_angle

    openness = zenith_sum / n_directions
    o_min, o_max = openness.min(), openness.max()
    if o_max > o_min:
        return ((openness - o_min) / (o_max - o_min)).astype(np.float32)
    return np.full_like(openness, 0.5, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. DTM / nDSM / CHM Ableitung
# ---------------------------------------------------------------------------

def estimate_dtm(dsm: np.ndarray, opening_size: int = 11) -> np.ndarray:
    """Approximiert ein DTM (Bodenmodell) aus einem DSM ohne Klassifikation.

    Nutzt eine morphologische Öffnung (Grey-Opening) als einfachen, robusten
    Ground-Filter: lokale Erhebungen (Vegetation, Gebäude) werden entfernt,
    das grobe Geländerelief bleibt erhalten. Für Punktwolken mit vorhandener
    LAS-Klassifikation ist `dtm_from_classified_points` vorzuziehen.

    Args:
        dsm: Digitales Oberflächenmodell (2D Raster, z. B. aus allen Returns).
        opening_size: Strukturgröße (Pixel) des morphologischen Filters -
            sollte in etwa der größten zu entfernenden Objektgröße entsprechen.

    Returns:
        Approximiertes DTM (2D Raster).
    """
    return grey_opening(dsm, size=(opening_size, opening_size)).astype(np.float32)


def dtm_from_classified_points(rasterize_fn, df, res: float, ground_class: int = 2):
    """Erzeugt ein DTM aus klassifizierten LAS/LAZ-Punkten (ASPRS Klasse 2 = Boden).

    Args:
        rasterize_fn: Rasterisierungsfunktion mit Signatur (df, res) -> np.ndarray,
            z. B. die bestehende `rasterize_points` Funktion aus app.py.
        df: DataFrame mit Spalten x, y, z, classification.
        res: Zielauflösung in Metern.
        ground_class: ASPRS-Klassencode für Bodenpunkte (Standard: 2).

    Returns:
        DTM-Raster (2D).

    Raises:
        ValueError: falls keine 'classification' Spalte vorhanden ist.
    """
    if "classification" not in df.columns:
        raise ValueError("Für ein klassifiziertes DTM wird eine 'classification'-Spalte "
                          "benötigt (z. B. aus einer klassifizierten .las Datei).")
    ground_df = df[df["classification"] == ground_class]
    if len(ground_df) < 10:
        raise ValueError("Zu wenige Bodenpunkte (Klasse 2) für eine DTM-Berechnung gefunden.")
    return rasterize_fn(ground_df, res)


def calculate_ndsm(dsm: np.ndarray, dtm: np.ndarray) -> np.ndarray:
    """Berechnet das normalized Digital Surface Model (nDSM = DSM - DTM).

    Args:
        dsm: Digitales Oberflächenmodell.
        dtm: Digitales Geländemodell (Boden).

    Returns:
        nDSM (Höhe über Grund), negative Werte auf 0 geclippt.
    """
    ndsm = dsm - dtm
    return np.clip(ndsm, 0, None).astype(np.float32)


def calculate_chm(ndsm: np.ndarray, smoothing_sigma: float = 0.0) -> np.ndarray:
    """Berechnet das Canopy Height Model (CHM) aus dem nDSM.

    Args:
        ndsm: normalized Digital Surface Model.
        smoothing_sigma: optionale Glättung zur Rauschreduktion (0 = aus).

    Returns:
        CHM-Raster (2D).
    """
    if smoothing_sigma > 0:
        return gaussian_filter(ndsm, sigma=smoothing_sigma).astype(np.float32)
    return ndsm.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Automatische Feature-Erkennung
# ---------------------------------------------------------------------------

@dataclass
class LinearFeature:
    """Repräsentiert ein erkanntes lineares Feature (z. B. Wall, Graben)."""
    start: Tuple[int, int]   # (row, col) im Rasterindex
    end: Tuple[int, int]
    length_px: float
    orientation_deg: float


@dataclass
class CircularFeature:
    """Repräsentiert ein erkanntes zirkuläres Feature (z. B. Grabhügel, Ringwall)."""
    center: Tuple[int, int]  # (row, col) im Rasterindex
    radius_px: float
    strength: float           # Akkumulator-Score der Hough-Transformation


def detect_linear_features(relief_image: np.ndarray, min_length_px: int = 15,
                            line_gap_px: int = 3, canny_sigma: float = 1.5) -> List[LinearFeature]:
    """Erkennt lineare Strukturen (Wälle, Gräben, Hohlwege, Terrassenkanten).

    Pipeline: Kantenerkennung (Canny) auf dem Relief-Bild (idealerweise LRM
    oder Multi-Hillshade), gefolgt von einer probabilistischen
    Hough-Transformation zur Extraktion einzelner Liniensegmente.

    Args:
        relief_image: 2D Array, normalisiert auf [0, 1] (z. B. LRM oder MDS).
        min_length_px: Minimale Linienlänge in Pixeln, um Rauschen zu filtern.
        line_gap_px: Maximale Lücke zwischen Liniensegmenten, die noch als
            eine Linie gilt.
        canny_sigma: Glättungsparameter der Canny-Kantenerkennung.

    Returns:
        Liste erkannter LinearFeature-Objekte.

    Raises:
        ImportError: falls scikit-image nicht installiert ist.
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("Das Paket 'scikit-image' wird für die Feature-Erkennung "
                           "benötigt (pip install scikit-image).")

    img = np.nan_to_num(relief_image)
    edges = canny(img, sigma=canny_sigma)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=min_length_px,
                                      line_gap=line_gap_px)

    features = []
    for (x0, y0), (x1, y1) in lines:
        length = float(np.hypot(x1 - x0, y1 - y0))
        orientation = float(np.rad2deg(np.arctan2(y1 - y0, x1 - x0)) % 180)
        features.append(LinearFeature(start=(y0, x0), end=(y1, x1),
                                       length_px=length, orientation_deg=orientation))
    return features


def detect_circular_features(relief_image: np.ndarray, min_radius_px: int = 5,
                              max_radius_px: int = 40, radius_step: int = 2,
                              max_features: int = 25,
                              canny_sigma: float = 1.5) -> List[CircularFeature]:
    """Erkennt zirkuläre / symmetrische Strukturen (Ringwälle, Grabhügel, Pingo-Ruinen).

    Pipeline: Kantenerkennung (Canny), anschließend Hough-Circle-Transformation
    über einen Bereich plausibler Radien mit Peak-Erkennung.

    Args:
        relief_image: 2D Array, normalisiert auf [0, 1] (z. B. LRM oder SVF).
        min_radius_px: Minimaler Suchradius in Pixeln.
        max_radius_px: Maximaler Suchradius in Pixeln.
        radius_step: Schrittweite zwischen den getesteten Radien.
        max_features: Maximale Anzahl zurückgegebener Kreise (stärkste zuerst).
        canny_sigma: Glättungsparameter der Canny-Kantenerkennung.

    Returns:
        Liste erkannter CircularFeature-Objekte, absteigend nach Stärke sortiert.

    Raises:
        ImportError: falls scikit-image nicht installiert ist.
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("Das Paket 'scikit-image' wird für die Feature-Erkennung "
                           "benötigt (pip install scikit-image).")

    img = np.nan_to_num(relief_image)
    edges = canny(img, sigma=canny_sigma)

    radii = np.arange(min_radius_px, max_radius_px, radius_step)
    if len(radii) == 0:
        return []

    hough_res = hough_circle(edges, radii)
    accums, cx, cy, radii_out = hough_circle_peaks(hough_res, radii, total_num_peaks=max_features)

    features = []
    for accum, x, y, r in zip(accums, cx, cy, radii_out):
        features.append(CircularFeature(center=(int(y), int(x)), radius_px=float(r),
                                         strength=float(accum)))
    return sorted(features, key=lambda f: f.strength, reverse=True)


# ---------------------------------------------------------------------------
# 5. Regelbasierte Klassifikation: anthropogen vs. natürlich
# ---------------------------------------------------------------------------

def classify_feature_origin(feature, slope_at_location: Optional[float] = None,
                             regularity_threshold: float = 0.7) -> str:
    """Einfache regelbasierte Heuristik zur Unterscheidung anthropogen/natürlich.

    Dies ist bewusst ein transparentes, nachvollziehbares Regelwerk und
    kein ML-Klassifikator - für Prospektionszwecke dient es als schnelle
    Vorfilterung, die von Fachpersonal überprüft werden sollte.

    Regeln:
        - CircularFeature: nahezu perfekte Kreise mit moderatem Radius und
          hohem Hough-Score gelten als potentiell anthropogen (Grabhügel,
          Ringwall). Sehr große oder sehr schwache Kreise gelten als
          wahrscheinlich natürlich (z. B. Dolinen, Baumkronen).
        - LinearFeature: sehr gerade, längere Segmente mit typischen
          Vorzugsrichtungen (z. B. parallel zu anderen Linien) gelten als
          potentiell anthropogen (Wälle, Terrassenkanten); kurze, stark
          gekrümmte oder zufällig orientierte Segmente eher als natürlich
          (Erosionsrinnen, Baumwurf).

    Args:
        feature: LinearFeature oder CircularFeature Instanz.
        slope_at_location: Optionale mittlere Hangneigung (Grad) am Ort des
            Features - sehr steile Lagen sprechen eher für natürliche
            Erosionsformen.
        regularity_threshold: Schwellenwert (0-1) für die geforderte
            "Regelmäßigkeit", ab der ein Feature als potentiell anthropogen
            eingestuft wird.

    Returns:
        Einer von "wahrscheinlich_anthropogen", "wahrscheinlich_natuerlich"
        oder "unklar".
    """
    if isinstance(feature, CircularFeature):
        # Normalisierter Score: höhere Werte = klarerer, regelmäßigerer Kreis
        regularity = min(1.0, feature.strength / (2 * np.pi * max(feature.radius_px, 1)))
        if regularity >= regularity_threshold and 3 <= feature.radius_px <= 60:
            if slope_at_location is not None and slope_at_location > 25:
                return "unklar"  # könnte auch ein natürlicher Krater/Doline im Steilhang sein
            return "wahrscheinlich_anthropogen"
        return "wahrscheinlich_natuerlich"

    if isinstance(feature, LinearFeature):
        if feature.length_px >= 20:
            # Gerade, längere Segmente sind typisch für Wälle/Terrassenkanten/Wege
            if slope_at_location is not None and slope_at_location > 35:
                return "unklar"  # in Steillagen oft Erosionsrinnen
            return "wahrscheinlich_anthropogen"
        return "unklar"

    return "unklar"
