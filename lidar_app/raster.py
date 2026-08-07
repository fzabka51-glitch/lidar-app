"""
Archäologische Terrain-Visualisierungsmodelle.

Jede Funktion nimmt ein NumPy-Höhenraster und gibt ein normalisiertes
NumPy-Array zurück (Rückgabe immer NumPy, damit Matplotlib/Plotly sie
direkt konsumieren können – GPU-Beschleunigung passiert intern und
transparent über `backend.get_backend`).
"""
from __future__ import annotations

import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter, laplace

from .backend import get_backend, to_numpy


def _normalize(arr, low_pct: float = 2, high_pct: float = 98):
    p_low, p_high = np.percentile(arr, (low_pct, high_pct))
    clipped = np.clip(arr, p_low, p_high)
    lo, hi = clipped.min(), clipped.max()
    if hi > lo:
        return (clipped - lo) / (hi - lo)
    return np.full_like(arr, 0.5)


@st.cache_data(show_spinner=False, max_entries=16)
def calculate_hillshade(
    data: np.ndarray, azimuth: float = 315, angle_altitude: float = 45,
    res: float = 1.0, use_gpu: bool = True,
) -> np.ndarray:
    xp, is_gpu = get_backend(use_gpu)
    d = xp.asarray(data)

    azimuth_rad = xp.deg2rad(azimuth)
    altitude_rad = xp.deg2rad(angle_altitude)
    gy, gx = xp.gradient(d, res, res)
    slope = xp.arctan(xp.sqrt(gx ** 2 + gy ** 2))
    aspect = xp.arctan2(-gy, gx)
    shade = (xp.cos(altitude_rad) * xp.cos(slope)) + (
        xp.sin(altitude_rad) * xp.sin(slope) * xp.cos(azimuth_rad - aspect)
    )
    result = (shade + 1) / 2
    return to_numpy(result).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=8)
def calculate_multi_hillshade(data: np.ndarray, res: float = 1.0, use_gpu: bool = True) -> np.ndarray:
    """Multi-Directional Shading (MDS) aus 4 Richtungen – reduziert Richtungs-Bias einzelner Hillshades."""
    shades = [
        calculate_hillshade(data, az, 45, res, use_gpu)
        for az in (315, 45, 135, 225)
    ]
    return np.mean(shades, axis=0).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=8)
def calculate_lrm(data: np.ndarray, sigma: float = 15) -> np.ndarray:
    """Local Relief Model: hebt kleinräumige Strukturen (Wälle, Gräben) gegenüber dem Großrelief hervor."""
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    return _normalize(residual, 5, 95).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=8)
def calculate_slope(data: np.ndarray, res: float = 1.0, use_gpu: bool = True) -> np.ndarray:
    xp, is_gpu = get_backend(use_gpu)
    d = xp.asarray(data)
    gy, gx = xp.gradient(d, res, res)
    slope_deg = xp.rad2deg(xp.arctan(xp.sqrt(gx ** 2 + gy ** 2)))
    slope_deg = to_numpy(slope_deg)
    p_high = np.nanpercentile(slope_deg, 98)
    return np.clip(slope_deg, 0, p_high).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=8)
def calculate_curvature(data: np.ndarray, pre_smooth_sigma: float = 1.0) -> np.ndarray:
    """
    Lokale Krümmung via Laplace.

    Anders als im Original wird VOR der Laplace-Operation leicht geglättet:
    Laplace reagiert extrem empfindlich auf Messrauschen, ungeglättet
    dominiert das Rauschen das Ergebnis und archäologische Strukturen
    gehen unter.
    """
    smoothed = gaussian_filter(data, sigma=pre_smooth_sigma) if pre_smooth_sigma > 0 else data
    curv = -laplace(smoothed)
    return _normalize(curv, 2, 98).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=8)
def calculate_composite(mds: np.ndarray, lrm: np.ndarray, lrm_weight: float = 0.3) -> np.ndarray:
    """Fusion aus MDS-Hillshade und LRM – die eigentliche 'Prospektions'-Ansicht."""
    return np.clip(mds + (lrm - 0.5) * lrm_weight, 0, 1).astype(np.float32)
