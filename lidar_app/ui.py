"""
Rasterisierung: Punktwolke -> regelmäßiges Höhenraster (DTM-Grid).

Wichtige Korrekturen gegenüber der ursprünglichen Version:
  * Canvas-Größe wird gedeckelt (LIMITS.MAX_GRID_CELLS), damit eine zu
    fein gewählte Auflösung bei großer Fläche nicht den Prozess killt.
  * Lücken (Zellen ohne Punkt) werden per Nearest-Neighbour-Interpolation
    gefüllt statt mit dem globalen Mittelwert – letzterer verwischt genau
    die kleinen Erhebungen, die archäologisch interessant sind.
  * Ergebnis wird über st.cache_data cachebar gemacht (Input-Hash = Punkte
    + Auflösung), damit ein reiner UI-Interaktions-Rerun nicht neu rechnet.
"""
from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import datashader as ds
from scipy.ndimage import distance_transform_edt

from .config import LIMITS

logger = logging.getLogger(__name__)


class RasterizationError(Exception):
    pass


def _safe_grid_shape(width_m: float, height_m: float, res: float) -> Tuple[int, int]:
    if res <= 0:
        raise RasterizationError("Auflösung muss > 0 sein.")

    cols = max(LIMITS.MIN_GRID_DIM, int(width_m / res))
    rows = max(LIMITS.MIN_GRID_DIM, int(height_m / res))

    n_cells = cols * rows
    if n_cells > LIMITS.MAX_GRID_CELLS:
        # Auflösung automatisch so weit vergröbern, dass das Limit eingehalten wird,
        # statt einfach mit MemoryError abzustürzen.
        scale = math.sqrt(n_cells / LIMITS.MAX_GRID_CELLS)
        cols = max(LIMITS.MIN_GRID_DIM, int(cols / scale))
        rows = max(LIMITS.MIN_GRID_DIM, int(rows / scale))
        logger.warning(
            "Grid auf %sx%s begrenzt (angefragt hätte %s Zellen überschritten).",
            cols, rows, LIMITS.MAX_GRID_CELLS,
        )
    return cols, rows


def _fill_gaps(grid: np.ndarray) -> np.ndarray:
    """Füllt NaN-Zellen per Nearest-Neighbour statt globalem Mittelwert."""
    mask = np.isnan(grid)
    if not mask.any():
        return grid
    if mask.all():
        raise RasterizationError("Rasterisierung ergab ein komplett leeres Grid.")
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return grid[tuple(idx)]


@st.cache_data(show_spinner=False, max_entries=8)
def rasterize_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, res: float) -> np.ndarray:
    """Rasterisiert Punkte via Datashader (Mean-Aggregation) zu einem Höhenraster."""
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    min_x, max_x = df.x.min(), df.x.max()
    min_y, max_y = df.y.min(), df.y.max()

    cols, rows = _safe_grid_shape(max_x - min_x, max_y - min_y, res)

    cvs = ds.Canvas(
        plot_width=cols,
        plot_height=rows,
        x_range=(min_x, max_x),
        y_range=(min_y, max_y),
    )
    agg = cvs.points(df, "x", "y", ds.mean("z"))
    grid = np.array(agg.values, dtype=np.float32)
    return _fill_gaps(grid)
