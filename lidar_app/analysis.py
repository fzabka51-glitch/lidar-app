"""
Pipeline-Orchestrierung: bündelt Rasterisierung + alle Terrain-Modelle
zu einem Ergebnisobjekt. Das ist die einzige Stelle, die "weiß", welche
Layer es gibt und in welcher Reihenfolge sie berechnet werden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from . import terrain_models as tm
from .config import LAYER_SPECS
from .raster import rasterize_points


@dataclass
class AnalysisResult:
    grid: np.ndarray                       # rohes Höhenraster (DTM)
    layers: Dict[str, np.ndarray]           # key -> normalisiertes Array
    bounds: Tuple[float, float, float, float]  # min_x, max_x, min_y, max_y

    def layer_display(self, key: str):
        """Gibt (Anzeigename, Colormap, zeigt_Skala, Daten) für einen Layer-Key zurück."""
        name, cmap, show_scale = LAYER_SPECS[key]
        return name, cmap, show_scale, self.layers[key]


def run_analysis(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    res: float, lrm_sigma: float, use_gpu: bool = True,
) -> AnalysisResult:
    grid = rasterize_points(x, y, z, res)

    nw_h = tm.calculate_hillshade(grid, 315, 45, res, use_gpu)
    mds = tm.calculate_multi_hillshade(grid, res, use_gpu)
    lrm = tm.calculate_lrm(grid, lrm_sigma)
    slope = tm.calculate_slope(grid, res, use_gpu)
    curv = tm.calculate_curvature(grid)
    composite = tm.calculate_composite(mds, lrm)

    layers = {
        "composite": composite,
        "hillshade_nw": nw_h,
        "mds": mds,
        "lrm": lrm,
        "slope": slope,
        "curvature": curv,
    }
    bounds = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    return AnalysisResult(grid=grid, layers=layers, bounds=bounds)
