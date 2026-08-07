"""
Zentrale Konfiguration der LiDAR-Archäologie-App.

Alle "magischen Zahlen" (Limits, Defaults) leben hier statt verstreut
im Code, damit sie an einer Stelle geprüft und angepasst werden können.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AppLimits:
    MAX_POINTS: int = 2_000_000          # Harte Obergrenze für Punktwolken (RAM-Schutz)
    MAX_GRID_CELLS: int = 16_000_000     # z.B. 4000 x 4000 -> verhindert OOM bei sehr feiner Auflösung
    MIN_GRID_DIM: int = 4                # Unterhalb dessen ist ein Grid sinnlos / crasht Gradientenberechnung
    RANDOM_STATE: int = 42


@dataclass(frozen=True)
class DefaultParams:
    EPSG_CODE: int = 25832               # UTM Zone 32N (ETRS89) – gängig in DE
    GRID_RES: float = 1.0
    LRM_SIGMA: int = 15
    Z_EXAGGERATION: float = 0.5
    HILLSHADE_AZIMUTH: float = 315.0
    HILLSHADE_ALTITUDE: float = 45.0


LIMITS = AppLimits()
DEFAULTS = DefaultParams()

# Reihenfolge & Metadaten der Analyse-Layer (statt Dict-Literal mitten im UI-Code)
# key -> (Anzeigename, Colormap, zeigt_farbskala)
LAYER_SPECS = {
    "composite": ("Final Composite (Fusion)", "gray", False),
    "hillshade_nw": ("NW Hillshade", "gray", False),
    "mds": ("MDS Composite", "gray", False),
    "lrm": ("Restrelief (LRM)", "RdBu", True),
    "slope": ("Hangneigung (Slope)", "plasma", True),
    "curvature": ("Krümmung (Curvature)", "RdYlGn", True),
}
