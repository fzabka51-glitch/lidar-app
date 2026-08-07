"""
Backend-Abstraktion: nutzt CuPy (GPU) transparent, wenn verfügbar,
sonst NumPy (CPU). Der Rest der App importiert nur `xp` und weiß nicht,
auf welcher Hardware gerechnet wird.

Das ist der zentrale Hebel für "GPU-Optionen berücksichtigen", ohne dass
jede Funktion einzeln CUDA-Code enthalten muss.
"""
import numpy as np

try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


def get_backend(use_gpu: bool = True):
    """Liefert (xp, ist_gpu) – xp ist entweder cupy oder numpy."""
    if use_gpu and GPU_AVAILABLE:
        return cp, True
    return np, False


def to_numpy(arr):
    """Holt ein Array garantiert als NumPy-Array zurück (z.B. für Plotly/Matplotlib)."""
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)
