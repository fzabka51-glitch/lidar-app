# lidar_app/backend.py

import numpy as np

def get_backend(use_gpu=False):
    """
    Gibt ein Backend-Objekt zurück.
    Minimalversion: CPU-only.
    """
    return {
        "name": "cpu",
        "to_numpy": to_numpy,
    }

def to_numpy(arr):
    """
    Stellt sicher, dass ein Array ein NumPy-Array ist.
    """
    if isinstance(arr, np.ndarray):
        return arr
    try:
        return np.asarray(arr)
    except Exception:
        raise TypeError("Array kann nicht in NumPy konvertiert werden.")

