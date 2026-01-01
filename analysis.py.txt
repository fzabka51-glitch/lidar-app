import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def elevation_histogram(z, bins=50):
    fig, ax = plt.subplots()
    ax.hist(z.flatten(), bins=bins, color='gray')
    ax.set_title("Höhenverteilung")
    ax.set_xlabel("Höhe (m)")
    ax.set_ylabel("Anzahl")
    return fig

def slope_stats(slope):
    return {
        "Minimum": float(np.nanmin(slope)),
        "Maximum": float(np.nanmax(slope)),
        "Mittelwert": float(np.nanmean(slope)),
        "Median": float(np.nanmedian(slope))
    }

def extract_profile(array, p1, p2, n=200):
    x = np.linspace(p1[0], p2[0], n).astype(int)
    y = np.linspace(p1[1], p2[1], n).astype(int)
    vals = [array[yi, xi] for xi, yi in zip(x, y)]
    return pd.DataFrame({"Distanz": np.arange(n), "Höhe": vals})
