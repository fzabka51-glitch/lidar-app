"""
Reine Visualisierungsfunktionen: bekommen fertige Arrays, geben Figures
zurück. Keine Streamlit-Aufrufe hier (Testbarkeit!) – das UI-Modul
entscheidet, wie/wo eine Figure angezeigt wird.
"""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def make_2d_figure(data: np.ndarray, title: str, cmap: str, figsize: Tuple[float, float] = (6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    # origin='lower' korrigiert die sonst spiegelverkehrte Y-Achse von imshow
    ax.imshow(data, cmap=cmap, interpolation="none", origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def make_3d_figure(
    grid: np.ndarray,
    texture: np.ndarray,
    bounds: Tuple[float, float, float, float],
    title: str,
    cmap: str,
    show_scale: bool,
    z_exaggeration: float,
    max_render_points: int = 400_000,
):
    min_x, max_x, min_y, max_y = bounds
    step = max(1, int(np.sqrt(grid.size / max_render_points)))
    z_plot = grid[::step, ::step]
    surface_tex = texture[::step, ::step]

    x_vals = np.linspace(min_x, max_x, z_plot.shape[1])
    y_vals = np.linspace(min_y, max_y, z_plot.shape[0])

    fig = go.Figure(data=[go.Surface(
        x=x_vals,
        y=y_vals,
        z=z_plot,
        surfacecolor=surface_tex,
        colorscale=cmap,
        showscale=show_scale,
        lighting=dict(ambient=0.6, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=0.5),
        lightposition=dict(x=100, y=100, z=1000),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Höhe: %{z:.2f}m<extra></extra>",
    )])

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            aspectratio=dict(x=1, y=1, z=z_exaggeration),
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)"),
            zaxis=dict(title="Höhe (m)"),
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=40),
        title=title,
    )
    return fig
