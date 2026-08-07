"""
UI-Schicht. Enthält bewusst KEINE Analyse-Algorithmen – nur Streamlit-
Widgets und das Zusammenstecken von io_utils/analysis/visualization.
"""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from .backend import GPU_AVAILABLE
from .config import DEFAULTS, LAYER_SPECS, LIMITS
from .io_utils import PYPROJ_AVAILABLE


@dataclass
class SidebarParams:
    epsg_code: int
    grid_res: float
    lrm_sigma: int
    z_exag: float
    view_mode: str
    use_gpu: bool


def render_sidebar() -> tuple:
    """Rendert die Sidebar und gibt (uploaded_file, SidebarParams) zurück."""
    with st.sidebar:
        st.header("⚙️ Parameter")
        uploaded_file = st.file_uploader(
            "Punktwolke laden (.xyz, .txt, .las, .laz)",
            type=["xyz", "txt", "las", "laz"],
        )

        st.divider()
        st.subheader("Geo-Referenz")
        epsg_code = st.number_input("EPSG Code (z.B. UTM 32N: 25832)", value=DEFAULTS.EPSG_CODE)
        if not PYPROJ_AVAILABLE:
            st.warning("⚠️ 'pyproj' nicht gefunden. Koordinatenumrechnung deaktiviert.")

        st.divider()
        st.subheader("Raster & Filter")
        grid_res = st.number_input(
            "Auflösung (m)", 0.1, 10.0, DEFAULTS.GRID_RES,
            help="Niedrigerer Wert = höhere Schärfe, aber mehr Rechenzeit/RAM "
                 f"(automatisch begrenzt auf max. {LIMITS.MAX_GRID_CELLS:,} Zellen).",
        )
        lrm_sigma = st.slider("LRM Glättung (Sigma)", 1, 50, DEFAULTS.LRM_SIGMA)

        st.subheader("3D-Eigenschaften")
        z_exag = st.slider("Z-Überhöhung", 0.1, 5.0, DEFAULTS.Z_EXAGGERATION, step=0.1)

        st.subheader("Anzeige")
        view_mode = st.radio("Ansicht 2D:", ["Gitter-Übersicht", "Einzelansicht"])

        use_gpu = True
        if GPU_AVAILABLE:
            use_gpu = st.checkbox("GPU-Beschleunigung nutzen (CuPy erkannt)", value=True)
        else:
            st.caption("ℹ️ Keine GPU/CuPy erkannt – Berechnung läuft auf CPU.")

        params = SidebarParams(
            epsg_code=int(epsg_code),
            grid_res=float(grid_res),
            lrm_sigma=int(lrm_sigma),
            z_exag=float(z_exag),
            view_mode=view_mode,
            use_gpu=use_gpu,
        )
    return uploaded_file, params


def render_location_info(lat, lon) -> None:
    if lat is not None and lon is not None:
        url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        st.markdown(f"**📍 Standort (Zentrum):** [{lat:.5f}, {lon:.5f}]({url})")


def render_2d_tab(result, view_mode: str) -> None:
    from .visualization import make_2d_figure  # lokal, um Zirkularimporte zu vermeiden

    if view_mode == "Gitter-Übersicht":
        cols = st.columns(2)
        for i, key in enumerate(LAYER_SPECS):
            name, cmap, _, data = result.layer_display(key)
            with cols[i % 2]:
                fig = make_2d_figure(data, name, cmap)
                st.pyplot(fig)
                import matplotlib.pyplot as plt
                plt.close(fig)
    else:
        keys = list(LAYER_SPECS.keys())
        labels = [LAYER_SPECS[k][0] for k in keys]
        sel_label = st.selectbox("Modell wählen:", labels)
        sel_key = keys[labels.index(sel_label)]
        name, cmap, _, data = result.layer_display(sel_key)
        fig = make_2d_figure(data, name, cmap, figsize=(10, 6))
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


def render_3d_tab(result, z_exag: float) -> None:
    from .visualization import make_3d_figure

    st.subheader("3D-Viewer")
    keys = list(LAYER_SPECS.keys())
    labels = [LAYER_SPECS[k][0] for k in keys]
    sel_label = st.selectbox("Wähle Analyse-Ebene für die 3D-Oberfläche:", labels, index=0)
    sel_key = keys[labels.index(sel_label)]
    name, cmap, show_scale, data = result.layer_display(sel_key)

    fig = make_3d_figure(
        grid=result.grid,
        texture=data,
        bounds=result.bounds,
        title=f"3D Ansicht: {name}",
        cmap=cmap,
        show_scale=show_scale,
        z_exaggeration=z_exag,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 Pro-Tipp für Schärfe: Auflösung in Sidebar auf 0.5m stellen und Z-Überhöhung auf ca. 1.0 erhöhen.")
