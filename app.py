import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import pyproj
from typing import Tuple, Optional

# --- CORE ENGINE (Logic Layer) ---
class LidarAnalyzer:
    """Klasse zur performanten Verarbeitung von LiDAR-Rasterdaten."""
    
    @staticmethod
    @st.cache_data(show_spinner="Rasterisiere Punktwolke...")
    def rasterize_points(df: pd.DataFrame, res: float) -> np.ndarray:
        """Wandelt XYZ-Punkte blitzschnell in ein Raster um."""
        cvs = ds.Canvas(
            plot_width=int((df.x.max() - df.x.min()) / res),
            plot_height=int((df.y.max() - df.y.min()) / res),
            x_range=(df.x.min(), df.x.max()),
            y_range=(df.y.min(), df.y.max())
        )
        agg = cvs.points(df, 'x', 'y', ds.mean('z'))
        return np.array(agg.values, dtype=np.float32)

    @staticmethod
    def calculate_hillshade(data: np.ndarray, azimuth: float, altitude: float, res: float) -> np.ndarray:
        """Berechnet archäologische Schummerung."""
        az_rad = np.deg2rad(azimuth)
        alt_rad = np.deg2rad(altitude)
        gy, gx = np.gradient(data, res, res)
        slope = np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(-gy, gx)
        shade = (np.cos(alt_rad) * np.cos(slope)) + \
                (np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
        return ((shade + 1) / 2).clip(0, 1)

# --- UI LAYER (Streamlit) ---
def main():
    st.title("🏛️ LiDAR Archäologie Pro")
    st.markdown("---")

    <context>
    Sidebar-Konfiguration für Georeferenzierung und Filter.
    </context>
    
    with st.sidebar:
        st.header("⚙️ Parameter")
        uploaded_file = st.file_uploader("XYZ Datei hochladen", type=["xyz", "txt"])
        grid_res = st.number_input("Auflösung (m)", 0.1, 10.0, 1.0)
        epsg_code = st.number_input("EPSG Code", value=25832)
        z_exag = st.slider("3D Z-Überhöhung", 0.1, 5.0, 1.5)

    if uploaded_file:
        try:
            # 1. Daten laden (Cached)
            df = pd.read_csv(uploaded_file, sep=None, engine='python', names=['x', 'y', 'z'])
            
            if len(df) > 2_000_000:
                df = df.sample(2_000_000)
                st.warning("Datensatz auf 2 Mio. Punkte reduziert für UI-Performance.")

            # 2. Prozessierung
            analyzer = LidarAnalyzer()
            gz = analyzer.rasterize_points(df, grid_res)
            # NaNs bereinigen
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))

            # 3. Tabs für 2D/3D
            tab1, tab2 = st.tabs(["🖼️ 2D-Analyse", "🌐 3D-Viewer"])

            with tab1:
                st.subheader("Relief-Visualisierung")
                hs = analyzer.calculate_hillshade(gz, 315, 45, grid_res)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(hs, cmap="gray", origin='lower')
                ax.axis('off')
                st.pyplot(fig)

            with tab2:
                # 3D Plotly Surface
                step = max(1, int(np.sqrt(gz.size / 200000)))
                z_plot = gz[::step, ::step]
                
                fig3d = go.Figure(data=[go.Surface(z=z_plot, colorscale='Viridis')])
                fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=z_exag/2)))
                st.plotly_chart(fig3d, use_container_width=True)

        except Exception as e:
            st.error(f"Fehler bei der Verarbeitung: {e}")
    else:
        st.info("Bitte laden Sie eine XYZ-Datei hoch, um zu beginnen.")

if __name__ == "__main__":
    main()
