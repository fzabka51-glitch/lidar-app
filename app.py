import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datashader as ds

# --- LOGIC LAYER (ENGINE) ---
class LidarProcessor:
    """Verarbeitet LiDAR-Daten performant mittels Rasterisierung."""
    
    @staticmethod
    @st.cache_data(show_spinner="Verarbeite Daten...")
    def create_grid(df, res):
        """Erzeugt ein Raster aus der Punktwolke."""
        cvs = ds.Canvas(
            plot_width=int((df.x.max() - df.x.min()) / res),
            plot_height=int((df.y.max() - df.y.min()) / res),
            x_range=(df.x.min(), df.x.max()),
            y_range=(df.y.min(), df.y.max())
        )
        agg = cvs.points(df, 'x', 'y', ds.mean('z'))
        return np.array(agg.values, dtype=np.float32)

    @staticmethod
    def calculate_relief(grid, azimuth=315, altitude=45):
        """Berechnet archäologische Schattierung (Hillshade)."""
        az_rad = np.deg2rad(azimuth)
        alt_rad = np.deg2rad(altitude)
        gy, gx = np.gradient(grid)
        slope = np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(-gy, gx)
        shade = (np.cos(alt_rad) * np.cos(slope)) + \
                (np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
        return ((shade + 1) / 2).clip(0, 1)

# --- UI LAYER (STREAMLIT) ---
def main():
    st.set_page_config(page_title="LiDAR Archäologie", layout="wide")
    st.title("🏛️ LiDAR Archäologie Analyse")
    
    # Sidebar für Parameter (Context Engineering)
    with st.sidebar:
        st.header("Einstellungen")
        file = st.file_uploader("XYZ Datei laden", type=["xyz", "txt"])
        res = st.slider("Raster-Auflösung (m)", 0.1, 5.0, 1.0)
        z_exag = st.slider("3D-Überhöhung", 1.0, 5.0, 1.5)

    if file:
        try:
            # Daten laden (SPEC: Spezifisch & Klar)
            df = pd.read_csv(file, sep=None, engine='python', names=['x', 'y', 'z'])
            
            # Prozessierung
            proc = LidarProcessor()
            grid = proc.create_grid(df, res)
            grid = np.nan_to_num(grid, nan=np.nanmean(grid))
            
            tab1, tab2 = st.tabs(["🖼️ 2D Relief", "🌐 3D Modell"])
            
            with tab1:
                st.subheader("Digitales Geländemodell")
                hillshade = proc.calculate_relief(grid)
                fig, ax = plt.subplots()
                ax.imshow(hillshade, cmap='gray', origin='lower')
                ax.axis('off')
                st.pyplot(fig)
                
            with tab2:
                st.subheader("Interaktive Prospektion")
                # Downsampling für Plotly-Performance
                step = max(1, int(grid.size / 100000))
                fig3d = go.Figure(data=[go.Surface(z=grid[::step, ::step])])
                fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=z_exag/2)))
                st.plotly_chart(fig3d, use_container_width=True)
                
        except Exception as e:
            st.error(f"Fehler bei der Verarbeitung: {e}")
    else:
        st.info("Bitte laden Sie eine Datei hoch, um die Analyse zu starten.")

if __name__ == "__main__":
    main()
