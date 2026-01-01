import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import laplace, gaussian_filter
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäo-Analyse", layout="wide")
st.title("🏛️ Archäologische LiDAR-Prospektion")
st.markdown("Diese App führt eine vollständige archäologische Gelände-Analyse durch.")

# --- BERECHNUNGS-FUNKTIONEN ---

def get_hillshade(data, az=315, alt=45, res=1.0):
    az_rad, alt_rad = np.deg2rad(az), np.deg2rad(alt)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(alt_rad) * np.cos(slope)) + (np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)

def get_multi_hillshade(data, res=1.0):
    # Vier Himmelsrichtungen für MDS
    h1 = get_hillshade(data, 315, 45, res)
    h2 = get_hillshade(data, 45, 45, res)
    h3 = get_hillshade(data, 135, 45, res)
    h4 = get_hillshade(data, 225, 45, res)
    return (h1 + h2 + h3 + h4) / 4.0

def get_lrm(data, sigma_val=10):
    smoothed = gaussian_filter(data, sigma=sigma_val)
    diff = data - smoothed
    # Normalisierung auf 0-1 für die Anzeige
    p_low, p_high = np.percentile(diff, (5, 95))
    diff_clipped = np.clip(diff, p_low, p_high)
    return (diff_clipped - np.min(diff_clipped)) / (np.max(diff_clipped) - np.min(diff_clipped))

def get_slope(data, res=1.0):
    gy, gx = np.gradient(data, res, res)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))
    return np.clip(slope_deg, 0, np.percentile(slope_deg, 98))

# --- SEITENLEISTE (EINSTELLUNGEN) ---
with st.sidebar:
    st.header("1. Daten-Upload")
    uploaded_file = st.file_uploader("XYZ-Datei wählen", type=["xyz", "txt"])
    
    st.header("2. Parameter")
    resolution = st.number_input("Raster-Auflösung (m)", value=1.0, min_value=0.1, step=0.1)
    lrm_filter = st.slider("LRM Filterstärke (Sigma)", 5, 30, 10)
    z_scale = st.slider("3D Überhöhung", 0.1, 1.0, 0.3)

# --- HAUPTFENSTER ---
if uploaded_file is not None:
    try:
        with st.spinner("Verarbeite Daten..."):
            # Daten einlesen
            df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['x', 'y', 'z'])
            
            # Raster erstellen (Gridding)
            xi = np.arange(df.x.min(), df.x.max(), resolution)
            yi = np.arange(df.y.min(), df.y.max(), resolution)
            grid_x, grid_y = np.meshgrid(xi, yi)
            
            # Interpolation
            grid_z = griddata((df.x, df.y), df.z, (grid_x, grid_y), method='linear')
            grid_z = np.nan_to_num(grid_z, nan=np.nanmean(grid_z))

            # Analysen berechnen
            mds = get_multi_hillshade(grid_z, resolution)
            lrm = get_lrm(grid_z, lrm_filter)
            slope = get_slope(grid_z, resolution)
            
            # Ergebnisse visualisieren
            st.subheader("📊 Analyse-Ergebnisse")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1, ax1 = plt.subplots()
                ax1.imshow(mds, cmap='gray')
                ax1.set_title("MDS Hillshade")
                ax1.axis('off')
                st.pyplot(fig1)
                
            with col2:
                fig2, ax2 = plt.subplots()
                ax2.imshow(lrm, cmap='RdBu_r')
                ax2.set_title("LRM (Relief)")
                ax2.axis('off')
                st.pyplot(fig2)
                
            with col3:
                fig3, ax3 = plt.subplots()
                ax3.imshow(slope, cmap='plasma')
                ax3.set_title("Hangneigung")
                ax3.axis('off')
                st.pyplot(fig3)

            # 3D Ansicht
            st.divider()
            st.subheader("🧊 Interaktives 3D-Modell")
            fig_3d = go.Figure(data=[go.Surface(z=grid_z, colorscale='Greys', surfacecolor=mds)])
            fig_3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=z_scale)))
            st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler bei der Berechnung: {e}")
else:
    st.info("Bitte lade eine .xyz-Datei über die Seitenleiste hoch.")
