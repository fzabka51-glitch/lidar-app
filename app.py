import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import laplace, gaussian_filter
import plotly.graph_objects as go

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")
st.title("🏛️ LiDAR Analyse & Große 3D-Prospektion")

# --- ARCHÄOLOGISCHE ANALYSE-FUNKTIONEN ---

def calc_hillshade(data, az=315, alt=45, res=1.0):
    """Berechnet Schattierung aus einer bestimmten Richtung."""
    az_rad, alt_rad = np.deg2rad(az), np.deg2rad(alt)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(alt_rad) * np.cos(slope)) + (np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)

def calc_mds(data, res=1.0):
    """Multi-Directional Hillshade (MDS) aus 4 Richtungen."""
    h1 = calc_hillshade(data, 315, 45, res)
    h2 = calc_hillshade(data, 45, 45, res)
    h3 = calc_hillshade(data, 135, 45, res)
    h4 = calc_hillshade(data, 225, 45, res)
    return (h1 + h2 + h3 + h4) / 4.0

def calc_lrm(data, sigma=10):
    """Local Relief Model (LRM) zur Isolation von Mikro-Relief."""
    smoothed = gaussian_filter(data, sigma=sigma)
    res = data - smoothed
    if np.max(res) == np.min(res): return np.zeros_like(res)
    p_low, p_high = np.percentile(res, (5, 95))
    res = np.clip(res, p_low, p_high)
    return (res - np.min(res)) / (np.max(res) - np.min(res))

def calc_slope(data, res=1.0):
    """Berechnet die Hangneigung in Grad."""
    gy, gx = np.gradient(data, res, res)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))
    return np.clip(slope_deg, 0, np.nanpercentile(slope_deg, 98))

# --- SIDEBAR (STEUERUNG) ---
with st.sidebar:
    st.header("1. Daten-Upload")
    uploaded_file = st.file_uploader("XYZ Datei laden", type=["xyz", "txt"])
    
    st.header("2. Analyse-Parameter")
    grid_res = st.number_input("Raster-Auflösung (m)", value=1.0, step=0.1, min_value=0.1)
    lrm_sigma = st.slider("LRM Filterstärke (Strukturen)", 1, 50, 15)
    z_exag = st.slider("3D Überhöhung (Z-Achse)", 0.1, 5.0, 1.0)
    
    st.header("3. Anzeige")
    view_mode = st.radio("Modus:", ["Übersicht (Gitter)", "Einzelansicht (Groß)"])

# --- HAUPTBEREICH ---
if uploaded_file:
    try:
        # 1. Daten laden
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['x', 'y', 'z'], dtype=np.float32)
        
        # 2. Rasterung (Interpolation)
        xi = np.arange(df.x.min(), df.x.max(), grid_res)
        yi = np.arange(df.y.min(), df.y.max(), grid_res)
        gx, gy = np.meshgrid(xi, yi)
        gz = griddata((df.x, df.y), df.z, (gx, gy), method='linear')
        gz = np.nan_to_num(gz, nan=np.nanmean(gz))

        # 3. Berechnungen durchführen
        mds = calc_mds(gz, grid_res)
        lrm = calc_lrm(gz, lrm_sigma)
        slope = calc_slope(gz, grid_res)
        curv = -laplace(gz)
        # Fusion aus MDS und LRM für maximale Details
        comp = np.clip(mds + (lrm - 0.5) * 0.5, 0, 1)

        # Dictionary für die Anzeige
        results = {
            "MDS Hillshade (Relief)": (mds, "gray"),
            "LRM (Mikro-Strukturen)": (lrm, "RdBu_r"),
            "Fusion (MDS + LRM)": (comp, "gray"),
            "Hangneigung (Slope)": (slope, "plasma"),
            "Krümmung (Curvature)": (curv, "RdYlGn")
        }

        # 4. 2D Visualisierung
        if view_mode == "Übersicht (Gitter)":
            st.subheader("Analyse-Übersicht")
            cols = st.columns(3)
            for i, (name, (data, cmap)) in enumerate(results.items()):
                with cols[i % 3]:
                    fig, ax = plt.subplots()
                    if "Curvature" in name:
                        p_l, p_h = np.percentile(data, (2, 98))
                        data = np.clip(data, p_l, p_h)
                    ax.imshow(data, cmap=cmap)
                    ax.set_title(name, fontsize=10)
                    ax.axis('off')
                    st.pyplot(fig)
        else:
            sel = st.selectbox("Wähle eine Analyse für die Großansicht:", list(results.keys()))
            data, cmap = results[sel]
            fig, ax = plt.subplots(figsize=(12, 8))
            if "Curvature" in sel:
                p_l, p_h = np.percentile(data, (2, 98))
                data = np.clip(data, p_l, p_h)
            ax.imshow(data, cmap=cmap)
            ax.axis('off')
            st.pyplot(fig)

        # 5. GROSSES 3D MODELL
        st.divider()
        st.subheader("🌐 Interaktive 3D-Prospektion (Großansicht)")
        
        # Plotly Surface mit Fusions-Textur
        fig3d = go.Figure(data=[go.Surface(
            z=gz, 
            surfacecolor=comp, 
            colorscale='Greys',
            contours_z=dict(show=True, usecolormap=True, project_z=True, highlightcolor="limegreen")
        )])
        
        fig3d.update_layout(
            scene=dict(
                aspectratio=dict(x=1, y=1, z=z_exag),
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Höhe (m)'
            ),
            height=900, # Maximale Höhe für Details
            margin=dict(l=0, r=0, b=0, t=0)
        )
        
        st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler bei der Verarbeitung: {e}")
else:
    st.info("Willkommen! Bitte lade eine .xyz oder .txt LiDAR-Punktwolke in der Sidebar hoch.")
