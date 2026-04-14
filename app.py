import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, laplace
import datashader as ds

# --- KONFIGURATION ---
st.set_page_config(page_title="LiDAR Prospektion Pro", layout="wide")

# --- CORE FUNKTIONEN ---

def rasterize_points(df, res):
    """Rasterisierung mit Datashader."""
    cvs = ds.Canvas(
        plot_width=int((df.x.max() - df.x.min()) / res),
        plot_height=int((df.y.max() - df.y.min()) / res),
        x_range=(df.x.min(), df.x.max()),
        y_range=(df.y.min(), df.y.max())
    )
    agg = cvs.points(df, 'x', 'y', ds.mean('z'))
    return np.array(agg.values, dtype=np.float32)

def get_slope(data, res):
    """Berechnet die Hangneigung (Slope)."""
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    return np.rad2deg(slope)

def get_hillshade(data, azimuth, altitude, res):
    """Klassisches Hillshading."""
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(altitude)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(altitude_rad) * np.cos(slope)) + \
            (np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return np.clip(shade, 0, 1)

def get_multi_hillshade(data, res):
    """MDS: Multi-Directional Shading aus 8 Richtungen für maximale Struktur-Erkennung."""
    combined = np.zeros_like(data)
    for az in range(0, 360, 45):
        combined += get_hillshade(data, az, 35, res)
    return combined / 8.0

def get_lrm(data, sigma):
    """Local Relief Model - Extrahiert die Mikro-Topographie."""
    low_pass = gaussian_filter(data, sigma=sigma)
    diff = data - low_pass
    # Kontrast-Stretch auf 0-1
    p_low, p_high = np.percentile(diff, (2, 98))
    diff_clipped = np.clip(diff, p_low, p_high)
    return (diff_clipped - p_low) / (p_high - p_low)

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("🏛️ Prospektions-Parameter")
    uploaded_file = st.file_uploader("XYZ Datei hochladen", type=["xyz", "txt"])
    
    st.divider()
    grid_res = st.slider("Raster-Auflösung (m)", 0.1, 2.0, 0.5, step=0.1)
    lrm_sigma = st.slider("Struktur-Fokus (Sigma)", 5, 50, 15, help="Kleiner = feine Mauern, Größer = breite Gräben")
    
    st.divider()
    st.subheader("3D Darstellung")
    z_exag = st.slider("Z-Überhöhung", 1.0, 15.0, 3.0)
    overlay_opacity = st.slider("Overlay Deckkraft", 0.0, 1.0, 0.7)

# --- MAIN APP ---
if uploaded_file:
    try:
        # 1. Load Data
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
        if len(df) > 2000000:
            df = df.sample(2000000)
            st.warning("⚠️ Datensatz auf 2 Mio. Punkte reduziert.")

        # 2. Process
        with st.spinner("Berechne archäologische Layer..."):
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            # Layer Generierung
            slope = get_slope(gz, grid_res)
            mds = get_multi_hillshade(gz, grid_res)
            lrm = get_lrm(gz, lrm_sigma)
            
            # Profi-Ansatz: RGB Composite (Slope = Rot, LRM = Grün, MDS = Blau)
            # Das lässt archäologische Anomalien farblich hervortreten
            slope_norm = (slope - slope.min()) / (slope.max() - slope.min())
            composite_rgb = np.stack([slope_norm, lrm, mds], axis=-1)

        tab1, tab2 = st.tabs(["🔍 Analyse-Dashboard", "🌐 Interaktives 3D"])

        with tab1:
            st.subheader("Visualisierungs-Vergleich")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Multi-Directional Hillshade (MDS)**")
                fig1, ax1 = plt.subplots()
                ax1.imshow(mds, cmap="gray", origin="lower")
                ax1.axis("off")
                st.pyplot(fig1)
                
                st.write("**Hangneigung (Slope)**")
                fig2, ax2 = plt.subplots()
                ax2.imshow(slope, cmap="magma", origin="lower")
                ax2.axis("off")
                st.pyplot(fig2)

            with col2:
                st.write("**Local Relief Model (Röntgenblick)**")
                fig3, ax3 = plt.subplots()
                ax3.imshow(lrm, cmap="RdGy", origin="lower")
                ax3.axis("off")
                st.pyplot(fig3)
                
                st.write("**Archäologische Fusion (RGB)**")
                st.info("Rot: Steilheit | Grün: Relief | Blau: Schatten")
                fig4, ax4 = plt.subplots()
                ax4.imshow(composite_rgb, origin="lower")
                ax4.axis("off")
                st.pyplot(fig4)

        with tab2:
            st.subheader("3D Gelände-Exploration")
            
            # Downsampling für 3D Performance
            step = max(1, gz.shape[0] // 300)
            z_3d = gz[::step, ::step]
            
            # Wähle Textur
            tex_mode = st.radio("Oberfläche:", ["RGB Fusion", "Röntgen (LRM)", "Schattierung (MDS)"], horizontal=True)
            if tex_mode == "RGB Fusion":
                tex_3d = composite_rgb[::step, ::step]
                colorscale = None # RGB nutzt direkt die Daten
            elif tex_mode == "Röntgen (LRM)":
                tex_3d = lrm[::step, ::step]
                colorscale = "RdGy"
            else:
                tex_3d = mds[::step, ::step]
                colorscale = "gray"

            fig3d = go.Figure(data=[go.Surface(
                z=z_3d,
                surfacecolor=tex_3d if colorscale else None,
                colorscale=colorscale,
                opacity=overlay_opacity
            )])

            fig3d.update_layout(
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=z_exag/10),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(title="Höhe")
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=800
            )
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte XYZ-Datei hochladen.")
