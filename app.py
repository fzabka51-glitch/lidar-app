import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io
import os

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro++", layout="wide")
st.title("🏛️ LiDAR Analyse & High-Performance 3D")

# --- FUNKTIONEN (KERN) ---

def rasterize_points(df, res):
    """Erzeugt blitzschnell ein Raster aus Millionen von Punkten."""
    cvs = ds.Canvas(
        plot_width=int((df.x.max() - df.x.min()) / res),
        plot_height=int((df.y.max() - df.y.min()) / res),
        x_range=(df.x.min(), df.x.max()),
        y_range=(df.y.min(), df.y.max())
    )
    agg = cvs.points(df, 'x', 'y', ds.mean('z'))
    return np.array(agg.values, dtype=np.float32)

def calc_hillshade(data, az=315, alt=45, res=1.0):
    az_rad, alt_rad = np.deg2rad(az), np.deg2rad(alt)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(alt_rad) * np.cos(slope)) + (np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)

def calc_mds(data, res=1.0):
    h1 = calc_hillshade(data, 315, 45, res)
    h2 = calc_hillshade(data, 45, 45, res)
    h3 = calc_hillshade(data, 135, 45, res)
    h4 = calc_hillshade(data, 225, 45, res)
    return (h1 + h2 + h3 + h4) / 4.0

def calc_lrm(data, sigma=15):
    smoothed = gaussian_filter(data, sigma=sigma)
    res = data - smoothed
    p_l, p_h = np.percentile(res, (5, 95))
    res = np.clip(res, p_l, p_h)
    return (res - res.min()) / (res.max() - res.min())

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Einstellungen")
    uploaded_file = st.file_uploader("XYZ Datei laden", type=["xyz", "txt"])
    grid_res = st.number_input("Rasterauflösung (m)", 0.1, 10.0, 1.0)
    lrm_sigma = st.slider("LRM Glättung (Strukturen)", 1, 50, 15)
    z_exag = st.slider("3D Überhöhung", 0.1, 5.0, 1.0)
    cmap_choice = st.selectbox("Farbschema", ["gray", "RdBu_r", "plasma", "terrain"])

# --- HAUPTBEREICH ---
if uploaded_file:
    try:
        # 1. Daten laden (Begrenzung auf 2 Mio für Streamlit Cloud Stabilität)
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['x','y','z'], dtype=np.float32)
        if len(df) > 2000000:
            df = df.sample(2000000, random_state=42)
            st.warning("⚠️ Datensatz auf 2 Mio. Punkte begrenzt.")

        # 2. Rasterung & Analyse
        gz = rasterize_points(df, grid_res)
        gz = np.nan_to_num(gz, nan=np.nanmean(gz))
        
        mds = calc_mds(gz, grid_res)
        lrm = calc_lrm(gz, lrm_sigma)
        slope = np.rad2deg(np.arctan(np.sqrt(np.square(np.gradient(gz, grid_res, grid_res)).sum(axis=0))))
        curv = -laplace(gz)
        comp = np.clip(mds + (lrm - 0.5) * 0.4, 0, 1)

        # TABS für die Ordnung
        tab1, tab2, tab3 = st.tabs(["🖼️ Analysen", "🌐 3D-Modell", "📊 Statistiken"])

        with tab1:
            st.subheader("Archäologische Gelände-Filter")
            results = {"MDS": mds, "LRM": lrm, "Fusion": comp, "Slope": slope}
            cols = st.columns(2)
            for i, (name, img) in enumerate(results.items()):
                with cols[i % 2]:
                    fig, ax = plt.subplots()
                    ax.imshow(img, cmap=cmap_choice)
                    ax.set_title(name)
                    ax.axis('off')
                    st.pyplot(fig)

        with tab2:
            st.subheader("Interaktives 3D-Relief")
            # Downsampling für flüssige 3D-Anzeige
            step = max(1, int(np.sqrt(gz.size / 100000)))
            fig3d = go.Figure(data=[go.Surface(
                z=gz[::step, ::step], 
                surfacecolor=comp[::step, ::step], 
                colorscale='Greys',
                showscale=False
            )])
            fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=z_exag)), height=800)
            st.plotly_chart(fig3d, use_container_width=True)

        with tab3:
            st.subheader("Höhenverteilung")
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(gz.flatten(), bins=50, color='gray')
            st.pyplot(fig_hist)
            
            st.write("**Gelände-Statistiken:**")
            st.write(f"Min Höhe: {gz.min():.2f}m | Max Höhe: {gz.max():.2f}m")

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte .xyz Datei in der Sidebar hochladen.")
