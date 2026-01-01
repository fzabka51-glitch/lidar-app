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
st.markdown("Vollständige Analyse: Hillshade, MDS, LRM, Curvature, Slope & Fusion")

# --- ALLE FUNKTIONEN ---

def calculate_hillshade(data, azimuth=315, angle_altitude=45, dx=1.0, dy=1.0):
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)
    gy, gx = np.gradient(data, dy, dx)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    hillshade = (np.cos(altitude_rad) * np.cos(slope)) + \
                (np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return ((hillshade + 1) / 2).astype(np.float32)

def calculate_multi_hillshade(data, dx=1.0, dy=1.0):
    h1 = calculate_hillshade(data, 315, 45, dx, dy)
    h2 = calculate_hillshade(data, 45, 45, dx, dy)
    h3 = calculate_hillshade(data, 135, 45, dx, dy)
    h4 = calculate_hillshade(data, 225, 45, dx, dy)
    return ((h1 + h2 + h3 + h4) / 4.0)

def calculate_slope(data, dx=1.0, dy=1.0):
    gy, gx = np.gradient(data, dy, dx)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))
    return np.clip(slope_deg, 0, np.nanpercentile(slope_deg, 98))

def calculate_curvature(data):
    relief = -laplace(data)
    p_low, p_high = np.percentile(relief[~np.isnan(relief)], (2, 98))
    relief = np.clip(relief, p_low, p_high)
    return (relief - np.nanmin(relief)) / (np.nanmax(relief) - np.nanmin(relief))

def calculate_residual_topography(data, sigma=10):
    smoothed = gaussian_filter(data, sigma=sigma)
    res = data - smoothed
    p_low, p_high = np.percentile(res[~np.isnan(res)], (5, 95))
    res = np.clip(res, p_low, p_high)
    return (res - np.nanmin(res)) / (np.nanmax(res) - np.nanmin(res))

def calculate_composite(mds, lrm, gain=0.3):
    return np.clip(mds + (lrm - 0.5) * gain, 0, 1)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Parameter")
    uploaded_file = st.file_uploader("XYZ Datei laden", type=["xyz", "txt"])
    grid_res = st.number_input("Auflösung (m)", value=1.0, step=0.1)
    lrm_sigma = st.slider("LRM Filterstärke", 5, 30, 10)
    comp_gain = st.slider("Fusion Intensität", 0.1, 1.0, 0.3)
    z_exag = st.slider("3D Überhöhung", 0.1, 1.0, 0.3)

# --- VERARBEITUNG ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['x', 'y', 'z'], dtype=np.float32)
        grid_x, grid_y = np.mgrid[df.x.min():df.x.max():grid_res, df.y.min():df.y.max():grid_res]
        dgm = griddata(df[['x', 'y']].values, df['z'].values, (grid_x, grid_y), method='linear').T
        dgm = np.nan_to_num(dgm, nan=np.nanmean(dgm))

        # Modelle berechnen
        nw_hill = calculate_hillshade(dgm, dx=grid_res, dy=grid_res)
        mds = calculate_multi_hillshade(dgm, dx=grid_res, dy=grid_res)
        lrm = calculate_residual_topography(dgm, sigma=lrm_sigma)
        curv = calculate_curvature(dgm)
        slope = calculate_slope(dgm, dx=grid_res, dy=grid_res)
        comp = calculate_composite(mds, lrm, gain=comp_gain)

        # Darstellung 2D
        st.subheader("Analyse-Ergebnisse")
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        r2_c1, r2_c2, r2_c3 = st.columns(3)

        with r1_c1:
            f, a = plt.subplots(); a.imshow(nw_hill, cmap='gray'); a.set_title("NW Hillshade"); a.axis('off'); st.pyplot(f)
        with r1_c2:
            f, a = plt.subplots(); a.imshow(mds, cmap='gray'); a.set_title("MDS (Multi)"); a.axis('off'); st.pyplot(f)
        with r1_c3:
            f, a = plt.subplots(); a.imshow(comp, cmap='gray'); a.set_title("Fusion (MDS+LRM)"); a.axis('off'); st.pyplot(f)
        with r2_c1:
            f, a = plt.subplots(); a.imshow(lrm, cmap='RdBu'); a.set_title("Restrelief (LRM)"); a.axis('off'); st.pyplot(f)
        with r2_c2:
            f, a = plt.subplots(); a.imshow(curv, cmap='RdYlGn'); a.set_title("Krümmung"); a.axis('off'); st.pyplot(f)
        with r2_c3:
            f, a = plt.subplots(); a.imshow(slope, cmap='plasma'); a.set_title("Slope"); a.axis('off'); st.pyplot(f)

        # 3D Modell
        st.subheader("3D Ansicht")
        fig3d = go.Figure(data=[go.Surface(z=dgm, surfacecolor=comp, colorscale='Greys')])
        fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=z_exag)), margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte .xyz Datei hochladen.")
