import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import laplace, gaussian_filter
import plotly.graph_objects as go
import os

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäo-Analyse", layout="wide")

st.title("🏛️ Archäologische LiDAR-Prospektion")
st.markdown("Lade eine `.xyz` Punktwolke hoch, um die volle archäologische Analyse zu starten.")

# --- ANALYSE FUNKTIONEN (DEIN ORIGINAL-CODE) ---

def calculate_hillshade(data, azimuth=315, angle_altitude=45, dx=1.0, dy=1.0):
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)
    gy, gx = np.gradient(data, dy, dx)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    hillshade = (np.cos(altitude_rad) * np.cos(slope)) + \
                (np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return ((hillshade + 1) / 2).astype(np.float32)

def calculate_multi_hillshade(data, dx=1.0, dy=1.0, altitude=45):
    h1 = calculate_hillshade(data, azimuth=315, angle_altitude=altitude, dx=dx, dy=dy)
    h2 = calculate_hillshade(data, azimuth=45, angle_altitude=altitude, dx=dx, dy=dy)
    h3 = calculate_hillshade(data, azimuth=135, angle_altitude=altitude, dx=dx, dy=dy)
    h4 = calculate_hillshade(data, azimuth=225, angle_altitude=altitude, dx=dx, dy=dy)
    return ((h1 + h2 + h3 + h4) / 4.0).astype(np.float32)

def calculate_slope(data, dx=1.0, dy=1.0):
    gy, gx = np.gradient(data, dy, dx)
    tan_slope = np.sqrt(gx**2 + gy**2)
    slope_deg = np.rad2deg(np.arctan(tan_slope))
    p_high = np.nanpercentile(slope_deg[~np.isnan(slope_deg)], 98)
    return np.clip(slope_deg, 0, p_high)

def calculate_curvature(data):
    relief = -laplace(data)
    valid_relief = relief[~np.isnan(relief)]
    p_low, p_high = np.percentile(valid_relief, (2, 98))
    relief_normalized = np.clip(relief, p_low, p_high)
    return (relief_normalized - np.nanmin(relief_normalized)) / (np.nanmax(relief_normalized) - np.nanmin(relief_normalized))

def calculate_residual_topography(data, sigma=10):
    smoothed_data = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed_data
    valid_residual = residual[~np.isnan(residual)]
    p_low, p_high = np.percentile(valid_residual, (5, 95))
    residual_normalized = np.clip(residual, p_low, p_high)
    return (residual_normalized - np.nanmin(residual_normalized)) / (np.nanmax(residual_normalized) - np.nanmin(residual_normalized))

def calculate_composite_viz(mds_data, lrm_data, gain=0.3):
    lrm_influence = (lrm_data - 0.5) * gain
    composite = mds_data + lrm_influence
    return np.clip(composite, 0.0, 1.0)

# --- SIDEBAR EINSTELLUNGEN ---
with st.sidebar:
    st.header("⚙️ Einstellungen")
    uploaded_file = st.file_uploader("XYZ Datei hochladen", type=["xyz", "txt"])
    grid_res = st.number_input("Raster-Auflösung (Meter)", value=1.0, step=0.1)
    lrm_sigma = st.slider("LRM Glättung (Sigma)", 5, 30, 10)
    comp_gain = st.slider("Composite Gain", 0.1, 1.0, 0.3)
    z_exag = st.slider("3D Z-Überhöhung", 0.1, 1.0, 0.3)

# --- HAUPTBEREICH ---
if uploaded_file:
    with st.spinner("Verarbeite LiDAR-Daten..."):
        # Daten laden
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['x', 'y', 'z'], dtype=np.float32)
        
        # Grid/Raster erstellen
        grid_x, grid_y = np.mgrid[df.x.min():df.x.max():grid_res, df.y.min():df.y.max():grid_res]
        dgm_data = griddata(df[['x', 'y']].values, df['z'].values, (grid_x, grid_y), method='linear').T
        dgm_data = np.nan_to_num(dgm_data, nan=np.nanmean(dgm_data))

        # Analysen berechnen
        nw_hill = calculate_hillshade(dgm_data, dx=grid_res, dy=grid_res)
        mds = calculate_multi_hillshade(dgm_data, dx=grid_res, dy=grid_res)
        lrm = calculate_residual_topography(dgm_data, sigma=lrm_sigma)
        curv = calculate_curvature(dgm_data)
        slope = calculate_slope(dgm_data, dx=grid_res, dy=grid_res)
        comp = calculate_composite_viz(mds, lrm, gain=comp_gain)

        # 2D GRID ANZEIGE (2 x 3)
        st.subheader("2D Analyse-Modelle")
        plots = [
    ("NW Hillshade", nw_hill, "gist_gray"),
    ("MDS Composite", mds, "gist_gray"),
    ("Final Composite", comp, "gist_gray"),
    ("Restrelief (LRM)", lrm, "RdBu"),
    ("Lokale Krümmung", curv, "RdYlGn"),
    ("Hangneigung (Slope)", slope, "plasma")
#

