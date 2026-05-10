import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io

# <system_check>
# Versuche pyproj für die Koordinatenumrechnung zu importieren
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
# </system_check>

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")
st.title("🏛️ LiDAR Analyse & High-Performance 3D")

# --- KOORDINATEN-FUNKTION ---
def convert_coords(x, y, from_epsg=25832):
    """Wandelt metrische Koordinaten in Lat/Lon um."""
    if not PYPROJ_AVAILABLE:
        return None, None
    try:
        transformer = pyproj.Transformer.from_crs(f"epsg:{from_epsg}", "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except:
        return None, None

# --- ARCHÄOLOGISCHE ANALYSE-FUNKTIONEN ---

def rasterize_points(df, res):
    """Blitzschnelle Rasterisierung mittels Datashader."""
    cvs = ds.Canvas(
        plot_width=int((df.x.max() - df.x.min()) / res),
        plot_height=int((df.y.max() - df.y.min()) / res),
        x_range=(df.x.min(), df.x.max()),
        y_range=(df.y.min(), df.y.max())
    )
    agg = cvs.points(df, 'x', 'y', ds.mean('z'))
    return np.array(agg.values, dtype=np.float32)

def calculate_hillshade(data, azimuth=315, angle_altitude=45, res=1.0):
    """Berechnet ein Schummerungsbild (Hillshade)."""
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(altitude_rad) * np.cos(slope)) + \
            (np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)

def calculate_multi_hillshade(data, res=1.0):
    """Multi-Directional Shading (MDS)."""
    h1 = calculate_hillshade(data, 315, 45, res)
    h2 = calculate_hillshade(data, 45, 45, res)
    h3 = calculate_hillshade(data, 135, 45, res)
    h4 = calculate_hillshade(data, 225, 45, res)
    return (h1 + h2 + h3 + h4) / 4.0

def calculate_lrm(data, sigma=15):
    """Local Relief Model (LRM) zur Visualisierung von Kleinformen."""
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p_low, p_high = np.percentile(residual, (5, 95))
    res_clipped = np.clip(residual, p_low, p_high)
    res_min, res_max = res_clipped.min(), res_clipped.max()
    if res_max > res_min:
        return (res_clipped - res_min) / (res_max - res_min)
    return np.full_like(residual, 0.5)

def calculate_slope(data, res=1.0):
    """Hangneigung in Grad."""
    gy, gx = np.gradient(data, res, res)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))
    p_high = np.nanpercentile(slope_deg, 98)
    return np.clip(slope_deg, 0, p_high)

def calculate_curvature(data):
    """Lokale Krümmung (Laplace)."""
    curv = -laplace(data)
    p_low, p_high = np.percentile(curv, (2, 98))
    curv_clipped = np.clip(curv, p_low, p_high)
    c_min, c_max = curv_clipped.min(), curv_clipped.max()
    if c_max > c_min:
        return (curv_clipped - c_min) / (c_max - c_min)
    return np.full_like(curv, 0.5)

def calculate_svf_approx(data):
    """Sky-View Factor Approximation (Offenheit des Reliefs)."""
    svf = -laplace(gaussian_filter(data, sigma=1.0))
    svf_norm = (svf - svf.min()) / (svf.max() - svf.min() + 1e-6)
    return 1.0 - svf_norm

def detect_anomalies(data, threshold=2.0):
    """Statistische Anomalien-Detektion (Z-Score)."""
    local_mean = gaussian_filter(data, sigma=5)
    local_std = np.sqrt(gaussian_filter((data - local_mean)**2, sigma=5))
    z_score = np.abs((data - local_mean) / (local_std + 1e-6))
    return np.where(z_score > threshold, 1.0, 0.0)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Parameter")
    uploaded_file = st.file_uploader("XYZ Datei laden", type=["xyz", "txt"])
    
    st.divider()
    st.subheader("Geo-Referenz")
    epsg_code = st.number_input("EPSG Code", value=25832)
    
    st.divider()
    st.subheader("Analyse-Optionen")
    grid_res = st.number_input("Auflösung (m)", 0.1, 10.0, 1.0)
    lrm_sigma = st.slider("LRM Glättung", 1, 50, 15)
    anomaly_thresh = st.slider("Anomalien-Schwellenwert", 1.0, 5.0, 2.0)
    
    st.subheader("3D-Eigenschaften")
    z_exag = st.slider("Z-Überhöhung", 0.1, 5.0, 0.5)
    view_mode = st.radio("Ansicht 2D:", ["Gitter-Übersicht", "Einzelansicht"])

# --- HAUPTBEREICH ---
if uploaded_file:
    try:
        # 1. Daten laden (SPEC: Spezifische Datentypen für Performance)
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
        if len(df) > 2000000:
            df = df.sample(2000000, random_state=42)
            st.warning("⚠️ Datensatz auf 2 Mio. Punkte reduziert.")

        min_x, max_x = df.x.min(), df.x.max()
        min_y, max_y = df.y.min(), df.y.max()

        # Standort berechnen
        lat, lon = convert_coords(df.x.mean(), df.y.mean(), epsg_code)
        if lat and lon:
            st.markdown(f"**📍 Zentrum:** [{lat:.5f}, {lon:.5f}](https://www.google.com/maps/search/?api=1&query={lat},{lon})")

        # 2. Berechnungen (CORE: Refine & Execute)
        with st.spinner("Analysiere Gelände..."):
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            # Modelle
            nw_h = calculate_hillshade(gz, 315, 45, grid_res)
            mds = calculate_multi_hillshade(gz, grid_res)
            lrm = calculate_lrm(gz, lrm_sigma)
            slope = calculate_slope(gz, grid_res)
            curv = calculate_curvature(gz)
            svf = calculate_svf_approx(gz)
            anom = detect_anomalies(lrm, anomaly_thresh)
            
            # Fusion
            comp = np.clip(mds + (lrm - 0.5) * 0.4 + (svf * 0.2), 0, 1)

            analysis_models = {
                "Final Composite (Fusion)": (comp, "gray", False),
                "Sky-View Factor (SVF)": (svf, "bone", False),
                "Anomalien-Karte": (anom, "YlOrRd", True),
                "NW Hillshade": (nw_h, "gray", False),
                "MDS Composite": (mds, "gray", False),
                "Restrelief (LRM)": (lrm, "RdBu", True),
                "Hangneigung (Slope)": (slope, "plasma", True),
                "Krümmung (Curvature)": (curv, "RdYlGn", True)
            }

        tab1, tab2 = st.tabs(["🖼️ 2D-Analyse", "🌐 3D-Prospektion"])

        with tab1:
            if view_mode == "Gitter-Übersicht":
                c1, c2 = st.columns(2)
                for i, (name, (data, cmap, _)) in enumerate(analysis_models.items()):
                    with [c1, c2][i % 2]:
                        fig, ax = plt.subplots()
                        ax.imshow(data, cmap=cmap, origin='lower')
                        ax.set_title(name)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                sel_2d = st.selectbox("Modell wählen:", list(analysis_models.keys()))
                data, cmap, _ = analysis_models[sel_2d]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(data, cmap=cmap, origin='lower')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

        with tab2:
            st.subheader("3D-Viewer")
            selected_texture = st.selectbox("Wähle Textur für 3D-Oberfläche:", list(analysis_models.keys()))
            tex_data, tex_cmap, show_scale = analysis_models[selected_texture]
            
            step = max(1, int(np.sqrt(gz.size / 400000)))
            z_plot = gz[::step, ::step]
            surface_tex = tex_data[::step, ::step]

            fig3d = go.Figure(data=[go.Surface(
                z=z_plot, surfacecolor=surface_tex, colorscale=tex_cmap, showscale=show_scale
            )])
            fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=z_exag)), height=800)
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte XYZ-Datei hochladen.")
