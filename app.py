import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import pyproj
from datetime import datetime

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide", page_icon="🏛️")

# Custom CSS für besseres UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- FUNKTIONEN ---

def convert_coords(x, y, from_epsg=25832):
    """
    Wandelt Koordinaten um (Standard: UTM Zone 32N - oft für DE LiDAR genutzt).
    Gibt Lat/Lon für Google Maps zurück.
    """
    try:
        transformer = pyproj.Transformer.from_crs(f"epsg:{from_epsg}", "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except:
        return None, None

def rasterize_points(df, res):
    cvs = ds.Canvas(
        plot_width=int((df.x.max() - df.x.min()) / res),
        plot_height=int((df.y.max() - df.y.min()) / res),
        x_range=(df.x.min(), df.x.max()),
        y_range=(df.y.min(), df.y.max())
    )
    agg = cvs.points(df, 'x', 'y', ds.mean('z'))
    return np.array(agg.values, dtype=np.float32)

def calculate_hillshade(data, azimuth=315, angle_altitude=45, res=1.0):
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(altitude_rad) * np.cos(slope)) + \
            (np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)

def calculate_lrm(data, sigma=15):
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p_low, p_high = np.percentile(residual, (5, 95))
    res_clipped = np.clip(residual, p_low, p_high)
    res_min, res_max = res_clipped.min(), res_clipped.max()
    return (res_clipped - res_min) / (res_max - res_min) if res_max > res_min else np.full_like(residual, 0.5)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/archaeology.png", width=80)
    st.header("LiDAR Pro Control")
    uploaded_file = st.file_uploader("XYZ Datei laden", type=["xyz", "txt", "csv"])
    
    st.divider()
    epsg_code = st.number_input("EPSG Code (z.B. UTM 32N: 25832)", value=25832)
    grid_res = st.slider("Raster-Auflösung (m)", 0.2, 5.0, 1.0)
    z_exag = st.slider("3D Überhöhung", 0.5, 5.0, 1.5)
    
    if uploaded_file:
        st.success("Datei bereit!")

# --- HAUPTBEREICH ---
st.title("🏛️ LiDAR Archäologie & Gelände-Analyse")

if uploaded_file:
    # 1. Daten laden
    df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['x','y','z'])
    
    # Koordinaten-Zentrum für Google Maps
    center_x, center_y = df.x.mean(), df.y.mean()
    lat, lon = convert_coords(center_x, center_y, epsg_code)

    # Dashboard Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Punkte", f"{len(df):,}")
    m2.metric("Min Höhe", f"{df.z.min():.2f} m")
    m3.metric("Max Höhe", f"{df.z.max():.2f} m")
    
    if lat and lon:
        google_maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        m4.markdown(f"**📍 Standort**\n\n[{lat:.5f}, {lon:.5f}]({google_maps_url})")
    
    # 2. Prozessierung
    with st.spinner("Generiere Geländemodelle..."):
        gz = rasterize_points(df, grid_res)
        gz = np.nan_to_num(gz, nan=np.nanmean(gz))
        
        mds = calculate_hillshade(gz, 315, 45, grid_res)
        lrm = calculate_lrm(gz, 15)
        # Kombiniertes Modell für Archäologen
        combined = np.clip(mds * 0.7 + lrm * 0.3, 0, 1)

    # Tabs
    t1, t2, t3 = st.tabs(["🗺️ 2D Analyse", "🧊 3D Prospektion", "📍 Geo-Info"])

    with t1:
        st.subheader("Visualisierung der Oberflächenmerkmale")
        col_view, col_opt = st.columns([3, 1])
        
        model_type = col_opt.radio("Modell:", ["Kombiniert", "Schummerung", "Lokalrelief (LRM)"])
        active_data = combined if model_type == "Kombiniert" else (mds if model_type == "Schummerung" else lrm)
        
        # Plotly 2D Heatmap (Zoombar!)
        fig2d = px.imshow(active_data, color_continuous_scale='gray' if "LRM" not in model_type else 'RdBu_r',
                          origin='lower', aspect='equal')
        fig2d.update_layout(margin=dict(l=0,r=0,b=0,t=0), coloraxis_showscale=False)
        st.plotly_chart(fig2d, use_container_width=True)

    with t2:
        st.subheader("Interaktive 3D Geländeoberfläche")
        # Downsampling für Performance
        step = max(1, int(np.sqrt(gz.size / 200000)))
        z_plot = gz[::step, ::step]
        tex_plot = active_data[::step, ::step]

        fig3d = go.Figure(data=[go.Surface(z=z_plot, surfacecolor=tex_plot, colorscale='gray')])
        fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=z_exag/2),
                                      xaxis_visible=False, yaxis_visible=False),
                            height=700, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig3d, use_container_width=True)

    with t3:
        st.subheader("Geografische Einordnung")
        if lat and lon:
            # Einfache Map-Anzeige
            map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
            st.map(map_df)
            st.info(f"Zentrum der Punktwolke (EPSG:{epsg_code}): X={center_x:.2f}, Y={center_y:.2f}")
        else:
            st.warning("Keine gültigen Geo-Koordinaten gefunden. Bitte EPSG-Code prüfen.")

    # Export
    st.divider()
    if st.button("Download Analyse als Bild"):
        buf = io.BytesIO()
        plt.imsave(buf, active_data, cmap='gray', format='png')
        st.download_button(label="Bild speichern", data=buf.getvalue(), 
                           file_name=f"lidar_export_{datetime.now().strftime('%Y%m%d_%H%M')}.png", mime="image/png")

else:
    # Willkommens-Bildschirm
    st.info("👋 Willkommen! Bitte lade eine .xyz Datei hoch, um mit der archäologischen Analyse zu beginnen.")
    st.markdown("""
    ### Features dieser Version:
    - **Automatische Geo-Links:** Erzeugt Google Maps Links aus UTM-Koordinaten.
    - **LRM-Filter:** Macht kleinste Bodenstrukturen (Wälle, Gräben) sichtbar.
    - **Interaktives 2D:** Zoombar mittels Plotly.
    - **Performance:** Nutzt Datashader für schnelle Rasterisierung.
    """)
