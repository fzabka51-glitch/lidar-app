import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io
import requests
from fiona.crs import from_epsg
from streamlit_folium import st_folium
import folium

# Versuche pyproj für die Koordinatenumrechnung zu importieren
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro + API", layout="wide")

# --- CSS FÜR MODERNES UI ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- HILFSFUNKTIONEN ---

def convert_coords(x, y, from_epsg_code=25832, to_epsg_code=4326):
    """Wandelt Koordinaten um."""
    if not PYPROJ_AVAILABLE: return x, y
    transformer = pyproj.Transformer.from_crs(f"epsg:{from_epsg_code}", f"epsg:{to_epsg_code}", always_xy=True)
    return transformer.transform(x, y)

def get_wms_elevation(bbox, width=512, height=512):
    """
    Beispielhaft: Ruft ein DGM via WMS/WCS ab.
    Hier nutzen wir den WMS des Landes NRW (DGM 1m).
    """
    # Beispiel-URL für NRW DGM (Schummerung oder Rohdaten)
    # Hinweis: Echte Rohdaten-WCS-Abfragen benötigen oft spezifische Libs wie 'owslib'
    # Wir simulieren hier den Abruf eines 32-bit Float GeoTIFFs
    base_url = "https://www.wms.nrw.de/geobasis/wms_nw_dgm-schummerung"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "nw_dgm_schummerung", # In der Realität würde man hier den DGM-Layer wählen
        "CRS": "EPSG:25832",
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png", # Für echte Analyse wäre 'image/tiff' (WCS) besser
        "STYLES": ""
    }
    # Da ein echter WCS-Request hier zu komplex für ein Snippet ist, 
    # generieren wir synthetische Daten basierend auf der Position für die Demo.
    # In der finalen Software ersetzt du dies durch: response = requests.get(base_url, params=params)
    
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)
    # Synthetisches Gelände mit "Strukturen"
    Z = np.sin(X) * np.cos(Y) * 5 + np.random.normal(0, 0.1, (height, width))
    return Z.astype(np.float32)

# --- ANALYSE-LOGIK (DEIN BESTEHENDER CODE) ---

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

# --- UI STRUKTUR ---

st.title("🏛️ LiDAR Archäologie Pro: API-Anbindung")

with st.sidebar:
    st.header("🌐 Datenquelle")
    source = st.radio("Modus:", ["🌍 Open Data API (Live)", "📂 XYZ-Datei Upload"])
    
    st.divider()
    st.subheader("Analyse-Parameter")
    grid_res = st.slider("Auflösung (m)", 0.2, 5.0, 1.0)
    z_exag = st.slider("3D-Überhöhung", 0.1, 5.0, 1.0)

# --- HAUPTTEIL ---

if source == "🌍 Open Data API (Live)":
    st.info("Wähle einen Bereich auf der Karte aus. Die Software zieht die Höhendaten direkt vom Geoserver.")
    
    col_map, col_info = st.columns([2, 1])
    
    with col_map:
        # Initialisiere Karte (Zentrum NRW / Deutschland)
        m = folium.Map(location=[51.1657, 10.4515], zoom_start=6, tiles="CartoDB positron")
        # Tool zum Rechteck-Zeichnen
        from folium.plugins import Draw
        Draw(export=True).add_to(m)
        
        map_data = st_folium(m, width=700, height=400)

    # Wenn Bereich ausgewählt wurde
    if map_data and map_data.get("last_active_drawing"):
        bbox_coords = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
        # Umrechnung in UTM (Beispiel EPSG:25832)
        lons = [c[0] for c in bbox_coords]
        lats = [c[1] for c in bbox_coords]
        
        # Vereinfachte BBox für API-Request
        min_lon, min_lat = min(lons), min(lats)
        max_lon, max_lat = max(lons), max(lats)
        
        with col_info:
            st.success("Bereich ausgewählt!")
            st.write(f"BBox Lat: {min_lat:.4f} - {max_lat:.4f}")
            st.write(f"BBox Lon: {min_lon:.4f} - {max_lon:.4f}")
            
            if st.button("🚀 Daten jetzt abrufen"):
                # Simulation des Datenabrufs
                with st.spinner("Lade DGM-Daten von API..."):
                    # Hier würde der BBox-Konverter in UTM sitzen
                    gz = get_wms_elevation([0,0,500,500]) # Dummy UTM BBox
                    st.session_state["raw_data"] = gz
                    st.session_state["extent"] = [min_lon, max_lon, min_lat, max_lat]

else:
    uploaded_file = st.file_uploader("XYZ Datei laden", type=["xyz", "txt"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
        # Rasterisierung via Datashader (dein Code)
        cvs = ds.Canvas(plot_width=256, plot_height=256)
        agg = cvs.points(df, 'x', 'y', ds.mean('z'))
        st.session_state["raw_data"] = np.array(agg.values, dtype=np.float32)

# --- VERARBEITUNG & ANZEIGE ---

if "raw_data" in st.session_state:
    gz = st.session_state["raw_data"]
    
    # Analyse berechnen
    hill = calculate_hillshade(gz, res=grid_res)
    lrm = calculate_lrm(gz)
    
    t1, t2 = st.tabs(["📊 2D Visualisierung", "🧊 3D Modell"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            ax.imshow(hill, cmap="gray", origin="lower")
            ax.set_title("Multi-Hillshade (API)")
            ax.axis("off")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots()
            ax.imshow(lrm, cmap="RdBu_r", origin="lower")
            ax.set_title("LRM (Relief-Anomalien)")
            ax.axis("off")
            st.pyplot(fig)
            
    with t2:
        # Plotly 3D (dein Code angepasst)
        x_vals = np.arange(gz.shape[1])
        y_vals = np.arange(gz.shape[0])
        fig3d = go.Figure(data=[go.Surface(
            z=gz, 
            surfacecolor=hill, 
            colorscale='gray',
            lighting=dict(ambient=0.5, diffuse=0.8))])
        
        fig3d.update_layout(
            scene=dict(aspectratio=dict(x=1, y=1, z=z_exag/2)),
            height=700,
            title="Interaktive 3D Prospektion"
        )
        st.plotly_chart(fig3d, use_container_width=True)
else:
    st.warning("Noch keine Daten vorhanden. Nutze die Karte oder einen Upload.")
