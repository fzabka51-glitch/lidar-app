import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io
import base64
import json
import requests
import time

# --- API KONFIGURATION ---
apiKey = "" # Wird von der Umgebung automatisch gefüllt

def call_gemini_vision(base64_image, analysis_type):
    """Sendet das Bild an Gemini zur archäologischen Analyse."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    
    prompt = f"""
    Du bist ein Experte für LiDAR-Archäologie. Analysiere dieses LiDAR-Geländemodell ({analysis_type}).
    Suche nach anthropogenen (menschengemachten) Strukturen wie:
    1. Grabhügel (kleine, kreisförmige Erhebungen)
    2. Hohlwege (lineare, tief eingeschnittene Pfade)
    3. Wallanlagen oder Gräben (geometrische Strukturen)
    4. Siedlungsreste (rechteckige Grundrisse)
    5. Landwirtschaftliche Spuren (Wölbäcker oder alte Flurgrenzen)

    Beschreibe auffällige Merkmale und gib eine Einschätzung ab, ob es sich um natürliche Geologie oder potenzielle Archäologie handelt.
    Antworte auf Deutsch, präzise und fachlich fundiert.
    """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64_image
                    }
                }
            ]
        }]
    }

    # Exponential Backoff Implementierung
    for delay in [1, 2, 4, 8, 16]:
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Keine Analyse möglich.")
            elif response.status_code == 429: # Rate Limit
                time.sleep(delay)
                continue
            else:
                return f"Fehler: {response.status_code} - {response.text}"
        except Exception as e:
            time.sleep(delay)
            last_err = str(e)
    return f"API-Verbindungsfehler nach mehreren Versuchen: {last_err}"

# Versuche pyproj für die Koordinatenumrechnung zu importieren
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")
st.title("🏛️ LiDAR Analyse & AI Prospektion")

# --- KOORDINATEN-FUNKTION ---
def convert_coords(x, y, from_epsg=25832):
    if not PYPROJ_AVAILABLE:
        return None, None
    try:
        transformer = pyproj.Transformer.from_crs(f"epsg:{from_epsg}", "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except:
        return None, None

# --- ANALYSE-FUNKTIONEN ---
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

def calculate_multi_hillshade(data, res=1.0):
    h1 = calculate_hillshade(data, 315, 45, res)
    h2 = calculate_hillshade(data, 45, 45, res)
    h3 = calculate_hillshade(data, 135, 45, res)
    h4 = calculate_hillshade(data, 225, 45, res)
    return (h1 + h2 + h3 + h4) / 4.0

def calculate_lrm(data, sigma=15):
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p_low, p_high = np.percentile(residual, (5, 95))
    res_clipped = np.clip(residual, p_low, p_high)
    res_min, res_max = res_clipped.min(), res_clipped.max()
    if res_max > res_min:
        return (res_clipped - res_min) / (res_max - res_min)
    return np.full_like(residual, 0.5)

def calculate_slope(data, res=1.0):
    gy, gx = np.gradient(data, res, res)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))
    p_high = np.nanpercentile(slope_deg, 98)
    return np.clip(slope_deg, 0, p_high)

def calculate_curvature(data):
    curv = -laplace(data)
    p_low, p_high = np.percentile(curv, (2, 98))
    curv_clipped = np.clip(curv, p_low, p_high)
    c_min, c_max = curv_clipped.min(), curv_clipped.max()
    if c_max > c_min:
        return (curv_clipped - c_min) / (c_max - c_min)
    return np.full_like(curv, 0.5)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Parameter")
    uploaded_file = st.file_uploader("XYZ Datei laden (.xyz, .txt)", type=["xyz", "txt"])
    
    st.divider()
    st.subheader("Geo-Referenz")
    epsg_code = st.number_input("EPSG Code (UTM 32N: 25832)", value=25832)
    
    st.divider()
    st.subheader("Raster & Filter")
    grid_res = st.number_input("Auflösung (m)", 0.1, 10.0, 1.0)
    lrm_sigma = st.slider("LRM Glättung (Sigma)", 1, 50, 15)
    
    st.subheader("3D-Eigenschaften")
    z_exag = st.slider("Z-Überhöhung", 0.1, 5.0, 0.5, step=0.1)
    
    view_mode = st.radio("Ansicht 2D:", ["Einzelansicht", "Gitter-Übersicht"])

# --- HAUPTBEREICH ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
        if len(df) > 2000000:
            df = df.sample(2000000, random_state=42)
            st.warning("⚠️ Datensatz auf 2 Mio. Punkte reduziert.")

        min_x, max_x = df.x.min(), df.x.max()
        min_y, max_y = df.y.min(), df.y.max()
        center_x, center_y = df.x.mean(), df.y.mean()
        lat, lon = convert_coords(center_x, center_y, epsg_code)
        
        location_placeholder = st.empty()
        if lat and lon:
            google_maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            location_placeholder.markdown(f"**📍 Standort:** [{lat:.5f}, {lon:.5f}]({google_maps_url})")

        with st.spinner("Analysiere Gelände..."):
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            nw_h = calculate_hillshade(gz, 315, 45, grid_res)
            mds = calculate_multi_hillshade(gz, grid_res)
            lrm = calculate_lrm(gz, lrm_sigma)
            slope = calculate_slope(gz, grid_res)
            curv = calculate_curvature(gz)
            comp = np.clip(mds + (lrm - 0.5) * 0.3, 0, 1)

            analysis_models = {
                "Final Composite (Fusion)": (comp, "gray", False),
                "NW Hillshade": (nw_h, "gray", False),
                "MDS Composite": (mds, "gray", False),
                "Restrelief (LRM)": (lrm, "RdBu", True),
                "Hangneigung (Slope)": (slope, "plasma", True),
                "Krümmung (Curvature)": (curv, "RdYlGn", True)
            }

        tab1, tab2, tab3 = st.tabs(["🖼️ 2D-Analyse", "🤖 AI-Assistent", "🌐 3D-Prospektion"])

        with tab1:
            if view_mode == "Gitter-Übersicht":
                c1, c2 = st.columns(2)
                for i, (name, (data, cmap, _)) in enumerate(analysis_models.items()):
                    with [c1, c2][i % 2]:
                        fig, ax = plt.subplots()
                        ax.imshow(data, cmap=cmap, interpolation='none', origin='lower')
                        ax.set_title(name)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                sel_2d = st.selectbox("Modell wählen:", list(analysis_models.keys()))
                data, cmap, _ = analysis_models[sel_2d]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(data, cmap=cmap, interpolation='none', origin='lower')
                ax.axis('off')
                st.pyplot(fig)
                st.session_state['current_fig'] = fig
                st.session_state['current_model_name'] = sel_2d

        with tab2:
            st.subheader("🤖 KI-Struktur-Erkennung")
            st.write("Lassen Sie die Karte von einer KI auf archäologische Merkmale prüfen.")
            
            if 'current_fig' in st.session_state:
                if st.button("🗺️ Aktuelle Ansicht analysieren"):
                    with st.spinner("KI studiert die Karte..."):
                        # Bild konvertieren
                        buf = io.BytesIO()
                        st.session_state['current_fig'].savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        
                        # API Aufruf
                        report = call_gemini_vision(img_base64, st.session_state['current_model_name'])
                        
                        st.markdown("### 📜 Archäologischer Vorbericht")
                        st.write(report)
            else:
                st.info("Bitte wähle zuerst ein Modell im 2D-Tab (Einzelansicht) aus.")

        with tab3:
            selected_texture = st.selectbox("Textur für 3D:", list(analysis_models.keys()))
            tex_data, tex_cmap, show_scale = analysis_models[selected_texture]
            step = max(1, int(np.sqrt(gz.size / 400000)))
            z_plot = gz[::step, ::step]
            surface_tex = tex_data[::step, ::step]
            x_vals = np.linspace(min_x, max_x, z_plot.shape[1])
            y_vals = np.linspace(min_y, max_y, z_plot.shape[0])

            fig3d = go.Figure(data=[go.Surface(
                x=x_vals, y=y_vals, z=z_plot, 
                surfacecolor=surface_tex, colorscale=tex_cmap, showscale=show_scale,
                lighting=dict(ambient=0.6, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=0.5)
            )])
            fig3d.update_layout(scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=z_exag)), height=800)
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte XYZ-Datei hochladen.")
