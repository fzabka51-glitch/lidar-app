import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io
import base64
import requests
import time
import os

# --- API KONFIGURATION ---
# WICHTIG: apiKey MUSS ein leerer String sein. 
# Das System injiziert den Schlüssel zur Laufzeit automatisch.
apiKey = ""

def call_gemini_vision(base64_image, analysis_type):
    """
    Sendet das Bild an Gemini zur archäologischen Analyse.
    Verwendet gemini-2.5-flash-preview-09-2025 für Image Understanding.
    """
    # Wir verwenden apiKey direkt ohne manuelle Validierung,
    # da die Umgebung diesen Wert zur Laufzeit füllt.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    
    prompt_text = f"""Analysiere dieses LiDAR-Geländemodell ({analysis_type}).
    Suche nach anthropogenen (menschengemachten) Strukturen wie:
    1. Grabhügel (kreisförmige Erhebungen)
    2. Hohlwege (lineare Vertiefungen)
    3. Wallanlagen oder Gräben
    4. Siedlungsreste (rechteckige Strukturen)
    5. Landwirtschaftliche Spuren (Wölbäcker)

    Beschreibe auffällige Merkmale und gib eine fachliche Einschätzung ab (Deutsch)."""

    # Payload gemäß technischer Spezifikation
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt_text},
                {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64_image
                    }
                }
            ]
        }]
    }

    # Exponential Backoff Implementierung (1s, 2s, 4s, 8s, 16s)
    last_error = "Keine Antwort erhalten."
    for delay in [1, 2, 4, 8, 16]:
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Keine Antwort vom Modell.")
            elif response.status_code == 429:
                time.sleep(delay)
                continue
            else:
                last_error = f"Status {response.status_code}: {response.text}"
                time.sleep(delay)
        except Exception as e:
            last_error = str(e)
            time.sleep(delay)
    
    return f"KI-Analyse fehlgeschlagen nach mehreren Versuchen. Fehlerdetails: {last_error}"

# --- HILFSFUNKTIONEN ---

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

def convert_coords(x, y, from_epsg=25832):
    if not PYPROJ_AVAILABLE: return None, None
    try:
        transformer = pyproj.Transformer.from_crs(f"epsg:{from_epsg}", "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except: return None, None

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
    az_rad, al_rad = np.deg2rad(azimuth), np.deg2rad(angle_altitude)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(al_rad) * np.cos(slope)) + (np.sin(al_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)

def calculate_lrm(data, sigma=15):
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p5, p95 = np.percentile(residual, (5, 95))
    res_clipped = np.clip(residual, p5, p95)
    rmin, rmax = res_clipped.min(), res_clipped.max()
    return (res_clipped - rmin) / (rmax - rmin) if rmax > rmin else np.full_like(residual, 0.5)

# --- APP LAYOUT ---

st.set_page_config(page_title="LiDAR AI Pro", layout="wide")
st.title("🏛️ LiDAR Analyse & AI Prospektion")

with st.sidebar:
    st.header("⚙️ Konfiguration")
    uploaded_file = st.file_uploader("XYZ Datei hochladen", type=["xyz", "txt"])
    st.divider()
    epsg_code = st.number_input("EPSG Code (z.B. 25832)", value=25832)
    grid_res = st.slider("Raster-Auflösung (m)", 0.2, 5.0, 1.0)
    lrm_sigma = st.slider("LRM Glättung", 5, 50, 15)
    z_exag = st.slider("3D Überhöhung", 0.1, 5.0, 1.0)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
        if len(df) > 1500000:
            df = df.sample(1500000, random_state=42)
            st.warning("Datensatz auf 1.5 Mio. Punkte reduziert.")

        with st.spinner("Modelle werden berechnet..."):
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            hill = calculate_hillshade(gz, 315, 45, grid_res)
            lrm = calculate_lrm(gz, lrm_sigma)
            # Einfache Fusion für bessere Sichtbarkeit
            fusion = np.clip(hill * 0.7 + lrm * 0.3, 0, 1)

            models = {
                "Schummerung (Hillshade)": (hill, "gray"),
                "Restrelief (LRM)": (lrm, "RdBu_r"),
                "Fusion (Empfohlen für KI)": (fusion, "gray")
            }

        t1, t2, t3 = st.tabs(["🖼️ 2D Analyse", "🤖 KI Assistent", "🌐 3D Prospektion"])

        with t1:
            sel = st.selectbox("Modell wählen", list(models.keys()), index=2)
            data, cmap = models[sel]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(data, cmap=cmap, origin='lower')
            ax.axis('off')
            st.pyplot(fig)
            st.session_state['current_img'] = fig
            st.session_state['current_name'] = sel

        with t2:
            st.subheader("KI-Struktur-Erkennung")
            st.write("Die KI analysiert das aktuell gewählte 2D-Modell.")
            if st.button("🚀 Analyse starten"):
                if 'current_img' in st.session_state:
                    with st.spinner("KI analysiert das Gelände..."):
                        buf = io.BytesIO()
                        st.session_state['current_img'].savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                        plt.close(st.session_state['current_img'])
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        
                        report = call_gemini_vision(b64, st.session_state['current_name'])
                        st.info(report)
                else:
                    st.warning("Bitte wählen Sie zuerst ein Modell im 2D-Tab.")

        with t3:
            st.subheader("Interaktive 3D-Ansicht")
            step = max(1, int(np.sqrt(gz.size / 400000)))
            z_plot = gz[::step, ::step]
            tex_plot = fusion[::step, ::step]
            
            fig3d = go.Figure(data=[go.Surface(
                z=z_plot, surfacecolor=tex_plot, colorscale='gray',
                lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5)
            )])
            fig3d.update_layout(scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=z_exag)), height=700)
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte laden Sie eine LiDAR-Datei (.xyz) hoch.")
