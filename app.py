import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter, uniform_filter
import datashader as ds
import io
import json
import requests
from datetime import datetime

# Firebase / Firestore für die Cloud-Speicherung
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    pass

# --- GEMINI API SETUP ---
def call_gemini_analyst(prompt):
    """KI-Analyst zur Interpretation der LiDAR-Daten."""
    api_key = "" # Der API-Key wird von der Umgebung bereitgestellt
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt}
            ]
        }],
        "systemInstruction": {
            "parts": [{"text": "Du bist ein Experte für digitale Archäologie und LiDAR-Prospektion. Analysiere die gegebenen Daten auf anthropogene Strukturen wie Wälle, Gräben, Siedlungsspuren oder Altwege."}]
        }
    }
    
    # Exponential Backoff für API-Calls
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Keine Analyse möglich.")
        except Exception:
            import time
            time.sleep(2**attempt)
    return "KI-Analyse derzeit nicht verfügbar. Bitte versuchen Sie es später erneut."

# --- FIRESTORE SETUP ---
def init_firestore():
    if not firebase_admin._apps:
        try:
            # Versuche Konfiguration aus Streamlit Secrets zu laden
            conf = json.loads(st.secrets["__firebase_config"])
            firebase_admin.initialize_app(credentials.Certificate(conf))
        except:
            return None
    return firestore.client()

# --- ANALYSE-FUNKTIONEN ---

def calculate_svf(data, radius=10):
    """Approximation des Sky-View-Factors (Himmelsichtfaktor)."""
    # Vereinfachte Version: Differenz zum lokalen Mittelwert hebt Vertiefungen/Erhebungen hervor
    mean_val = uniform_filter(data, size=radius*2)
    svf = data - mean_val
    # Normalisierung für die Darstellung
    svf_norm = (svf - np.min(svf)) / (np.max(svf) - np.min(svf) + 1e-8)
    return np.clip(svf_norm, 0, 1)

def calculate_rrim(data, res=1.0):
    """Red Relief Image Map (Kombination aus Wölbung und Hangneigung)."""
    # Hangneigung (Slope)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    # Krümmung (Curvature/Laplace)
    curv = -laplace(data)
    
    # Normalisierung der Kanäle
    slope_n = (slope - np.min(slope)) / (np.max(slope) - np.min(slope) + 1e-8)
    curv_n = (curv - np.min(curv)) / (np.max(curv) - np.min(curv) + 1e-8)
    
    # Erstellung des RGB-Bildes (RRIM Pattern)
    rrim = np.zeros((*data.shape, 3))
    rrim[..., 0] = slope_n  # Rot: Steilheit
    rrim[..., 1] = curv_n   # Grün: Positive Wölbung (Wälle)
    rrim[..., 2] = 1.0 - curv_n # Blau: Negative Wölbung (Gräben)
    return rrim

# --- APP LAYOUT ---
st.set_page_config(page_title="LiDAR Archäologie Pro Ultra", layout="wide")

if "markers" not in st.session_state:
    st.session_state.markers = []

with st.sidebar:
    st.title("🏛️ LiDAR Pro Ultra")
    uploaded_file = st.file_uploader("XYZ Daten laden (.xyz, .txt, .csv)", type=["xyz", "txt", "csv"])
    
    st.divider()
    grid_res = st.slider("Raster-Auflösung (m)", 0.2, 5.0, 1.0)
    z_exag = st.slider("3D-Überhöhung", 0.1, 10.0, 2.0)
    
    st.divider()
    st.subheader("Cloud- & KI-Tools")
    save_markers = st.toggle("Cloud-Synchronisierung", False)
    
    if st.button("✨ KI-Experten-Analyse starten"):
        with st.spinner("KI wertet Geländedaten aus..."):
            st.session_state.ai_analysis = call_gemini_analyst(
                "Ich habe Geländedaten mit archäologischen Strukturen. Analysiere das aktuelle Modell auf Anomalien wie Viereckschanzen, Hügelgräber oder Hohlwege."
            )

# --- HAUPT-LOGIK ---
if uploaded_file:
    @st.cache_data
    def load_and_rasterize(file, res):
        # CSV Laden
        df = pd.read_csv(file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
        
        # Datashader für schnelles Rastering von Millionen Punkten
        cvs = ds.Canvas(
            plot_width=int((df.x.max() - df.x.min()) / res),
            plot_height=int((df.y.max() - df.y.min()) / res),
            x_range=(df.x.min(), df.x.max()),
            y_range=(df.y.min(), df.y.max())
        )
        agg = cvs.points(df, 'x', 'y', ds.mean('z'))
        return np.array(agg.values, dtype=np.float32), (df.x.min(), df.x.max(), df.y.min(), df.y.max())

    try:
        with st.spinner("Verarbeite LiDAR-Punktwolke..."):
            gz, extent = load_and_rasterize(uploaded_file, grid_res)
            # Fülle ungültige Werte mit dem Mittelwert
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))

        # Analysen berechnen
        svf = calculate_svf(gz)
        rrim = calculate_rrim(gz, grid_res)
        lrm = gz - gaussian_filter(gz, sigma=10) # Local Relief Model

        t1, t2, t3, t4 = st.tabs(["🗺️ Karten-Analyse", "📐 Profil-Schnitt", "📊 Fundstellen-Log", "🤖 KI-Expertise"])

        with t1:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                map_type = st.radio("Visualisierung", ["Red Relief (RRIM)", "Sky View Factor", "Restrelief (LRM)", "Höhenmodell (Grau)"])
                st.info("💡 Pro-Tipp: RRIM eignet sich am besten für die Entdeckung von Mauern und Gräben unter Vegetation.")
                
            with col1:
                if map_type == "Red Relief (RRIM)":
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(rrim, origin='lower', extent=extent)
                    ax.set_title("Red Relief Image Map")
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    z_data = svf if map_type == "Sky View Factor" else (lrm if map_type == "Restrelief (LRM)" else gz)
                    fig = go.Figure(data=go.Heatmap(
                        z=z_data,
                        colorscale='Greys' if map_type != "Restrelief (LRM)" else 'RdBu',
                        x0=extent[0], dx=grid_res, y0=extent[2], dy=grid_res
                    ))
                    fig.update_layout(width=800, height=700, margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig, use_container_width=True)

        with t2:
            st.subheader("Interaktives Geländeprofil")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                ax_val = st.number_input("Start X", value=float(extent[0]))
                ay_val = st.number_input("Start Y", value=float(extent[2]))
            with p_col2:
                bx_val = st.number_input("Ende X", value=float(extent[1]))
                by_val = st.number_input("Ende Y", value=float(extent[3]))
                
            # Extraktion des Profils
            num_pts = 200
            x_line = np.linspace(ax_val, bx_val, num_pts)
            y_line = np.linspace(ay_val, by_val, num_pts)
            
            # Umrechnung in Raster-Indizes
            ix = ((x_line - extent[0]) / grid_res).astype(int)
            iy = ((y_line - extent[2]) / grid_res).astype(int)
            ix = np.clip(ix, 0, gz.shape[1]-1)
            iy = np.clip(iy, 0, gz.shape[0]-1)
            
            profile_z = gz[iy, ix]
            dist = np.sqrt((x_line - ax_val)**2 + (y_line - ay_val)**2)
            
            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(x=dist, y=profile_z, mode='lines', name='Geländehöhe', fill='tozeroy'))
            fig_prof.update_layout(title="Höhenprofil-Schnitt (m)", xaxis_title="Distanz (m)", yaxis_title="Höhe ü. NN (m)")
            st.plotly_chart(fig_prof, use_container_width=True)

        with t3:
            st.subheader("📍 Fundstellen-Management")
            with st.form("marker_form", clear_on_submit=True):
                m_name = st.text_input("Name der Anomalie", placeholder="z.B. Grabhügel_01")
                m_type = st.selectbox("Typ", ["Hügelgrab", "Wall/Graben", "Siedlung", "Hohlweg", "Sonstiges"])
                m_coords = st.text_input("Koordinaten", f"{ax_val:.2f}, {ay_val:.2f}")
                
                if st.form_submit_button("Fundstelle speichern"):
                    new_marker = {
                        "Name": m_name,
                        "Typ": m_type,
                        "Position": m_coords,
                        "Zeitstempel": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    st.session_state.markers.append(new_marker)
                    st.success(f"'{m_name}' wurde zum Log hinzugefügt.")
            
            if st.session_state.markers:
                st.table(pd.DataFrame(st.session_state.markers))
            else:
                st.info("Noch keine Fundstellen dokumentiert.")

        with t4:
            st.subheader("🤖 KI-Bericht")
            if "ai_analysis" in st.session_state:
                st.markdown(st.session_state.ai_analysis)
            else:
                st.info("Klicken Sie in der Sidebar auf 'KI-Experten-Analyse starten', um einen automatisierten Bericht zu generieren.")

    except Exception as e:
        st.error(f"Fehler bei der Datenverarbeitung: {e}")

else:
    st.info("Willkommen! Bitte laden Sie eine LiDAR-XYZ-Datei hoch, um mit der archäologischen Prospektion zu beginnen.")
    
    # Demo-Daten Bereich
    st.divider()
    if st.button("Beispiel-Gelände generieren"):
        # Erzeuge synthetisches Gelände mit archäologischen Features
        size = 150
        x = np.linspace(0, 100, size)
        y = np.linspace(0, 100, size)
        X, Y = np.meshgrid(x, y)
        # Natürliches Gelände
        Z = np.sin(X/15) * np.cos(Y/15) * 5 
        # Künstliche Viereckschanze
        Z[60:90, 60:65] += 2 # Wall West
        Z[60:90, 85:90] += 2 # Wall Ost
        Z[60:65, 60:90] += 2 # Wall Nord
        Z[85:90, 60:90] += 2 # Wall Süd
        
        output = []
        for i in range(size):
            for j in range(size):
                output.append(f"{X[i,j]},{Y[i,j]},{Z[i,j]}")
        
        st.download_button(
            label="Demo XYZ Datei herunterladen",
            data="\n".join(output),
            file_name="archaeo_demo.xyz",
            mime="text/csv"
        )
