import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io

# <context>
# Überprüfung der Geodaten-Abhängigkeiten
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
# </context>

# --- TRIPLE-PROTOCOL DEFINITION ---
# Persona: Senior AI Orchestrator & Geospatial Engineer
# Goal: Bereitstellung eines lückenlosen, wissenschaftlich validen LiDAR-Analyse-Tools.
# Anti-Goal: Kein Weglassen von Code-Fragmenten; kein Verzicht auf mathematische Präzision (Filter-Radien).

# --- CORE-ENGINE: ARCHÄOLOGISCHE ALGORITHMEN ---

def rasterize_points(df, res):
    """Hochperformante Rasterisierung mittels Datashader."""
    cvs = ds.Canvas(
        plot_width=int((df.x.max() - df.x.min()) / res),
        plot_height=int((df.y.max() - df.y.min()) / res),
        x_range=(df.x.min(), df.x.max()),
        y_range=(df.y.min(), df.y.max())
    )
    agg = cvs.points(df, 'x', 'y', ds.mean('z'))
    return np.array(agg.values, dtype=np.float32)

def calculate_lrm(data, sigma=15):
    """
    Local Relief Model (LRM): 
    Extrahiert Mikro-Relief durch Subtraktion eines geglätteten DGM.
    """
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p_low, p_high = np.percentile(residual, (2, 98))
    res_clipped = np.clip(residual, p_low, p_high)
    return (res_clipped - res_clipped.min()) / (res_clipped.max() - res_clipped.min() + 1e-6)

def calculate_openness(data, op_type="positive"):
    """
    Sky-View Factor / Openness Approximation:
    Visualisiert Konvexitäten (Wälle) oder Konkavitäten (Gräben).
    """
    val = laplace(gaussian_filter(data, sigma=1.0 if op_type == "positive" else 2.0))
    if op_type == "negative": val = -val
    p_low, p_high = np.percentile(val, (2, 98))
    val = np.clip(val, p_low, p_high)
    return (val - val.min()) / (val.max() - val.min() + 1e-6)

def calculate_hillshade(data, azimuth=315, angle_altitude=45, res=1.0):
    """Klassische Schummerung für die Übersicht."""
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(altitude_rad) * np.cos(slope)) + \
            (np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)

def detect_structures(data, threshold=2.2):
    """
    Statistische Anomalien-Detektion (Z-Score):
    Markiert unnatürliche Geländeabweichungen.
    """
    local_mean = gaussian_filter(data, sigma=3)
    local_std = np.sqrt(gaussian_filter((data - local_mean)**2, sigma=3))
    z_score = np.abs((data - local_mean) / (local_std + 1e-6))
    return np.where(z_score > threshold, 1.0, 0.0)

# --- UI & WORKFLOW ---

st.set_page_config(page_title="LiDAR Archaeology Pro", layout="wide")
st.title("🏛️ LiDAR Archaeology Suite Pro")

with st.sidebar:
    st.header("⚙️ Konfiguration")
    uploaded_file = st.file_uploader("XYZ Datei hochladen", type=["xyz", "txt"])
    
    <instructions>
    # Parameter für archäologische Feinanalyse
    grid_res = st.number_input("Raster-Auflösung (m)", 0.1, 5.0, 0.5)
    lrm_sigma = st.slider("LRM Filterstärke", 5, 50, 20)
    anom_sens = st.slider("Anomalien-Sensitivität", 1.0, 5.0, 2.5)
    z_exag = st.slider("3D Überhöhung", 1.0, 10.0, 3.0)
    </instructions>

if uploaded_file:
    try:
        # Daten laden
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
        
        # Berechnung (CoT: Schritt-für-Schritt Prozessierung)
        with st.spinner("Analysiere Mikro-Relief..."):
            # 1. Rasterisierung
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            # 2. Archäologische Filter (Keine Auslassungen!)
            lrm = calculate_lrm(gz, lrm_sigma)
            pos_open = calculate_openness(gz, "positive")
            neg_open = calculate_openness(gz, "negative")
            hill = calculate_hillshade(gz, 315, 45, grid_res)
            anom = detect_structures(lrm, anom_sens)
            
            # 3. Composite (Visualisierungs-Verschmelzung)
            arch_comp = np.clip(pos_open * 0.4 + neg_open * 0.4 + lrm * 0.2, 0, 1)

        # Darstellung
        tab1, tab2 = st.tabs(["🖼️ 2D-Analyse", "🌐 3D-Befundung"])

        with tab1:
            models = {
                "Archäologisches Kombimodell": (arch_comp, "bone"),
                "LRM (Mikro-Strukturen)": (lrm, "RdBu_r"),
                "Negative Openness (Gräben)": (neg_open, "inferno"),
                "Anomalien-Detektion": (anom, "YlOrRd"),
                "Klassischer Hillshade": (hill, "gray")
            }
            
            cols = st.columns(2)
            for i, (name, (data, cmap)) in enumerate(models.items()):
                with cols[i % 2]:
                    fig, ax = plt.subplots()
                    ax.imshow(data, cmap=cmap, origin='lower')
                    ax.set_title(name)
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)

        with tab2:
            step = max(1, int(np.sqrt(gz.size / 300000)))
            fig3d = go.Figure(data=[go.Surface(
                z=gz[::step, ::step], 
                surfacecolor=arch_comp[::step, ::step], 
                colorscale="bone"
            )])
            fig3d.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=z_exag/5)))
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler bei der Implementierung: {e}")
else:
    st.info("Bitte laden Sie eine XYZ-Datei hoch, um die archäologische Prospektion zu starten.")

# <summary>
# Dieser Code integriert alle wissenschaftlichen Visualisierungsmethoden (LRM, Openness, Z-Score Detektion) 
# ohne Verluste aus vorherigen Iterationen.
# </summary>
