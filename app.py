import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io
import base64

# --- HILFSFUNKTIONEN ---

# Versuche pyproj für die Koordinatenumrechnung zu importieren
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

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

def rasterize_points(df, res):
    """Blitzschnelle Rasterisierung von Millionen Punkten mittels Datashader."""
    cvs = ds.Canvas(
        plot_width=int((df.x.max() - df.x.min()) / res),
        plot_height=int((df.y.max() - df.y.min()) / res),
        x_range=(df.x.min(), df.x.max()),
        y_range=(df.y.min(), df.y.max())
    )
    agg = cvs.points(df, 'x', 'y', ds.mean('z'))
    return np.array(agg.values, dtype=np.float32)

# --- ARCHÄOLOGISCHE ANALYSE-ALGORITHMEN ---

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

def calculate_lrm(data, sigma=15):
    """Local Relief Model (LRM) zur Hervorhebung von Gräben und Wällen."""
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p_low, p_high = np.percentile(residual, (2, 98))
    res_clipped = np.clip(residual, p_low, p_high)
    res_min, res_max = res_clipped.min(), res_clipped.max()
    if res_max > res_min:
        return (res_clipped - res_min) / (res_max - res_min)
    return np.full_like(residual, 0.5)

def calculate_openness(data, sigma=3):
    """Einfache Annäherung der 'Openness' zur strukturfokussierten Darstellung."""
    lap = -laplace(gaussian_filter(data, sigma=sigma))
    p_low, p_high = np.percentile(lap, (5, 95))
    norm = np.clip(lap, p_low, p_high)
    return (norm - norm.min()) / (norm.max() - norm.min())

# --- STREAMLIT UI ---

st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚙️ Analyse-Setup")
    uploaded_file = st.file_uploader("XYZ Datei laden", type=["xyz", "txt"])
    
    st.divider()
    st.subheader("Raster-Parameter")
    grid_res = st.slider("Auflösung (m)", 0.2, 5.0, 1.0)
    epsg_code = st.number_input("EPSG Code", value=25832)
    
    st.divider()
    st.subheader("Visualisierung")
    lrm_blend = st.slider("LRM Überlagerung", 0.0, 1.0, 0.3)
    z_exag = st.slider("3D Überhöhung", 0.1, 5.0, 1.0)

if uploaded_file:
    try:
        # 1. Daten laden
        @st.cache_data
        def get_data(file):
            data = pd.read_csv(file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
            if len(data) > 2000000:
                data = data.sample(2000000, random_state=42)
            return data

        df = get_data(uploaded_file)
        min_x, max_x = df.x.min(), df.x.max()
        min_y, max_y = df.y.min(), df.y.max()

        with st.spinner("Modelle werden berechnet..."):
            # 2. Modellierung
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            hill = calculate_hillshade(gz, 315, 45, grid_res)
            lrm = calculate_lrm(gz, 15)
            openness = calculate_openness(gz)
            
            # Fusion für die Hauptansicht
            composite = np.clip(hill * (1 - lrm_blend) + lrm * lrm_blend, 0, 1)

        # Tabs für verschiedene Werkzeuge
        tab1, tab2, tab3 = st.tabs(["🖼️ 2D-Analyse", "📈 Profil-Schnitt", "🌐 3D-Prospektion"])

        with tab1:
            st.subheader("Gelände-Visualisierung")
            col_sel, col_info = st.columns([3, 1])
            
            with col_sel:
                mode = st.selectbox("Darstellungsmodus", ["Klassische Fusion", "Schummerung", "Restrelief (LRM)", "Struktur-Fokus (Openness)"])
                
                display_map = {
                    "Klassische Fusion": composite,
                    "Schummerung": hill,
                    "Restrelief (LRM)": lrm,
                    "Struktur-Fokus (Openness)": openness
                }[mode]
                
                fig, ax = plt.subplots(figsize=(10, 7))
                im = ax.imshow(display_map, cmap="gray" if "LRM" not in mode else "RdBu_r", origin="lower", extent=[min_x, max_x, min_y, max_y])
                ax.set_xlabel("Rechtswert (m)")
                ax.set_ylabel("Hochwert (m)")
                st.pyplot(fig)
            
            with col_info:
                st.info("💡 Nutze LRM für Wälle/Gräben und Openness für feine Texturen.")
                lat, lon = convert_coords(df.x.mean(), df.y.mean(), epsg_code)
                if lat:
                    st.metric("Zentrum Breite", f"{lat:.5f}")
                    st.metric("Zentrum Länge", f"{lon:.5f}")
                    st.write(f"[In Google Maps öffnen](https://www.google.com/maps/search/?api=1&query={lat},{lon})")

        with tab2:
            st.subheader("Interaktiver Profil-Schnitt")
            st.write("Wähle eine Position für einen West-Ost Querschnitt durch das Gelände.")
            
            # Slider für die Y-Position des Schnitts
            y_pos = st.slider("Y-Koordinate (Nord-Süd)", float(min_y), float(max_y), float(df.y.mean()))
            
            # Finde den nächsten Index im Raster
            y_idx = int((y_pos - min_y) / (max_y - min_y) * (gz.shape[0] - 1))
            y_idx = np.clip(y_idx, 0, gz.shape[0]-1)
            
            profile_z = gz[y_idx, :]
            profile_x = np.linspace(min_x, max_x, len(profile_z))
            
            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(x=profile_x, y=profile_z, mode='lines', name='Gelände', fill='tozeroy', line_color='teal'))
            fig_prof.update_layout(
                xaxis_title="Meter (West-Ost)",
                yaxis_title="Höhe über NN (m)",
                hovermode="x unified",
                height=400
            )
            st.plotly_chart(fig_prof, use_container_width=True)
            st.caption("Dieses Profil hilft, die Tiefe von Gräben oder die Höhe von Strukturen präzise zu vermessen.")

        with tab3:
            st.subheader("High-Performance 3D Viewer")
            # Downsampling für Performance
            step = max(1, int(np.sqrt(gz.size / 400000)))
            z_plot = gz[::step, ::step]
            tex_plot = composite[::step, ::step]
            
            x_vals = np.linspace(min_x, max_x, z_plot.shape[1])
            y_vals = np.linspace(min_y, max_y, z_plot.shape[0])

            fig3d = go.Figure(data=[go.Surface(
                x=x_vals, y=y_vals, z=z_plot, 
                surfacecolor=tex_plot, 
                colorscale='gray',
                lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5)
            )])
            
            fig3d.update_layout(
                scene=dict(
                    aspectmode='data',
                    aspectratio=dict(x=1, y=1, z=z_exag),
                    xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Höhe"
                ),
                height=800, margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler bei der Verarbeitung: {e}")
else:
    st.info("Bitte laden Sie eine XYZ-Datei hoch (z.B. eine Punktwolke aus LiDAR-Scans).")
