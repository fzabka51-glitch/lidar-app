import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter, generic_filter
import datashader as ds
import io
import base64

# Versuche pyproj für die Koordinatenumrechnung zu importieren
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro+", layout="wide", initial_sidebar_state="expanded")

# Custom CSS für besseres Design
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 4px 4px 0px 0px; gap: 1px; }
    .stTabs [aria-selected="true"] { background-color: #e1e4e8; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- HILFSFUNKTIONEN ---

def convert_coords(x, y, from_epsg=25832):
    if not PYPROJ_AVAILABLE:
        return None, None
    try:
        transformer = pyproj.Transformer.from_crs(f"epsg:{from_epsg}", "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lat, lon
    except:
        return None, None

def get_image_download_link(fig, filename="analysis.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    img_str = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">📩 Bild speichern</a>'
    return href

# --- ARCHÄOLOGISCHE ANALYSE-FUNKTIONEN ---

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

def calculate_svf_approx(data, radius=5):
    """Vereinfachte Annäherung des Sky View Factors via Lokaler Offenheit."""
    def openness(x):
        center = x[len(x)//2]
        return np.mean(np.arctan((x - center) / radius))
    
    # Nutze Laplace als Proxy für schnelle Visualisierung von Kanten/Strukturen
    # SVF ist rechenintensiv, daher hier eine Kombination aus Weichzeichnung und Differenz
    smoothed = gaussian_filter(data, sigma=radius)
    diff = data - smoothed
    return np.clip((diff - diff.min()) / (diff.max() - diff.min()), 0, 1)

def calculate_lrm(data, sigma=15):
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p_low, p_high = np.percentile(residual, (5, 95))
    res_clipped = np.clip(residual, p_low, p_high)
    res_min, res_max = res_clipped.min(), res_clipped.max()
    if res_max > res_min:
        return (res_clipped - res_min) / (res_max - res_min)
    return np.full_like(residual, 0.5)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏛️ LiDAR Pro+")
    uploaded_file = st.file_uploader("Daten (.xyz, .txt)", type=["xyz", "txt"])
    
    st.divider()
    with st.expander("🌍 Geo-Referenz & Grid", expanded=True):
        epsg_code = st.number_input("EPSG Code", value=25832)
        grid_res = st.slider("Auflösung (m)", 0.2, 5.0, 1.0)
    
    with st.expander("🔍 Analyse-Optionen"):
        lrm_sigma = st.slider("LRM Glättung", 5, 50, 15)
        z_exag = st.slider("3D Überhöhung", 0.1, 5.0, 1.5)
        
    st.info("Hinweis: Große Dateien werden automatisch reduziert, um die Performance im Browser zu erhalten.")

# --- HAUPTBEREICH ---
if uploaded_file:
    try:
        # 1. Daten laden (Caching für Speed)
        @st.cache_data
        def load_data(file):
            data = pd.read_csv(file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
            if len(data) > 1500000:
                data = data.sample(1500000, random_state=42)
            return data

        df = load_data(uploaded_file)
        
        # Grid Info
        min_x, max_x = df.x.min(), df.x.max()
        min_y, max_y = df.y.min(), df.y.max()
        center_x, center_y = df.x.mean(), df.y.mean()

        with st.spinner("Berechne Modelle..."):
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            # Modelle
            hillshade = calculate_hillshade(gz, 315, 45, grid_res)
            lrm = calculate_lrm(gz, lrm_sigma)
            svf = calculate_svf_approx(gz)
            slope = np.rad2deg(np.arctan(np.sqrt(np.square(np.gradient(gz, grid_res)[0]) + np.square(np.gradient(gz, grid_res)[1]))))
            
            # Composite (Der klassische Archäologie-Look)
            composite = np.clip(hillshade * 0.7 + svf * 0.3, 0, 1)

            models = {
                "Übersicht (Composite)": (composite, "gray"),
                "Schummerung (Hillshade)": (hillshade, "gray"),
                "Restrelief (LRM)": (lrm, "RdBu_r"),
                "Struktur-Analyse (SVF-Proxy)": (svf, "bone"),
                "Hangneigung": (slope, "inferno")
            }

        # Layout Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 2D Analyse", "📈 Profil-Schnitt", "🌐 3D Viewer", "📊 Metriken"])

        with tab1:
            st.subheader("Archäologische Gelände-Visualisierung")
            sel_model = st.selectbox("Darstellungs-Modell:", list(models.keys()))
            data, cmap = models[sel_model]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(data, cmap=cmap, origin='lower', extent=[min_x, max_x, min_y, max_y])
            plt.colorbar(im, ax=ax, shrink=0.6)
            ax.set_title(sel_model)
            st.pyplot(fig)
            st.markdown(get_image_download_link(fig, f"{sel_model}.png"), unsafe_allow_html=True)

        with tab2:
            st.subheader("Interaktiver Profil-Schnitt")
            st.write("Lege einen Schnitt durch das Gelände, um Strukturen zu vermessen.")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                p_orient = st.radio("Schnitt-Richtung", ["Horizontal (West-Ost)", "Vertikal (Süd-Nord)"])
                if p_orient == "Horizontal (West-Ost)":
                    slice_pos = st.slider("Y-Position wählen", float(min_y), float(max_y), float(center_y))
                    # Finde Index
                    idx = int((slice_pos - min_y) / (max_y - min_y) * (gz.shape[0]-1))
                    profile_z = gz[idx, :]
                    profile_x = np.linspace(min_x, max_x, len(profile_z))
                else:
                    slice_pos = st.slider("X-Position wählen", float(min_x), float(max_x), float(center_x))
                    idx = int((slice_pos - min_x) / (max_x - min_x) * (gz.shape[1]-1))
                    profile_z = gz[:, idx]
                    profile_x = np.linspace(min_y, max_y, len(profile_z))

            with col2:
                fig_prof = go.Figure()
                fig_prof.add_trace(go.Scatter(x=profile_x, y=profile_z, mode='lines', line=dict(color='firebrick', width=2)))
                fig_prof.update_layout(title=f"Geländeprofil an Position {slice_pos:.2f}", xaxis_title="Meter", yaxis_title="Höhe (m)", height=400)
                st.plotly_chart(fig_prof, use_container_width=True)

        with tab3:
            st.subheader("3D Prospektion")
            # Downsampling für Performance
            step = max(1, int(gz.shape[0] / 300))
            z_3d = gz[::step, ::step]
            tex_3d = composite[::step, ::step]
            
            x_range = np.linspace(min_x, max_x, z_3d.shape[1])
            y_range = np.linspace(min_y, max_y, z_3d.shape[0])

            fig3d = go.Figure(data=[go.Surface(
                z=z_3d, x=x_range, y=y_range,
                surfacecolor=tex_3d,
                colorscale='gray'
            )])
            fig3d.update_layout(
                scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=z_exag/2)),
                height=700, margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)

        with tab4:
            st.subheader("Gelände-Statistik")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Min Höhe", f"{gz.min():.2f} m")
            c2.metric("Max Höhe", f"{gz.max():.2f} m")
            c3.metric("Ø Hangneigung", f"{slope.mean():.1f} °")
            c4.metric("Fläche", f"{(max_x-min_x)*(max_y-min_y)/10000:.1f} ha")
            
            # Koordinaten
            lat, lon = convert_coords(center_x, center_y, epsg_code)
            if lat:
                st.success(f"📍 Zentrum-Koordinaten (WGS84): {lat:.6f}, {lon:.6f}")
                st.write(f"[In Google Maps öffnen](https://www.google.com/maps/search/?api=1&query={lat},{lon})")

    except Exception as e:
        st.error(f"Fehler bei der Verarbeitung: {e}")
        st.info("Bitte prüfe, ob die Datei das Format X Y Z (mit Leerzeichen oder Komma getrennt) hat.")
else:
    # Willkommensbildschirm
    st.header("Willkommen beim LiDAR Archäologie-Analysetool")
    st.markdown("""
    Laden Sie eine `.xyz` oder `.txt` Datei mit Punktwolkendaten hoch, um fortzufahren.
    
    **Funktionen:**
    - **Composite Visualisierung:** Kombiniert Schummerung und Struktur-Analyse.
    - **LRM (Local Relief Model):** Entfernt großräumige Höhenunterschiede, um archäologische Merkmale (Wälle, Gräben) hervorzuheben.
    - **SVF-Proxy:** Macht Strukturen unabhängig vom Sonnenstand sichtbar.
    - **Profil-Tool:** Vermessen Sie Strukturen direkt im Browser.
    """)
    st.image("https://images.unsplash.com/photo-1510672981848-a1c4f1cb5ccf?auto=format&fit=crop&q=80&w=1000", caption="LiDAR ermöglicht den Blick durch das Blätterdach.")
