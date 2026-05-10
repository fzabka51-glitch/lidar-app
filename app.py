import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io

# --- KONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")

try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

# --- CACHED ANALYSE-FUNKTIONEN ---
@st.cache_data
def load_and_rasterize(df_raw, res):
    """Rasterisiert die Punktwolke und speichert sie im Cache."""
    cvs = ds.Canvas(
        plot_width=int((df_raw.x.max() - df_raw.x.min()) / res),
        plot_height=int((df_raw.y.max() - df_raw.y.min()) / res),
        x_range=(df_raw.x.min(), df_raw.x.max()),
        y_range=(df_raw.y.min(), df_raw.y.max())
    )
    agg = cvs.points(df_raw, 'x', 'y', ds.mean('z'))
    grid = np.array(agg.values, dtype=np.float32)
    # NaN Handling: Fülle Löcher mit dem Mittelwert
    mask = np.isnan(grid)
    grid[mask] = np.nanmean(grid)
    return grid

def calculate_hillshade(data, azimuth=315, angle_altitude=45, res=1.0):
    azimuth_rad = np.deg2rad(azimuth)
    altitude_rad = np.deg2rad(angle_altitude)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(altitude_rad) * np.cos(slope)) + \
            (np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    return ((shade + 1) / 2).clip(0, 1)

@st.cache_data
def compute_all_models(gz, res, lrm_sigma):
    """Berechnet alle Analysemodelle auf einmal."""
    # Hillshades
    h_nw = calculate_hillshade(gz, 315, 45, res)
    h_ne = calculate_hillshade(gz, 45, 45, res)
    h_se = calculate_hillshade(gz, 135, 45, res)
    h_sw = calculate_hillshade(gz, 225, 45, res)
    mds = (h_nw + h_ne + h_se + h_sw) / 4.0

    # LRM (Local Relief Model)
    smoothed = gaussian_filter(gz, sigma=lrm_sigma)
    lrm_raw = gz - smoothed
    p_low, p_high = np.percentile(lrm_raw, (2, 98))
    lrm_norm = np.clip((lrm_raw - p_low) / (p_high - p_low), 0, 1)

    # Slope
    gy, gx = np.gradient(gz, res, res)
    slope = np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))
    
    # Curvature
    curv = -laplace(gz)
    p_c_low, p_c_high = np.percentile(curv, (5, 95))
    curv_norm = np.clip((curv - p_c_low) / (p_c_high - p_c_low), 0, 1)

    # Fusion (MDS + LRM für maximale Detailtiefe)
    composite = np.clip(mds * 0.7 + lrm_norm * 0.3, 0, 1)

    return {
        "Kombiniert (MDS+LRM)": (composite, "gray"),
        "Multi-Directional Shade": (mds, "gray"),
        "Local Relief (Restrelief)": (lrm_norm, "RdBu_r"),
        "Hangneigung": (slope, "Viridis"),
        "Krümmung": (curv_norm, "balance")
    }

# --- UI HELPERS ---
def convert_coords(x, y, from_epsg):
    if not PYPROJ_AVAILABLE: return None, None
    try:
        transformer = pyproj.Transformer.from_crs(f"epsg:{from_epsg}", "epsg:4326", always_xy=True)
        return transformer.transform(x, y)
    except: return None, None

# --- SIDEBAR ---
st.sidebar.title("🏛️ LiDAR Prospektion")
uploaded_file = st.sidebar.file_uploader("XYZ Datei hochladen", type=["xyz", "txt", "csv"])

if uploaded_file:
    # Lade Header-Info zur Vorschau
    df_sample = pd.read_csv(uploaded_file, sep=None, engine='python', nrows=5)
    st.sidebar.write("Daten-Vorschau:", df_sample.head(2))
    
    grid_res = st.sidebar.number_input("Raster-Auflösung (m)", 0.1, 5.0, 0.5)
    lrm_sigma = st.sidebar.slider("LRM Glättung (Stärke)", 5, 50, 15)
    z_exag = st.sidebar.slider("3D Überhöhung", 0.1, 10.0, 1.5)
    epsg_code = st.sidebar.number_input("EPSG Code", value=25832)

    # --- DATENVERARBEITUNG ---
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['x','y','z'])
    
    # Räumliche Grenzen
    min_x, max_x = df.x.min(), df.x.max()
    min_y, max_y = df.y.min(), df.y.max()
    
    with st.spinner("Generiere Raster..."):
        gz = load_and_rasterize(df, grid_res)
        models = compute_all_models(gz, grid_res, lrm_sigma)

    # --- NAVIGATION ---
    tab1, tab2, tab3 = st.tabs(["📊 2D Analyse", "🧊 3D Viewer", "📏 Profil & Export"])

    # --- TAB 1: 2D ---
    with tab1:
        sel_model = st.selectbox("Analyse-Modell wählen", list(models.keys()))
        data, cmap = models[sel_model]
        
        fig_2d = px.imshow(
            data, 
            color_continuous_scale=cmap, 
            origin='lower',
            aspect='equal',
            labels={'color': 'Intensität'}
        )
        fig_2d.update_layout(height=700, margin=dict(l=0,r=0,b=0,t=0))
        st.plotly_chart(fig_2d, use_container_width=True)

    # --- TAB 2: 3D ---
    with tab3: # Wir nutzen Tab 3 für das Profiling
        st.subheader("Geländeprofil & Vermessung")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("Klicke in die Karte, um ein Profil zu ziehen (Start/Ende via Slider simuliert)")
            # Slider für Profilschnitt (da Plotly-Click-Events in Streamlit komplex sind)
            row_idx = st.slider("Horizontales Profil auf Y-Index", 0, gz.shape[0]-1, gz.shape[0]//2)
            profile_data = gz[row_idx, :]
            
            fig_prof = px.line(
                x=np.linspace(min_x, max_x, len(profile_data)),
                y=profile_data,
                title=f"Querschnitt bei Y-Koordinate: {min_y + row_idx*grid_res:.2f}m",
                labels={'x': 'X (Meter)', 'y': 'Höhe (m)'}
            )
            st.plotly_chart(fig_prof, use_container_width=True)
            
        with col2:
            st.subheader("Export")
            # CSV Export
            csv_buffer = io.StringIO()
            pd.DataFrame(gz).to_csv(csv_buffer, index=False, header=False)
            st.download_button(
                "Raster als CSV herunterladen",
                data=csv_buffer.getvalue(),
                file_name="lidar_raster.csv",
                mime="text/csv"
            )
            
            if PYPROJ_AVAILABLE:
                lon, lat = convert_coords(df.x.mean(), df.y.mean(), epsg_code)
                st.success(f"Zentrum: {lat:.5f}, {lon:.5f}")
                st.markdown(f"[In Google Maps öffnen](https://www.google.com/maps/search/?api=1&query={lat},{lat})")

    # --- TAB 2: 3D (Wiederhergestellt und optimiert) ---
    with tab2:
        st.subheader("Interaktive 3D Prospektion")
        
        # Reduziere Auflösung für 3D Performance falls nötig
        target_pts = 200000
        step = max(1, int(np.sqrt(gz.size / target_pts)))
        
        z_downsampled = gz[::step, ::step]
        tex_downsampled, _ = models[sel_model]
        tex_downsampled = tex_downsampled[::step, ::step]
        
        x_range = np.linspace(min_x, max_x, z_downsampled.shape[1])
        y_range = np.linspace(min_y, max_y, z_downsampled.shape[0])

        fig_3d = go.Figure(data=[go.Surface(
            x=x_range,
            y=y_range,
            z=z_downsampled,
            surfacecolor=tex_downsampled,
            colorscale=cmap,
            showscale=False,
            hovertemplate="H: %{z:.2f}m<extra></extra>"
        )])

        fig_3d.update_layout(
            scene=dict(
                aspectmode='data',
                aspectratio=dict(x=1, y=1, z=z_exag),
                xaxis_title="Easting",
                yaxis_title="Northing",
                zaxis_title="Höhe"
            ),
            height=800,
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

else:
    # Welcome Screen
    st.title("Willkommen beim LiDAR Archäologie Viewer")
    st.info("Bitte lade eine .xyz Datei in der Sidebar hoch, um mit der Analyse zu beginnen.")
    st.image("https://images.unsplash.com/photo-1508804185872-d7badad00f7d?auto=format&fit=crop&q=80&w=1000", caption="LiDAR hilft versteckte Strukturen im Wald zu finden.")
