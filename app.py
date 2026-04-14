import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter, uniform_filter
import datashader as ds
import io

# Versuche pyproj für die Koordinatenumrechnung zu importieren
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro X-Ray", layout="wide")
st.title("🏛️ LiDAR Archäologie: Virtual Excavation & X-Ray")

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
    """Blitzschnelle Rasterisierung von Millionen Punkten mittels Datashader."""
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

def calculate_lrm(data, sigma=15):
    """Local Relief Model (LRM) - Entfernt großräumige Topographie."""
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p_low, p_high = np.percentile(residual, (2, 98))
    res_clipped = np.clip(residual, p_low, p_high)
    # Normalisierung
    res_min, res_max = res_clipped.min(), res_clipped.max()
    if res_max > res_min:
        return (res_clipped - res_min) / (res_max - res_min)
    return np.full_like(residual, 0.5)

def calculate_openness(data, size=5):
    """Vereinfachter Sky View Factor / Openness Effekt. 
    Zeigt 'Eingrabungen' extrem deutlich."""
    mean_val = uniform_filter(data, size=size)
    diff = data - mean_val
    # Verstärke die Kontraste für archäologische Features
    p_low, p_high = np.percentile(diff, (5, 95))
    diff = np.clip(diff, p_low, p_high)
    d_min, d_max = diff.min(), diff.max()
    return (diff - d_min) / (d_max - d_min) if d_max > d_min else diff

# --- SIDEBAR (STEUERUNG) ---
with st.sidebar:
    st.header("⚙️ Analyse-Werkzeuge")
    uploaded_file = st.file_uploader("XYZ Datei laden (.xyz, .txt)", type=["xyz", "txt"])
    
    st.divider()
    st.subheader("Geo-Referenz")
    epsg_code = st.number_input("EPSG Code (UTM 32N: 25832)", value=25832)
    
    st.divider()
    st.subheader("Virtual Digging (LRM/SVF)")
    grid_res = st.number_input("Auflösung (m)", 0.1, 5.0, 0.5)
    lrm_sigma = st.slider("Filter-Tiefe (Sigma)", 1, 100, 20, help="Höherer Wert zeigt größere Strukturen, kleinerer Wert feine Details.")
    svf_intensity = st.slider("Openness Radius", 2, 20, 5)
    
    st.subheader("3D-Visualisierung")
    z_exag = st.slider("Z-Überhöhung", 0.1, 10.0, 2.0)
    opacity = st.slider("Oberflächen-Transparenz", 0.1, 1.0, 1.0)
    
    st.subheader("Schnitt-Werkzeug")
    profile_axis = st.radio("Profil-Richtung", ["Horizontal", "Vertikal"])

# --- HAUPTBEREICH ---
if uploaded_file:
    try:
        # 1. Daten laden
        df = pd.read_csv(uploaded_file, sep=None, engine='python', header=None, names=['x','y','z'], dtype=np.float32)
        if len(df) > 2000000:
            df = df.sample(2000000, random_state=42)
            st.warning("⚠️ Datensatz auf 2 Mio. Punkte reduziert.")

        min_x, max_x = df.x.min(), df.x.max()
        min_y, max_y = df.y.min(), df.y.max()
        
        # 2. Berechnungen
        with st.spinner("Generiere virtuelle Ausgrabung..."):
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            # Archäologische Layer
            lrm = calculate_lrm(gz, lrm_sigma)
            openness = calculate_openness(gz, svf_intensity)
            hs = calculate_hillshade(gz, 315, 45, grid_res)
            
            # Kombiniertes 'X-Ray' Bild
            xray = np.clip(hs * 0.4 + lrm * 0.6, 0, 1)

            analysis_models = {
                "X-Ray Composite": (xray, "gray", "Kombination aus Relief und Schatten"),
                "Restrelief (LRM)": (lrm, "RdBu_r", "Entfernt den Hang, zeigt nur Strukturen"),
                "Openness / SVF": (openness, "magma", "Highlights für Gräben und Mauern"),
                "Schummerung": (hs, "gray", "Klassische Ansicht"),
                "Höhenmodell": (gz, "terrain", "Absolute Höhen")
            }

        tab1, tab2, tab3 = st.tabs(["🔍 Detektion (2D)", "🏢 Gelände-Schnitt", "🌐 3D Prospektion"])

        # TAB 1: 2D DETEKTION
        with tab1:
            sel_2d = st.selectbox("Analyse-Modus:", list(analysis_models.keys()))
            data, cmap, desc = analysis_models[sel_2d]
            st.caption(desc)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(data, cmap=cmap, origin='lower', extent=[min_x, max_x, min_y, max_y])
            ax.set_xlabel("Easting")
            ax.set_ylabel("Northing")
            plt.colorbar(im, ax=ax, label="Intensität")
            st.pyplot(fig)
            plt.close(fig)

        # TAB 2: PROFIL-SCHNITT (Das "Unter-die-Erde" Tool)
        with tab2:
            st.subheader("Archäologischer Geländeschnitt")
            st.info("Bewege den Schieberegler, um das Gelände vertikal zu schneiden.")
            
            if profile_axis == "Horizontal":
                slice_idx = st.slider("Y-Position wählen", 0, gz.shape[0]-1, gz.shape[0]//2)
                profile_data = gz[slice_idx, :]
                dist_axis = np.linspace(0, gz.shape[1] * grid_res, gz.shape[1])
                title = f"Ost-West Schnitt bei Y-Index {slice_idx}"
            else:
                slice_idx = st.slider("X-Position wählen", 0, gz.shape[1]-1, gz.shape[1]//2)
                profile_data = gz[:, slice_idx]
                dist_axis = np.linspace(0, gz.shape[0] * grid_res, gz.shape[0])
                title = f"Nord-Süd Schnitt bei X-Index {slice_idx}"

            fig_prof = go.Figure()
            fig_prof.add_trace(go.Scatter(x=dist_axis, y=profile_data, fill='tozeroy', line=dict(color='brown', width=2)))
            fig_prof.update_layout(
                title=title,
                xaxis_title="Distanz im Schnitt (m)",
                yaxis_title="Höhe über NN (m)",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig_prof, use_container_width=True)
            st.write("💡 Hier siehst du die exakte Form von Gräben oder Wällen, die oft unter 20cm tief sind.")

        # TAB 3: 3D
        with tab3:
            selected_texture = st.selectbox("Textur für 3D-Modell:", list(analysis_models.keys()), key="3d_sel")
            tex_data, tex_cmap, _ = analysis_models[selected_texture]
            
            # Downsampling für Performance
            res_target = 200
            step = max(1, gz.shape[0] // res_target)
            z_plot = gz[::step, ::step]
            surface_tex = tex_data[::step, ::step]

            x_vals = np.linspace(min_x, max_x, z_plot.shape[1])
            y_vals = np.linspace(min_y, max_y, z_plot.shape[0])

            fig3d = go.Figure(data=[go.Surface(
                x=x_vals, y=y_vals, z=z_plot,
                surfacecolor=surface_tex,
                colorscale=tex_cmap,
                opacity=opacity,
                lighting=dict(ambient=0.7, diffuse=0.9),
                hovertemplate='Höhe: %{z:.2f}m<extra></extra>'
            )])
            
            fig3d.update_layout(
                scene=dict(
                    aspectmode='data',
                    aspectratio=dict(x=1, y=1, z=z_exag),
                ),
                height=700,
                margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)

    except Exception as e:
        st.error(f"Fehler bei der Analyse: {e}")
else:
    st.info("Bitte lade eine XYZ-Datei hoch, um mit der virtuellen Ausgrabung zu beginnen.")
