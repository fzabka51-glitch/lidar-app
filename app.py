import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")
st.title("🏛️ LiDAR Analyse & High-Performance 3D")

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

def calculate_multi_hillshade(data, res=1.0):
    """Multi-Directional Shading (MDS) aus 4 Richtungen."""
    h1 = calculate_hillshade(data, 315, 45, res)
    h2 = calculate_hillshade(data, 45, 45, res)
    h3 = calculate_hillshade(data, 135, 45, res)
    h4 = calculate_hillshade(data, 225, 45, res)
    return (h1 + h2 + h3 + h4) / 4.0

def calculate_lrm(data, sigma=15):
    """Local Relief Model (LRM) / Residual Topography."""
    smoothed = gaussian_filter(data, sigma=sigma)
    residual = data - smoothed
    p_low, p_high = np.percentile(residual, (5, 95))
    res_clipped = np.clip(residual, p_low, p_high)
    # Normalisierung auf 0-1 für Texturierung
    res_min, res_max = res_clipped.min(), res_clipped.max()
    if res_max > res_min:
        return (res_clipped - res_min) / (res_max - res_min)
    return np.full_like(residual, 0.5)

def calculate_slope(data, res=1.0):
    """Berechnet die Hangneigung in Grad."""
    gy, gx = np.gradient(data, res, res)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))
    p_high = np.nanpercentile(slope_deg, 98)
    return np.clip(slope_deg, 0, p_high)

def calculate_curvature(data):
    """Berechnet die lokale Krümmung (Laplace)."""
    curv = -laplace(data)
    p_low, p_high = np.percentile(curv, (2, 98))
    curv_clipped = np.clip(curv, p_low, p_high)
    c_min, c_max = curv_clipped.min(), curv_clipped.max()
    if c_max > c_min:
        return (curv_clipped - c_min) / (c_max - c_min)
    return np.full_like(curv, 0.5)

# --- SIDEBAR (STEUERUNG) ---
with st.sidebar:
    st.header("⚙️ Parameter")
    uploaded_file = st.file_uploader("XYZ Datei laden (.xyz, .txt)", type=["xyz", "txt"])
    
    st.subheader("Raster & Filter")
    grid_res = st.number_input("Auflösung (m)", 0.1, 10.0, 1.0, help="Niedrigerer Wert = Höhere Schärfe (z.B. 0.5m)")
    lrm_sigma = st.slider("LRM Glättung (Sigma)", 1, 50, 15)
    
    st.subheader("3D-Eigenschaften")
    z_exag = st.slider("Z-Überhöhung", 0.1, 5.0, 0.5, step=0.1)
    
    st.subheader("Anzeige")
    view_mode = st.radio("Ansicht 2D:", ["Gitter-Übersicht", "Einzelansicht"])

# --- HAUPTBEREICH ---
if uploaded_file:
    try:
        # 1. Daten laden
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['x','y','z'], dtype=np.float32)
        if len(df) > 2000000:
            df = df.sample(2000000, random_state=42)
            st.warning("⚠️ Datensatz auf 2 Mio. Punkte reduziert.")

        # 2. Berechnungen
        with st.spinner("Analysiere Gelände..."):
            gz = rasterize_points(df, grid_res)
            # Reinigung
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            # Alle Modelle berechnen
            nw_h = calculate_hillshade(gz, 315, 45, grid_res)
            mds = calculate_multi_hillshade(gz, grid_res)
            lrm = calculate_lrm(gz, lrm_sigma)
            slope = calculate_slope(gz, grid_res)
            curv = calculate_curvature(gz)
            # Fusion
            comp = np.clip(mds + (lrm - 0.5) * 0.3, 0, 1)

            analysis_models = {
                "Final Composite (Fusion)": (comp, "gray", False),
                "NW Hillshade": (nw_h, "gray", False),
                "MDS Composite": (mds, "gray", False),
                "Restrelief (LRM)": (lrm, "RdBu", True),
                "Hangneigung (Slope)": (slope, "plasma", True),
                "Krümmung (Curvature)": (curv, "RdYlGn", True)
            }

        tab1, tab2 = st.tabs(["🖼️ 2D-Analyse", "🌐 3D-Prospektion (Interaktiv)"])

        # TAB 1: 2D
        with tab1:
            if view_mode == "Gitter-Übersicht":
                c1, c2 = st.columns(2)
                for i, (name, (data, cmap, _)) in enumerate(analysis_models.items()):
                    with [c1, c2][i % 2]:
                        fig, ax = plt.subplots()
                        # 'none' interpolation verhindert Verschwimmen in Matplotlib
                        ax.imshow(data, cmap=cmap, interpolation='none')
                        ax.set_title(name)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                sel_2d = st.selectbox("Modell wählen:", list(analysis_models.keys()))
                data, cmap, _ = analysis_models[sel_2d]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(data, cmap=cmap, interpolation='none')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

        # TAB 2: 3D
        with tab2:
            st.subheader("Interaktiver 3D-Viewer")
            
            selected_texture = st.selectbox(
                "Wähle Analyse-Ebene für die 3D-Oberfläche:", 
                list(analysis_models.keys()),
                index=0
            )
            
            tex_data, tex_cmap, show_scale = analysis_models[selected_texture]
            
            # Schärfere Einstellung: Erhöhung des Limits auf 400.000 Punkte
            step = max(1, int(np.sqrt(gz.size / 400000)))
            z_plot = gz[::step, ::step]
            surface_tex = tex_data[::step, ::step]

            fig3d = go.Figure(data=[go.Surface(
                z=z_plot, 
                surfacecolor=surface_tex, 
                colorscale=tex_cmap,
                showscale=show_scale,
                lighting=dict(ambient=0.6, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=0.5),
                lightposition=dict(x=100, y=100, z=1000)
            )])
            
            fig3d.update_layout(
                scene=dict(
                    aspectmode='data',
                    aspectratio=dict(x=1, y=1, z=z_exag),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(title="Höhe (m)")
                ),
                height=900,
                margin=dict(l=0, r=0, b=0, t=40),
                title=f"3D Ansicht: {selected_texture}"
            )
            
            st.plotly_chart(fig3d, use_container_width=True)
            st.info("💡 Pro-Tipp für Schärfe: Auflösung in Sidebar auf 0.5m stellen und Z-Überhöhung auf ca. 1.0 erhöhen.")

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte XYZ-Datei hochladen.")
