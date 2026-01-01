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

def calc_hillshade(data, az=315, alt=45, res=1.0):
    """Berechnet Schattierung (Hillshade)."""
    az_rad, alt_rad = np.deg2rad(az), np.deg2rad(alt)
    gy, gx = np.gradient(data, res, res)
    slope = np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, gx)
    shade = (np.cos(alt_rad) * np.cos(slope)) + (np.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return ((shade + 1) / 2).astype(np.float32)

def calc_mds(data, res=1.0):
    """Multi-Directional Hillshade (MDS) aus 4 Richtungen."""
    h1 = calc_hillshade(data, 315, 45, res)
    h2 = calc_hillshade(data, 45, 45, res)
    h3 = calc_hillshade(data, 135, 45, res)
    h4 = calc_hillshade(data, 225, 45, res)
    return (h1 + h2 + h3 + h4) / 4.0

def calc_lrm(data, sigma=15):
    """Local Relief Model (LRM) zur Isolation von archäologischen Strukturen."""
    smoothed = gaussian_filter(data, sigma=sigma)
    res = data - smoothed
    p_l, p_h = np.percentile(res, (5, 95))
    res = np.clip(res, p_l, p_h)
    return (res - res.min()) / (res.max() - res.min())

def calc_slope(data, res=1.0):
    """Berechnet die Hangneigung in Grad."""
    gy, gx = np.gradient(data, res, res)
    slope_deg = np.rad2deg(np.arctan(np.sqrt(gx**2 + gy**2)))
    return np.clip(slope_deg, 0, np.nanpercentile(slope_deg, 98))

# --- SIDEBAR (STEUERUNG) ---
with st.sidebar:
    st.header("⚙️ Parameter")
    uploaded_file = st.file_uploader("XYZ Datei laden (.xyz, .txt)", type=["xyz", "txt"])
    
    st.subheader("Raster & Filter")
    grid_res = st.number_input("Auflösung (m)", 0.1, 10.0, 1.0, help="Standard: 1.0m für Übersicht, 0.5m für Details.")
    lrm_sigma = st.slider("LRM Glättung (Sigma)", 1, 50, 15, help="Normal: 10-15. Höher für große Wälle.")
    
    st.subheader("3D-Darstellung")
    z_exag = st.slider("3D Überhöhung (Z)", 0.1, 10.0, 2.0, help="Normal: 1.5 - 3.0 für Relief-Sichtbarkeit.")
    
    st.subheader("Visualisierung")
    cmap_choice = st.selectbox("Farbschema 2D", ["gray", "RdBu_r", "plasma", "terrain", "viridis"])
    view_mode = st.radio("Ansicht Modus:", ["Gitter-Übersicht", "Einzelansicht (Groß)"])

# --- HAUPTBEREICH ---
if uploaded_file:
    try:
        # 1. Daten laden (Begrenzung für Performance)
        df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['x','y','z'], dtype=np.float32)
        if len(df) > 2000000:
            df = df.sample(2000000, random_state=42)
            st.warning("⚠️ Datensatz wurde auf 2 Mio. Punkte reduziert, um die App stabil zu halten.")

        # 2. Rasterung & Analysen
        with st.spinner("Berechne Gelände-Modelle..."):
            gz = rasterize_points(df, grid_res)
            gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            mds = calc_mds(gz, grid_res)
            lrm = calc_lrm(gz, lrm_sigma)
            slope = calc_slope(gz, grid_res)
            curv = -laplace(gz)
            # Fusion: LRM Details kombiniert mit MDS Schattierung
            comp = np.clip(mds + (lrm - 0.5) * 0.4, 0, 1)

        # Tab-Struktur
        tab1, tab2, tab3 = st.tabs(["🖼️ 2D-Analyse", "🌐 3D-Prospektion", "📊 Statistik"])

        # TAB 1: 2D Visualisierung
        with tab1:
            results = {
                "MDS Hillshade": (mds, "gray"),
                "LRM (Strukturen)": (lrm, "RdBu_r"),
                "Fusion (MDS+LRM)": (comp, "gray"),
                "Hangneigung (Slope)": (slope, "plasma"),
                "Krümmung (Curvature)": (curv, "RdYlGn")
            }
            
            if view_mode == "Gitter-Übersicht":
                c1, c2 = st.columns(2)
                for i, (name, (data, default_cmap)) in enumerate(results.items()):
                    with [c1, c2][i % 2]:
                        fig, ax = plt.subplots()
                        # Clipping für Krümmung
                        if "Curvature" in name:
                            p_l, p_h = np.percentile(data, (2, 98))
                            data = np.clip(data, p_l, p_h)
                        ax.imshow(data, cmap=cmap_choice if i > 0 else default_cmap)
                        ax.set_title(name)
                        ax.axis('off')
                        st.pyplot(fig)
            else:
                sel = st.selectbox("Wähle Analyse für Großansicht:", list(results.keys()))
                data, default_cmap = results[sel]
                fig, ax = plt.subplots(figsize=(10, 6))
                if "Curvature" in sel:
                    p_l, p_h = np.percentile(data, (2, 98))
                    data = np.clip(data, p_l, p_h)
                ax.imshow(data, cmap=cmap_choice)
                ax.axis('off')
                st.pyplot(fig)

        # TAB 2: 3D Visualisierung (COLAB-STYLE)
        with tab2:
            st.subheader("Interaktives 3D-Relief")
            
            # Downsampling für flüssige Bedienung
            step = max(1, int(np.sqrt(gz.size / 150000)))
            z_plot = gz[::step, ::step]
            c_plot = comp[::step, ::step] # Nutzt Fusion als Oberflächentestur

            fig3d = go.Figure(data=[go.Surface(
                z=z_plot, 
                surfacecolor=c_plot, 
                colorscale='Greys', 
                showscale=False,
                lighting=dict(ambient=0.6, diffuse=0.7, fresnel=0.2, specular=0.1, roughness=0.8),
                lightposition=dict(x=100, y=100, z=1000)
            )])
            
            fig3d.update_layout(
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=z_exag),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(title="Höhe (m)", backgroundcolor="rgb(230, 230,230)")
                ),
                height=850,
                margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)
            st.info("💡 Nutze die linke Maustaste zum Drehen und das Mausrad zum Zoomen.")

        # TAB 3: Statistiken
        with tab3:
            st.subheader("Geländestatistiken")
            col_a, col_b = st.columns(2)
            with col_a:
                fig_h, ax_h = plt.subplots()
                ax_h.hist(gz.flatten(), bins=50, color='skyblue', edgecolor='black')
                ax_h.set_title("Höhenverteilung (Hypsometrie)")
                st.pyplot(fig_h)
            with col_b:
                st.write("**Wertebereiche:**")
                st.metric("Minimale Höhe", f"{gz.min():.2f} m")
                st.metric("Maximale Höhe", f"{gz.max():.2f} m")
                st.metric("Spannweite", f"{gz.max() - gz.min():.2f} m")

    except Exception as e:
        st.error(f"Kritischer Fehler bei der Verarbeitung: {e}")
else:
    st.info("Willkommen! Bitte lade eine LiDAR-Datei (.xyz) hoch, um die Analyse zu starten.")
