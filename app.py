import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io

# Versuche pyproj für die Koordinatenumrechnung zu importieren
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")
st.title("🏛️ LiDAR Analyse & High-Performance 3D")

# --- KOORDINATEN-FUNKTION ---
def convert_coords(x, y, from_epsg=2056):
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
    # Berechne Abmessungen des Rasters
    plot_width = int((df.x.max() - df.x.min()) / res)
    plot_height = int((df.y.max() - df.y.min()) / res)
    
    # Schutz vor leeren/ungültigen Dimensionen
    plot_width = max(2, plot_width)
    plot_height = max(2, plot_height)
    
    cvs = ds.Canvas(
        plot_width=plot_width,
        plot_height=plot_height,
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
    
    st.divider()
    st.subheader("Geo-Referenz")
    epsg_code = st.number_input(
        "EPSG Code (z.B. Schweiz LV95: 2056, Schweiz LV03: 21781, UTM 32N: 25832)", 
        value=2056
    )
    if not PYPROJ_AVAILABLE:
        st.warning("⚠️ 'pyproj' nicht gefunden. Koordinatenumrechnung deaktiviert.")
    
    st.divider()
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
        # 1. Daten robust und speicherschonend laden
        with st.spinner("Analysiere Dateistruktur und lade Daten..."):
            # Schnelle Formaterkennung vorab
            sample_bytes = uploaded_file.read(4096)
            uploaded_file.seek(0)
            sample_str = sample_bytes.decode('utf-8', errors='ignore')
            
            if ';' in sample_str:
                detected_sep = ';'
            elif ',' in sample_str:
                detected_sep = ','
            else:
                detected_sep = r'\s+'  # Standard: Leerzeichen/Tabs
            
            file_size_mb = uploaded_file.size / (1024 * 1024)
            estimated_rows = uploaded_file.size / 35  # Grobe Schätzung: ca. 35 Bytes pro Zeile
            target_rows = 1500000  # Maximal empfohlene Punktmenge für flüssiges Arbeiten
            
            # Strategie bei großen Dateien (> 30 MB oder geschätzt > 1.5 Mio Zeilen)
            if file_size_mb > 30.0 or estimated_rows > target_rows:
                st.info(f"⚡ Große Datei erkannt ({file_size_mb:.1f} MB). Lese und dezimiere Daten intelligent im Hintergrund...")
                
                sample_fraction = target_rows / estimated_rows
                sample_fraction = max(0.01, min(0.9, sample_fraction))  # Begrenze auf sinnvolle Werte
                
                chunks = []
                chunksize = 250000
                progress_bar = st.progress(0.0)
                
                # Chunked Reading, um RAM-Peaks zu vermeiden
                reader = pd.read_csv(
                    uploaded_file, 
                    sep=detected_sep, 
                    engine='python' if detected_sep == r'\s+' else 'c', 
                    header=None,
                    comment='#',
                    chunksize=chunksize
                )
                
                for i, chunk in enumerate(reader):
                    if chunk.shape[1] < 3:
                        continue
                    chunk = chunk.iloc[:, :3]
                    chunk.columns = ['x', 'y', 'z']
                    
                    # Schnelle numerische Bereinigung pro Chunk
                    for col in ['x', 'y', 'z']:
                        if chunk[col].dtype == object:
                            chunk[col] = chunk[col].astype(str).str.replace(',', '.', regex=False)
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                    
                    chunk = chunk.dropna()
                    
                    if len(chunk) > 0:
                        # Direkt im RAM runterskalieren, bevor wir den nächsten Chunk holen
                        sampled_chunk = chunk.sample(frac=sample_fraction, random_state=42)
                        chunks.append(sampled_chunk)
                    
                    # Update Fortschrittsbalken (gedeckelt bei max 95%)
                    progress_bar.progress(min(0.95, (i + 1) * 0.1))
                
                df = pd.concat(chunks, ignore_index=True)
                progress_bar.progress(1.0)
                st.success(f"✅ Datei eingelesen. Datensatz erfolgreich auf {len(df):,} repräsentative Punkte skaliert!")
            
            else:
                # Standard-Ladevorgang bei kleinen Dateien
                df = pd.read_csv(
                    uploaded_file, 
                    sep=detected_sep, 
                    engine='python' if detected_sep == r'\s+' else 'c', 
                    header=None,
                    comment='#'
                )
                
                if df.shape[1] < 3:
                    st.error("Die Datei enthält weniger als 3 Spalten. Bitte überprüfe das Dateiformat.")
                    st.stop()
                
                df = df.iloc[:, :3]
                df.columns = ['x', 'y', 'z']
                
                for col in ['x', 'y', 'z']:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna(subset=['x', 'y', 'z'])
                df = df.astype({'x': np.float32, 'y': np.float32, 'z': np.float32})
            
            # --- AUSREISSER-KOORDINATEN FILTER (Wichtig für Datashader) ---
            # Falls falsche Koordinaten (z.B. 0 oder gigantische Werte) eingestreut sind
            if len(df) > 10:
                q1_x, q3_x = df['x'].quantile(0.25), df['x'].quantile(0.75)
                iqr_x = q3_x - q1_x
                # Spatiale Grenzen (erlaubt maximale Streuung von 5 * IQR um den Median)
                df = df[(df['x'] >= q1_x - 5 * iqr_x) & (df['x'] <= q3_x + 5 * iqr_x)]
                
                q1_y, q3_y = df['y'].quantile(0.25), df['y'].quantile(0.75)
                iqr_y = q3_y - q1_y
                df = df[(df['y'] >= q1_y - 5 * iqr_y) & (df['y'] <= q3_y + 5 * iqr_y)]

        # Grenzen für Rückrechnung von Indizes auf Koordinaten
        min_x, max_x = df.x.min(), df.x.max()
        min_y, max_y = df.y.min(), df.y.max()

        # Standort berechnen (Standard: Zentrum)
        center_x, center_y = df.x.mean(), df.y.mean()
        lat, lon = convert_coords(center_x, center_y, epsg_code)
        
        # Container für dynamische Standort-Info
        location_placeholder = st.empty()
        if lat and lon:
            google_maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            location_placeholder.markdown(f"**📍 Standort (Zentrum):** [{lat:.5f}, {lon:.5f}]({google_maps_url})")

        # 2. Berechnungen
        with st.spinner("Analysiere Gelände..."):
            # --- CANVAS DIMENSION SCHUTZ ---
            # Ermittle wie groß das Raster werden würde
            width_px = int((max_x - min_x) / grid_res)
            height_px = int((max_y - min_y) / grid_res)
            max_pixels = 3000  # Maximal zulässige Breite/Höhe des Arrays im RAM
            
            grid_res_safe = grid_res
            if width_px > max_pixels or height_px > max_pixels:
                max_dim = max(width_px, height_px)
                scale_factor = max_dim / max_pixels
                grid_res_safe = grid_res * scale_factor
                st.warning(
                    f"⚠️ Die Geländefläche ist zu groß für eine Auflösung von {grid_res}m. "
                    f"Um einen RAM-Absturz zu verhindern, wurde die Auflösung automatisch "
                    f"auf **{grid_res_safe:.2f}m** korrigiert."
                )
            
            # Rasterisierung mit der sicheren Auflösung
            gz = rasterize_points(df, grid_res_safe)
            
            # Robustes Handling für ungültige Daten (z. B. wenn Raster nur NaNs enthält)
            if np.isnan(gz).all():
                gz = np.zeros_like(gz)
            else:
                gz = np.nan_to_num(gz, nan=np.nanmean(gz))
            
            # Alle Modelle berechnen
            nw_h = calculate_hillshade(gz, 315, 45, grid_res_safe)
            mds = calculate_multi_hillshade(gz, grid_res_safe)
            lrm = calculate_lrm(gz, lrm_sigma)
            slope = calculate_slope(gz, grid_res_safe)
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

        tab1, tab2 = st.tabs(["🖼️ 2D-Analyse", "🌐 3D-Prospektion"])

        # TAB 1: 2D
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
                plt.close(fig)

        # TAB 2: 3D
        with tab2:
            st.subheader("3D-Viewer")
            
            selected_texture = st.selectbox(
                "Wähle Analyse-Ebene für die 3D-Oberfläche:", 
                list(analysis_models.keys()),
                index=0
            )
            
            tex_data, tex_cmap, show_scale = analysis_models[selected_texture]
            
            # Schärfere Einstellung
            step = max(1, int(np.sqrt(gz.size / 400000)))
            z_plot = gz[::step, ::step]
            surface_tex = tex_data[::step, ::step]

            # Erstellung der Achsen-Werte
            x_vals = np.linspace(min_x, max_x, z_plot.shape[1])
            y_vals = np.linspace(min_y, max_y, z_plot.shape[0])

            fig3d = go.Figure(data=[go.Surface(
                x=x_vals,
                y=y_vals,
                z=z_plot, 
                surfacecolor=surface_tex, 
                colorscale=tex_cmap,
                showscale=show_scale,
                lighting=dict(ambient=0.6, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=0.5),
                lightposition=dict(x=100, y=100, z=1000),
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Höhe: %{z:.2f}m<extra></extra>'
            )])
            
            fig3d.update_layout(
                scene=dict(
                    aspectmode='data',
                    aspectratio=dict(x=1, y=1, z=z_exag),
                    xaxis=dict(title="X (m)"),
                    yaxis=dict(title="Y (m)"),
                    zaxis=dict(title="Höhe (m)")
                ),
                height=800,
                margin=dict(l=0, r=0, b=0, t=40),
                title=f"3D Ansicht: {selected_texture}"
            )
            
            st.plotly_chart(fig3d, use_container_width=True)

            st.info("💡 Pro-Tipp für Schärfe: Auflösung in Sidebar auf 0.5m stellen und Z-Überhöhung auf ca. 1.0 erhöhen.")

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte XYZ-Datei hochladen.")
