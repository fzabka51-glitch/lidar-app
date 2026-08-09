import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.ndimage import laplace, gaussian_filter
import datashader as ds
import io

# --- NEUE MODULE (additiv, verändern keine bestehende Logik) -----------
import importers
import archaeology as arch
import exporters

# Versuche pyproj für die Koordinatenumrechnung zu importieren
try:
    import pyproj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False

# --- SEITENKONFIGURATION ---
st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")
st.title("🏛️ LiDAR Analyse & High-Performance 3D")

# --- NEU: Colormap-Übersetzung Matplotlib -> Plotly ---
# Die analysis_models verwenden Matplotlib-Namen (für den 2D-Tab), die 3D-Surface
# (Plotly) hat aber eine eigene, teils abweichende Namensliste. Diese Tabelle
# übersetzt bei Bedarf, ohne die Matplotlib-Namen im Dict selbst zu ändern.
MPL_TO_PLOTLY_CMAP = {
    "terrain": "earth",
    "twilight": "phase",
    "RdBu": "RdBu",
    "RdYlGn": "RdYlGn",
    "plasma": "plasma",
    "gray": "gray",
}

def to_plotly_colorscale(mpl_cmap_name):
    """Übersetzt einen Matplotlib-Colormap-Namen in eine gültige Plotly-Colorscale."""
    return MPL_TO_PLOTLY_CMAP.get(mpl_cmap_name, mpl_cmap_name)

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
    uploaded_file = st.file_uploader(
        "Datei laden (Punktwolke oder Raster)",
        type=["xyz", "txt", "csv", "las", "laz", "ply", "tif", "tiff", "geotiff", "asc"],
        help="Punktwolken: .xyz, .txt, .csv, .las, .laz, .ply | Raster: .tif, .tiff, .geotiff, .asc"
    )

    st.divider()
    st.subheader("Geo-Referenz")
    epsg_code = st.number_input("EPSG Code (z.B. UTM 32N: 25832)", value=25832)
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

    # --- NEU: Erweiterte archäologische Parameter ---
    st.divider()
    st.subheader("🏺 Archäologische Tools")
    enable_svf_openness = st.checkbox("SVF & Openness berechnen", value=False,
                                       help="Rechenintensiver als die Basismodelle - für große Raster ggf. deaktivieren.")
    svf_directions = st.slider("SVF/Openness Richtungen", 4, 16, 8)
    svf_radius_m = st.slider("SVF/Openness Suchradius (m)", 5, 50, 20)

    st.subheader("🔍 Feature-Erkennung")
    enable_feature_detection = st.checkbox("Automatische Feature-Erkennung", value=False)
    min_line_length = st.slider("Min. Linienlänge (px)", 5, 60, 15)
    circle_radius_range = st.slider("Kreisradius-Bereich (px)", 3, 80, (5, 40))

# --- HAUPTBEREICH ---
if uploaded_file:
    filename = uploaded_file.name
    try:
        is_raster_input = importers.is_raster(filename)

        # Puffer erzeugen, damit die Datei bei Bedarf mehrfach gelesen werden kann
        file_bytes = uploaded_file.read()
        buffer = io.BytesIO(file_bytes)

        if is_raster_input:
            # --- NEUER PFAD: direktes Raster (GeoTIFF/ASC) laden ---
            raster = importers.load_raster(buffer, filename)
            gz = np.nan_to_num(raster["array"], nan=np.nanmean(raster["array"]))
            grid_res_effective = raster["res"] if raster["res"] else grid_res
            min_x, min_y, max_x, max_y = raster["bounds"]
            raster_transform = raster["transform"]
            raster_crs = raster["crs"]
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
        else:
            # --- URSPRÜNGLICHER PFAD: Punktwolke laden & rasterisieren ---
            # Nutzt den neuen Dispatcher, damit .las/.laz/.ply zusätzlich zu
            # .xyz/.txt/.csv unterstützt werden. Verhalten für .xyz/.txt
            # bleibt exakt wie zuvor (load_xyz_txt ist unverändert).
            df = importers.load_point_cloud(buffer, filename)
            if len(df) > 2_000_000:
                st.warning("⚠️ Datensatz auf 2 Mio. Punkte reduziert.")

            min_x, max_x = df.x.min(), df.x.max()
            min_y, max_y = df.y.min(), df.y.max()
            center_x, center_y = df.x.mean(), df.y.mean()
            grid_res_effective = grid_res

            with st.spinner("Rasterisiere Punktwolke..."):
                gz = rasterize_points(df, grid_res_effective)
                gz = np.nan_to_num(gz, nan=np.nanmean(gz))

            raster_transform = exporters.build_transform_from_bounds(
                min_x, min_y, max_x, max_y, gz.shape[1], gz.shape[0]
            ) if exporters.RASTERIO_AVAILABLE else None
            raster_crs = f"EPSG:{int(epsg_code)}" if PYPROJ_AVAILABLE else None

        # Standort berechnen (Standard: Zentrum)
        lat, lon = convert_coords(center_x, center_y, epsg_code)

        location_placeholder = st.empty()
        if lat and lon:
            google_maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            location_placeholder.markdown(f"**📍 Standort (Zentrum):** [{lat:.5f}, {lon:.5f}]({google_maps_url})")

        # 2. Berechnungen (Basismodelle - unverändert)
        with st.spinner("Analysiere Gelände..."):
            nw_h = calculate_hillshade(gz, 315, 45, grid_res_effective)
            mds = calculate_multi_hillshade(gz, grid_res_effective)
            lrm = calculate_lrm(gz, lrm_sigma)
            slope = calculate_slope(gz, grid_res_effective)
            curv = calculate_curvature(gz)
            comp = np.clip(mds + (lrm - 0.5) * 0.3, 0, 1)

            analysis_models = {
                "Final Composite (Fusion)": (comp, "gray", False),
                "NW Hillshade": (nw_h, "gray", False),
                "MDS Composite": (mds, "gray", False),
                "Restrelief (LRM)": (lrm, "RdBu", True),
                "Hangneigung (Slope)": (slope, "plasma", True),
                "Krümmung (Curvature)": (curv, "RdYlGn", True),
            }

            # --- NEU: SVF & Openness (optional, da rechenintensiv) ---
            if enable_svf_openness:
                with st.spinner("Berechne Sky-View-Factor & Openness..."):
                    svf = arch.calculate_sky_view_factor(
                        gz, res=grid_res_effective, n_directions=svf_directions,
                        search_radius_m=svf_radius_m
                    )
                    pos_open = arch.calculate_openness(
                        gz, res=grid_res_effective, n_directions=svf_directions,
                        search_radius_m=svf_radius_m, mode="positive"
                    )
                    neg_open = arch.calculate_openness(
                        gz, res=grid_res_effective, n_directions=svf_directions,
                        search_radius_m=svf_radius_m, mode="negative"
                    )
                analysis_models["Sky-View-Factor (SVF)"] = (svf, "gray", True)
                analysis_models["Positive Openness"] = (pos_open, "gray", True)
                analysis_models["Negative Openness"] = (neg_open, "gray", True)

            # --- NEU: DTM-Approximation & aspect immer verfügbar ---
            aspect = arch.calculate_aspect(gz, grid_res_effective)
            dtm_approx = arch.estimate_dtm(gz)
            analysis_models["Hangausrichtung (Aspect)"] = (aspect, "twilight", True)
            analysis_models["DTM (approximiert)"] = (dtm_approx, "terrain", False)

        tab1, tab2, tab3, tab4 = st.tabs([
            "🖼️ 2D-Analyse", "🌐 3D-Prospektion", "🔍 Feature-Erkennung", "📦 GIS-Export"
        ])

        # TAB 1: 2D
        with tab1:
            if view_mode == "Gitter-Übersicht":
                c1, c2 = st.columns(2)
                for i, (name, (data, cmap, _)) in enumerate(analysis_models.items()):
                    with [c1, c2][i % 2]:
                        fig, ax = plt.subplots()
                        # origin='lower' korrigiert die spiegelverkehrte Y-Achse
                        ax.imshow(data, cmap=cmap, interpolation='none', origin='lower')
                        ax.set_title(name)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
            else:
                sel_2d = st.selectbox("Modell wählen:", list(analysis_models.keys()))
                data, cmap, _ = analysis_models[sel_2d]
                fig, ax = plt.subplots(figsize=(10, 6))
                # origin='lower' korrigiert die spiegelverkehrte Y-Achse
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
                colorscale=to_plotly_colorscale(tex_cmap),
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

        # TAB 3: NEU - Feature-Erkennung
        with tab3:
            st.subheader("Automatische Feature-Erkennung")
            if not enable_feature_detection:
                st.info("Aktiviere '🔍 Automatische Feature-Erkennung' in der Sidebar, um lineare "
                        "und zirkuläre Strukturen zu erkennen.")
            elif not arch.SKIMAGE_AVAILABLE:
                st.error("Das Paket 'scikit-image' ist nicht installiert. Bitte requirements.txt "
                         "installieren, um diese Funktion zu nutzen.")
            else:
                base_layer_name = st.selectbox(
                    "Basis-Layer für die Erkennung:",
                    ["Restrelief (LRM)", "MDS Composite"] +
                    (["Sky-View-Factor (SVF)", "Negative Openness"] if enable_svf_openness else []),
                    index=0
                )
                base_layer, _, _ = analysis_models[base_layer_name]

                col_lin, col_circ = st.columns(2)

                with st.spinner("Suche nach linearen und zirkulären Strukturen..."):
                    linear_features = arch.detect_linear_features(
                        base_layer, min_length_px=min_line_length
                    )
                    circular_features = arch.detect_circular_features(
                        base_layer,
                        min_radius_px=circle_radius_range[0],
                        max_radius_px=circle_radius_range[1],
                    )

                # Klassifikation anthropogen/natürlich
                linear_classified = [
                    (f, arch.classify_feature_origin(f, slope_at_location=float(np.nanmean(slope))))
                    for f in linear_features
                ]
                circular_classified = [
                    (f, arch.classify_feature_origin(f, slope_at_location=float(np.nanmean(slope))))
                    for f in circular_features
                ]

                with col_lin:
                    st.metric("Lineare Strukturen erkannt", len(linear_features))
                    st.caption("Potentielle Wälle, Gräben, Hohlwege, Terrassenkanten")
                with col_circ:
                    st.metric("Zirkuläre Strukturen erkannt", len(circular_features))
                    st.caption("Potentielle Ringwälle, Grabhügel, Pingo-Ruinen")

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(base_layer, cmap="gray", origin="lower")
                for f, label in linear_classified:
                    color = "red" if label == "wahrscheinlich_anthropogen" else "yellow"
                    ax.plot([f.start[1], f.end[1]], [f.start[0], f.end[0]], color=color, linewidth=1.5)
                for f, label in circular_classified:
                    color = "lime" if label == "wahrscheinlich_anthropogen" else "cyan"
                    circle = plt.Circle((f.center[1], f.center[0]), f.radius_px,
                                         fill=False, edgecolor=color, linewidth=1.5)
                    ax.add_patch(circle)
                ax.set_title(f"Feature-Erkennung auf Basis: {base_layer_name}")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

                st.caption("🔴/🟢 = wahrscheinlich anthropogen · 🟡/🔵 = unklar / wahrscheinlich natürlich "
                           "· Regelbasierte Vorklassifikation, keine archäologische Verifikation.")

                # Für den Export-Tab merken
                st.session_state["_linear_features"] = linear_features
                st.session_state["_circular_features"] = circular_features
                st.session_state["_feature_transform"] = raster_transform
                st.session_state["_feature_crs"] = raster_crs

        # TAB 4: NEU - GIS-Export
        with tab4:
            st.subheader("GIS-Export")
            st.caption("Exportiert Reliefmodelle als GeoTIFF sowie erkannte Features als Shapefile/GeoJSON.")

            if raster_transform is None:
                st.warning("⚠️ Keine Georeferenzierung (Affine-Transform) verfügbar - "
                           "GeoTIFF-Export ist deaktiviert. Punktwolken benötigen 'rasterio', "
                           "um automatisch eine Transform abzuleiten.")
            elif not exporters.RASTERIO_AVAILABLE:
                st.error("Das Paket 'rasterio' ist nicht installiert.")
            else:
                export_layer_name = st.selectbox(
                    "Layer für GeoTIFF-Export wählen:", list(analysis_models.keys())
                )
                export_data, _, _ = analysis_models[export_layer_name]

                if st.button("📤 Als GeoTIFF exportieren"):
                    tif_bytes = exporters.geotiff_bytes(
                        export_data, raster_transform, crs=raster_crs, nodata=-9999.0
                    )
                    st.download_button(
                        "⬇️ GeoTIFF herunterladen", data=tif_bytes,
                        file_name=f"{export_layer_name.replace(' ', '_')}.tif",
                        mime="image/tiff"
                    )

            st.divider()
            st.markdown("**Erkannte Features exportieren**")
            if not exporters.GEOPANDAS_AVAILABLE:
                st.error("Die Pakete 'geopandas'/'shapely' sind nicht installiert.")
            elif "_linear_features" not in st.session_state:
                st.info("Führe zuerst die Feature-Erkennung im Tab '🔍 Feature-Erkennung' aus.")
            else:
                export_format = st.radio("Format:", ["Shapefile (.shp)", "GeoJSON (.geojson)"])
                if st.button("📤 Features exportieren"):
                    import tempfile, zipfile, os as _os

                    lin = st.session_state["_linear_features"]
                    circ = st.session_state["_circular_features"]
                    trans = st.session_state["_feature_transform"]
                    crs_val = st.session_state["_feature_crs"]

                    with tempfile.TemporaryDirectory() as tmpdir:
                        if export_format.startswith("Shapefile"):
                            lin_path = _os.path.join(tmpdir, "lineare_features.shp")
                            circ_path = _os.path.join(tmpdir, "zirkulaere_features.shp")
                            if lin:
                                exporters.export_linear_features_shapefile(lin, lin_path, trans, crs=crs_val)
                            if circ:
                                exporters.export_circular_features_shapefile(circ, circ_path, trans, crs=crs_val)

                            zip_path = _os.path.join(tmpdir, "features_export.zip")
                            with zipfile.ZipFile(zip_path, "w") as zf:
                                for f in _os.listdir(tmpdir):
                                    if f.endswith((".shp", ".shx", ".dbf", ".prj", ".cpg")):
                                        zf.write(_os.path.join(tmpdir, f), arcname=f)
                            with open(zip_path, "rb") as f:
                                st.download_button("⬇️ Shapefile-Paket (.zip) herunterladen",
                                                    data=f.read(), file_name="features_export.zip",
                                                    mime="application/zip")
                        else:
                            lin_path = _os.path.join(tmpdir, "lineare_features.geojson")
                            circ_path = _os.path.join(tmpdir, "zirkulaere_features.geojson")
                            if lin:
                                exporters.export_geojson(lin, lin_path, trans, feature_type="linear", crs=crs_val)
                                with open(lin_path, "rb") as f:
                                    st.download_button("⬇️ Lineare Features (.geojson)", data=f.read(),
                                                        file_name="lineare_features.geojson",
                                                        mime="application/geo+json")
                            if circ:
                                exporters.export_geojson(circ, circ_path, trans, feature_type="circular", crs=crs_val)
                                with open(circ_path, "rb") as f:
                                    st.download_button("⬇️ Zirkuläre Features (.geojson)", data=f.read(),
                                                        file_name="zirkulaere_features.geojson",
                                                        mime="application/geo+json")

    except Exception as e:
        st.error(f"Fehler: {e}")
else:
    st.info("Bitte eine Punktwolken- oder Raster-Datei hochladen "
            "(.xyz, .txt, .csv, .las, .laz, .ply, .tif, .tiff, .geotiff, .asc).")
