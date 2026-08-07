"""
LiDAR Archäologie Pro – Entry Point.

Diese Datei ist bewusst dünn: sie verdrahtet nur UI <-> IO <-> Analyse.
Die eigentliche Logik lebt in einzelnen Dateien wie analysis.py, io_utils.py usw.
"""

import logging
import streamlit as st

from analysis import run_analysis
from io_utils import PointCloudLoadError, convert_coords, load_point_cloud
from ui import render_2d_tab, render_3d_tab, render_location_info, render_sidebar

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="LiDAR Archäologie Pro", layout="wide")
st.title("🏛️ LiDAR Analyse & High-Performance 3D")

uploaded_file, params = render_sidebar()

if not uploaded_file:
    st.info("Bitte eine Punktwolke hochladen (.xyz, .txt, .las, .laz).")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    with st.spinner("Lade Punktwolke..."):
        cloud = load_point_cloud(file_bytes, uploaded_file.name)

    if cloud.n_original > len(cloud.x):
        st.warning(
            f"⚠️ Datensatz von {cloud.n_original:,} auf {len(cloud.x):,} Punkte reduziert."
        )

    min_x, max_x, min_y, max_y = cloud.bounds
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    lat, lon = convert_coords(center_x, center_y, params.epsg_code)
    render_location_info(lat, lon)

    with st.spinner("Analysiere Gelände..."):
        result = run_analysis(
            cloud.x, cloud.y, cloud.z,
            res=params.grid_res,
            lrm_sigma=params.lrm_sigma,
            use_gpu=params.use_gpu,
        )

    tab1, tab2 = st.tabs(["🖼️ 2D-Analyse", "🌐 3D-Prospektion"])
    with tab1:
        render_2d_tab(result, params.view_mode)
    with tab2:
        render_3d_tab(result, params.z_exag)

except PointCloudLoadError as exc:
    st.error(f"Datenfehler: {exc}")
except Exception as exc:
    logging.exception("Unerwarteter Fehler in der App-Pipeline")
    st.error(f"Unerwarteter Fehler: {exc}")
    st.caption("Details wurden geloggt. Bitte Datei/Parameter prüfen oder Issue melden.")
