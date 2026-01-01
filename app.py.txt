import streamlit as st, numpy as np, pandas as pd, plotly.graph_objects as go, matplotlib.pyplot as plt
from scipy.ndimage import laplace
from modules import preprocessing as pre, analysis as ana, export as exp
import os

st.set_page_config(page_title="LiDAR Archäologie Pro++", layout="wide")
st.title("🏛️ LiDAR Analyse & 3D-Prospektion (Pro++)")

with st.sidebar:
    file = st.file_uploader("XYZ-Datei", type=["xyz","txt"])
    res = st.number_input("Rasterauflösung (m)",0.1,10.0,1.0)
    sigma = st.slider("LRM-Glättung (σ)",1,50,15)
    zex = st.slider("3D-Überhöhung",0.1,5.0,0.8)
    cmap = st.selectbox("Farbschema",["gray","RdBu_r","plasma","terrain","viridis"])

if not file:
    st.info("Bitte Datei hochladen.")
    st.stop()

st.info("⏳ Verarbeite Punktdaten...")
df = pd.read_csv(file, sep=r"\s+", header=None, names=["x","y","z"], dtype=np.float32)
if len(df)>2_000_000:
    df = df.sample(2_000_000, random_state=1)
    st.warning("⚠️ auf 2 Mio. Punkte reduziert.")

use_gpu = pre.use_gpu()
xp = __import__("cupy") if use_gpu else np

gz = pre.rasterize_points(df, res)
gz = xp.asarray(gz) if use_gpu else gz
gz = xp.nan_to_num(gz, nan=xp.nanmean(gz))
mds = pre.calc_mds(gz, res, xp)
lrm = pre.calc_lrm(gz, sigma, xp)
slope = pre.calc_slope(gz, res, xp)
curv = -laplace(np.asarray(gz))
comp = xp.clip(mds+(lrm-0.5)*0.4,0,1)
to_np = lambda a: a.get() if use_gpu else a

tabs = st.tabs(["🖼️ 2D","🌐 3D","📊 Analyse","💾 Export"])
with tabs[0]:
    c1,c2,c3 = st.columns(3)
    imgs = {
        "MDS": to_np(mds),"LRM": to_np(lrm),
        "Fusion": to_np(comp),"Slope": to_np(slope),
        "Curvature": curv
    }
    for i,(k,v) in enumerate(imgs.items()):
        with [c1,c2,c3][i%3]:
            fig,ax=plt.subplots(); ax.imshow(v,cmap=cmap); ax.set_title(k); ax.axis("off"); st.pyplot(fig)

with tabs[1]:
    step = max(2,int(np.sqrt(gz.size/150000)))
    zplot = to_np(gz[::step,::step])
    cplot = to_np(comp[::step,::step])
    fig3d = go.Figure(data=[go.Surface(z=zplot,surfacecolor=cplot,colorscale=cmap,showscale=False)])
    fig3d.update_layout(scene=dict(aspectratio=dict(x=1,y=1,z=zex)),height=700,margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig3d,use_container_width=True)

with tabs[2]:
    st.pyplot(ana.elevation_histogram(to_np(gz)))
    st.json(ana.slope_stats(to_np(slope)))

with tabs[3]:
    tiff = exp.save_geotiff(to_np(gz), df.x.min(), df.y.max(), res)
    st.download_button("⬇️ GeoTIFF", data=open(tiff,"rb").read(), file_name=os.path.basename(tiff), mime="image/tiff")
    csv_data = exp.csv_buffer(df.head(1000))
    st.download_button("⬇️ CSV (1000 Punkte)", data=csv_data, file_name="punkte.csv", mime="text/csv")
