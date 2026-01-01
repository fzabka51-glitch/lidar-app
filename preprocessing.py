import numpy as np
import pandas as pd
import datashader as ds
import cupy as cp

def use_gpu():
    try:
        _ = cp.zeros((1,))
        return True
    except Exception:
        return False

def rasterize_points(df, res):
    cvs = ds.Canvas(
        plot_width=int((df.x.max()-df.x.min())/res),
        plot_height=int((df.y.max()-df.y.min())/res),
        x_range=(df.x.min(), df.x.max()),
        y_range=(df.y.min(), df.y.max())
    )
    agg = cvs.points(df, 'x', 'y', ds.mean('z'))
    return np.array(agg.values, dtype=np.float32)

def calc_hillshade(data, az=315, alt=45, res=1.0, xp=np):
    azr, altr = np.deg2rad(az), np.deg2rad(alt)
    gy, gx = xp.gradient(data, res, res)
    slope = xp.arctan(xp.sqrt(gx**2 + gy**2))
    aspect = xp.arctan2(-gy, gx)
    shade = (xp.cos(altr)*xp.cos(slope)) + (xp.sin(altr)*xp.sin(slope)*xp.cos(azr-aspect))
    return ((shade+1)/2).astype(xp.float32)

def calc_mds(data, res=1.0, xp=np):
    h1 = calc_hillshade(data, 315, 45, res, xp)
    h2 = calc_hillshade(data, 45, 45, res, xp)
    h3 = calc_hillshade(data, 135, 45, res, xp)
    h4 = calc_hillshade(data, 225, 45, res, xp)
    return (h1+h2+h3+h4)/4.0

def calc_lrm(data, sigma=10, xp=np):
    from scipy.ndimage import gaussian_filter
    f = gaussian_filter(np.asarray(data), sigma=sigma)
    res = data - (xp.asarray(f) if xp is cp else f)
    p1, p2 = xp.percentile(res, (5,95))
    res = xp.clip(res, p1, p2)
    return (res-res.min())/(res.max()-res.min())

def calc_slope(data, res=1.0, xp=np):
    gy, gx = xp.gradient(data, res, res)
    slope = xp.rad2deg(xp.arctan(xp.sqrt(gx**2 + gy**2)))
    return xp.clip(slope, 0, xp.nanpercentile(slope, 98))
