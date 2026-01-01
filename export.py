import rasterio
from rasterio.transform import from_origin
import io

def save_geotiff(array, x0, y0, res, fname="relief.tif", epsg="EPSG:25832"):
    transform = from_origin(x0, y0, res, res)
    with rasterio.open(
        fname, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1],
        count=1, dtype=array.dtype,
        crs=epsg, transform=transform
    ) as dst:
        dst.write(array, 1)
    return fname

def csv_buffer(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()
