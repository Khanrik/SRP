from typing import Iterable
import rioxarray
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np

class Dataforsyningen:
    URL = "https://api.dataforsyningen.dk/dhm_wcs_DAF"

    def __init__(self, target_resolution: int):
        """
        Args:
            target_resolution: The desired resolution of the DEM data in meters per pixel.
        """
        self.meters_per_pixel = target_resolution
        self.session = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        load_dotenv()

    def get_data(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        divided_data = list(self.read_copernicus(output_path.parent / "copernicus"))
        for data in tqdm(divided_data, desc="Downloading dataforsyningen tiles"):
            x_min, y_min, _, _ = data.rio.bounds()
            out_file = output_path / f"dataforsyningen_{x_min:.2f}_{y_min:.2f}_meter.tif"

            if out_file.exists():
                continue

            params = self.get_params(data)
            response = self.session.get(self.URL, params=params, timeout=120)
            response.raise_for_status()

            with open(str(out_file), "wb") as f:
                f.write(response.content)

    def close(self):
        self.session.close()

    def read_copernicus(self, file_path: str | Path) -> Iterable[xr.DataArray]:
        for file in Path(file_path).glob("*_meter.tif"):
            with rioxarray.open_rasterio(file) as data:
                yield data.squeeze().drop_vars("band").load()
    
    def get_params(self, data: xr.DataArray) -> dict:
        x_min, y_min, x_max, y_max = data.rio.bounds()
        
        upscale_factor = int(data.rio.resolution()[0]) // self.meters_per_pixel
        target_height, target_width = np.array(data.shape) * upscale_factor

        params = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": "dhm_overflade",
            "crs": "EPSG:25832",
            "bbox": f"{x_min},{y_min},{x_max},{y_max}",
            "height": f"{target_height}",
            "width": f"{target_width}",
            "format": "GTiff",
            "token": os.getenv("DATATOKEN")
        }
        return params

def main():
    current_dir = Path(__file__).parent
    output_path = current_dir.parent / "data" / "dataforsyningen"
    dem_resolution = 10
    dataforsyningen = Dataforsyningen(target_resolution=dem_resolution)
    try:
        dataforsyningen.get_data(output_path)
    finally:
        dataforsyningen.close()

if __name__ == "__main__":
    main()