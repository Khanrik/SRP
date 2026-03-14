from typing import List
from pyproj import Transformer
import rioxarray
import xarray as xr
from planetary_computer import sign
from pathlib import Path
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import os

class Dataforsyningen:
    # predetermined bounding box for the area of interest
    LON_MIN = 9.0
    LAT_MIN = 55.000277777777775
    LON_MAX = 9.999583333333334
    LAT_MAX = 57.0

    URL = "https://api.dataforsyningen.dk/dhm_wcs_DAF"

    def __init__(self, target_resolution: tuple[int, int]):
        """
        Args:
            target_resolution: (height, width) of the output data.
        """
        self.target_resolution = target_resolution
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        load_dotenv()

    def get_data(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        divided_data = self.read_copernicus(output_path.parent / "copernicus")
        for data in tqdm(divided_data, desc="Downloading dataforsyningen tiles"):
            lon_min, lat_min, _, _ = data.rio.bounds()
            out_file = output_path / f"dataforsyningen_{lon_min:.5f}_{lat_min:.5f}.tif"
            params = self.get_params(data)
            response = requests.get(self.URL, params=params)

            with open(str(out_file), "wb") as f:
                f.write(response.content)

    def read_copernicus(self, file_path: str | Path) -> List[xr.DataArray]:
        data_list = []
        for file in Path(file_path).glob("*.tif"):
            data = rioxarray.open_rasterio(file)
            data_list.append(data.squeeze().drop_vars("band"))

        return data_list
    
    def get_params(self, data: xr.DataArray) -> dict:
        lon_min, lat_min, lon_max, lat_max = data.rio.bounds()

        # Transform the bounds to WGS84
        x_min, y_min = self.transformer.transform(lon_min, lat_min)
        x_max, y_max = self.transformer.transform(lon_max, lat_max)

        params = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": "dhm_terraen",
            "crs": "EPSG:25832",
            "bbox": f"{x_min},{y_min},{x_max},{y_max}",
            "height": f"{self.target_resolution[0]}",
            "width": f"{self.target_resolution[1]}",
            "format": "GTiff",
            "token": os.getenv("DATATOKEN")
        }
        return params

def main():
    current_dir = Path(__file__).parent
    output_path = current_dir.parent / "data" / "dataforsyningen"
    resolution = (512*3, 512*3)
    dataforsyningen = Dataforsyningen(target_resolution=resolution)
    dataforsyningen.get_data(output_path)

if __name__ == "__main__":
    main()