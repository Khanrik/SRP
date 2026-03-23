import rioxarray
import xarray as xr
from pathlib import Path
import requests
from dotenv import load_dotenv
import os
import numpy as np
from helpers import make_folders
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class Dataforsyningen:
    URL = "https://api.dataforsyningen.dk/dhm_wcs_DAF"

    def __init__(self, target_resolution: int):
        """
        Args:
            target_resolution: The desired resolution of the DEM data in meters per pixel.
            data_division: Proportions for dividing the data into train, validation, and test sets.
        """
        self.meters_per_pixel = target_resolution
        self.data_division_dict = {}
        self.upscale_factor = None
        load_dotenv()

        self.session = requests.Session()
        retry = Retry(
            total=3,
            read=3,
            connect=3,
            backoff_factor=0.3,
            status_forcelist=[104, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get_data(self, output_path: str | Path):
        output_path = Path(output_path)
        make_folders(output_path, "HR")

        cop_chunks = self.read_copernicus(output_path)
        self.get_dataforsyningen(cop_chunks, output_path)

    def read_copernicus(self, file_path: str | Path) -> tuple[xr.DataArray, tuple[int, int]]:
        chunks = []
        
        for division in ["train", "val", "test"]:
            for f in list(Path(file_path / division / "LR").glob("copernicus_*_meter.tif")):
                with rioxarray.open_rasterio(f) as raw_data:
                    data = raw_data.squeeze().drop_vars("band").load()
                    self.data_division_dict[id(data)] = division
                    chunks.append(data)

        self.upscale_factor = round(chunks[0].rio.resolution()[0]) // self.meters_per_pixel

        return chunks


    def get_dataforsyningen(self, lr_data: xr.DataArray, output_path: Path):
        datatoken = os.getenv("DATATOKEN")

        for data in tqdm(lr_data, desc="Fetching Dataforsyningen Data"):
            x_min, y_min, x_max, y_max = data.rio.bounds()
            out_file = output_path / self.data_division_dict[id(data)] / "HR" / f"dataforsyningen_{x_min:.0f}_{y_min:.0f}.tif"

            if out_file.exists():
                continue
            
            target_height, target_width = np.array(data.shape) * self.upscale_factor
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
                "token": datatoken
            }

            response = self.session.get(self.URL, params=params, timeout=60)
            response.raise_for_status()

            with open(str(out_file), "wb") as f:
                f.write(response.content)

def main():
    current_dir = Path(__file__).parent
    output_path = current_dir.parent / "data"
    dem_resolution = 10
    dataforsyningen = Dataforsyningen(target_resolution=dem_resolution)
    dataforsyningen.get_data(output_path)

if __name__ == "__main__":
    main()