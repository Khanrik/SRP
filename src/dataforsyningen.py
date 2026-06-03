from __future__ import annotations
import rioxarray
import xarray as xr
from pathlib import Path
import requests
from dotenv import load_dotenv
import os
import numpy as np
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class Dataforsyningen:
    """ Class for fetching and processing DEM data from the Dataforsyningen dataset using their WCS API. The class allows for reading existing Copernicus data to determine the necessary parameters for fetching the corresponding high-resolution DEM data from Dataforsyningen, and includes functionality to handle retries for failed requests.
    Args:
        target_resolution (int): The desired resolution of the DEM data in meters per pixel.
    """
    URL = "https://api.dataforsyningen.dk/dhm_wcs_DAF"

    def __init__(self, target_resolution: int):
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

    def get_data(self, output_path: Path):
        """Gets DEM data from Dataforsyningen based on existing Copernicus data, and saves the fetched data to the specified output path. The function first reads the existing Copernicus data to determine the necessary parameters for fetching the corresponding high-resolution DEM data from Dataforsyningen, then iterates through the required chunks of data, making requests to the Dataforsyningen WCS API and saving the results as GeoTIFF files in the output directory."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        cop_path = str(output_path).replace("dataforsyningen", "copernicus")
        cop_chunks = self.read_copernicus(cop_path)
        self.get_dataforsyningen(cop_chunks, output_path)

    def read_copernicus(self, file_path: str | Path) -> tuple[xr.DataArray, tuple[int, int]]:
        """Reads existing Copernicus DEM data from the specified file path, and returns a list of xarray.DataArray objects representing the individual chunks of data, along with the target resolution for the high-resolution DEM data in pixels. The function uses rioxarray to read the GeoTIFF files containing the Copernicus data, and calculates the upscale factor based on the resolution of the Copernicus data and the desired resolution for the Dataforsyningen data."
        Args:
            file_path (str | Path): The file path where the existing Copernicus DEM data is stored.
        Returns:
            tuple[list[xr.DataArray], tuple[int, int]]: A list of xarray.DataArray objects representing the individual chunks of data, along with the target resolution for the high-resolution DEM data in pixels.
        """
        chunks = []
        
        for f in list(Path(file_path).glob("*.tif")):
            with rioxarray.open_rasterio(f) as raw_data:
                data = raw_data.squeeze().drop_vars("band").load()
                chunks.append(data)

        self.upscale_factor = round(chunks[0].rio.resolution()[0] / self.meters_per_pixel)

        return chunks


    def get_dataforsyningen(self, dataforsyningen_data: xr.DataArray, output_path: Path):
        """Fetches DEM data from the Dataforsyningen WCS API based on the provided list of xarray.DataArray objects representing the required chunks of data, and saves the fetched data to the specified output path. The function iterates through the list of data chunks, making requests to the Dataforsyningen WCS API for each chunk using the appropriate parameters (including the bounding box and target resolution), and saves the resulting DEM data as GeoTIFF files in the output directory."""
        
        datatoken = os.getenv("DATATOKEN")

        for data in tqdm(dataforsyningen_data, desc="Fetching Dataforsyningen Data"):
            x_min, y_min, x_max, y_max = data.rio.bounds()
            out_file = output_path / f"dataforsyningen_{x_min:.0f}_{y_min:.0f}.tif"

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
