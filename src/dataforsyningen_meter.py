import rioxarray
import xarray as xr
from pathlib import Path
import requests
from dotenv import load_dotenv
import os
import numpy as np
from helpers import DataDivision, make_folders, divide, write

class Dataforsyningen:
    URL = "https://api.dataforsyningen.dk/dhm_wcs_DAF"

    def __init__(self, 
                 target_resolution: int,
                 data_division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1)):
        """
        Args:
            target_resolution: The desired resolution of the DEM data in meters per pixel.
            data_division: Proportions for dividing the data into train, validation, and test sets.
        """
        self.meters_per_pixel = target_resolution
        self.data_division = data_division
        self.upscale_factor = None
        load_dotenv()

    def get_data(self, output_path: str | Path):
        output_path = Path(output_path)
        make_folders(output_path, "HR")

        cop_merged, cop_chunk_shape = self.read_copernicus(output_path)
        dataforsyningen_merged = self.get_dataforsyningen(cop_merged, output_path, "dataforsyningen_meter_merged.tif")
        target_chunk_shape = np.array(cop_chunk_shape) * (round(dataforsyningen_merged.rio.resolution()[0]) // self.meters_per_pixel)
        divided_data = divide(dataforsyningen_merged, target_chunk_shape)
        write(data=divided_data, 
              output_path=output_path, 
              resolution="HR", 
              dataset="dataforsyningen", 
              unit="meter", 
              data_division=self.data_division)
        

    def read_copernicus(self, file_path: str | Path) -> tuple[xr.DataArray, tuple[int, int]]:
        with rioxarray.open_rasterio(file_path / "copernicus_meter_merged.tif") as raw_data:
            merged_data = raw_data.squeeze().drop_vars("band").load()

        cop_files = Path(file_path / "train" / "LR").glob("copernicus_*_meter.tif")
        with rioxarray.open_rasterio(next(cop_files)) as raw_data:
            data = raw_data.squeeze().drop_vars("band").load()
            chunk_shape = data.shape

        return merged_data, chunk_shape


    def get_dataforsyningen(self, lr_data: xr.DataArray, output_path: Path, filename: str) -> xr.DataArray:
        out_file = output_path / filename

        if not out_file.exists():
            self.upscale_factor = round(lr_data.rio.resolution()[0]) // self.meters_per_pixel
            target_height, target_width = np.array(lr_data.shape) * self.upscale_factor

            params = {
                "service": "WCS",
                "version": "1.0.0",
                "request": "GetCoverage",
                "coverage": "dhm_overflade",
                "crs": "EPSG:25832",
                "bbox": str.join(",", map(str, lr_data.rio.bounds())),
                "height": f"{target_height}",
                "width": f"{target_width}",
                "format": "GTiff",
                "token": os.getenv("DATATOKEN")
            }

            response = requests.get(self.URL, params=params, timeout=60)
            response.raise_for_status()

            with open(str(out_file), "wb") as f:
                f.write(response.content)
        
        with rioxarray.open_rasterio(out_file) as data:
            data = data.squeeze().drop_vars("band").load()

        return data
    

def main():
    current_dir = Path(__file__).parent
    output_path = current_dir.parent / "data"
    dem_resolution = 10
    dataforsyningen = Dataforsyningen(target_resolution=dem_resolution)
    dataforsyningen.get_data(output_path)

if __name__ == "__main__":
    main()