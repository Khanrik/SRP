from typing import List
import pystac
import pystac_client
import planetary_computer
import rioxarray
import xarray as xr
from planetary_computer import sign
from pathlib import Path
import numpy as np
from helpers import BoundingBoxDegree, BoundingBoxMeter, DataDivision, make_folders
from pyproj import Transformer
from tqdm import tqdm


class Copernicus:
    # bounding box limits for dataforsyningen in EPSG:25832
    LIMITS = BoundingBoxMeter(x_min=441000, y_min=6049000, x_max=894000, y_max=6403000)
    COP_CHUNK_SIZE_DEG = (1, 1) # how many degrees each edge of a copernicus chunk is in (lon, lat) - used to pad extra data

    def __init__(self,
                 aoi: BoundingBoxDegree,
                 target_resolution: tuple[int, int], 
                 target_crs: str = "EPSG:25832",
                 data_division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1),
                 catalog: pystac_client.Client | None = None):
        """
        Args:
            aoi: Area of interest defined by bounding box (lon_min, lat_min, lon_max, lat_max).
            target_resolution: (height, width) of the chunks when dividing the data.
            target_crs: CRS to reproject DEM data to before chunking.
            data_division: Proportions for dividing the data into train, validation, and test sets.
            catalog: Client connection to STAC API
        """
        self.aoi = aoi
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        self.data_division = data_division
        self.catalog = catalog or pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                                            modifier=planetary_computer.sign_inplace)
        

    def get_data(self, output_path: str | Path):
        output_path = Path(output_path)
        make_folders(output_path, "LR")

        items = self.search()
        data = self.merge(items)
        data_reprojected = self.reproject(data)
        divided_data = self.divide(data_reprojected)
        self.write(divided_data, output_path)
        self.write_merge(data_reprojected, output_path, "copernicus_meter_merged.tif")


    def search(self, collection_id: str = "cop-dem-glo-30") -> List[pystac.Item]:
        search = self.catalog.search(
            collections=[collection_id],
            bbox=[self.aoi.lon_min - self.COP_CHUNK_SIZE_DEG[0], 
                  self.aoi.lat_min - self.COP_CHUNK_SIZE_DEG[1], 
                  self.aoi.lon_max + self.COP_CHUNK_SIZE_DEG[0], 
                  self.aoi.lat_max + self.COP_CHUNK_SIZE_DEG[1]],
            query={"gsd": {"eq": 30}}
        )
        items = list(search.items())
        return items
    

    def merge(self, items: List[pystac.Item]) -> xr.DataArray:
        rasters = []

        for item in items:
            signed = sign(item.assets["data"])
            with rioxarray.open_rasterio(signed.href) as data:
                rasters.append(data.squeeze().drop_vars("band").load())
            
        merged = xr.combine_by_coords(rasters)

        return merged
    
    
    def reproject(self, data: xr.DataArray) -> xr.DataArray:
        # bound til dataforsyningen limits
        transformer = Transformer.from_crs("EPSG:4326", self.target_crs, always_xy=True).transform
        aoi_meter = BoundingBoxMeter(*transformer(self.aoi.lon_min, self.aoi.lat_min), *transformer(self.aoi.lon_max, self.aoi.lat_max))
        x_min = max(self.LIMITS.x_min, aoi_meter.x_min)
        y_min = max(self.LIMITS.y_min, aoi_meter.y_min)
        x_max = min(self.LIMITS.x_max, aoi_meter.x_max)
        y_max = min(self.LIMITS.y_max, aoi_meter.y_max)

        # meter difference from center to pixel corner
        reprojected_raw = data.rio.reproject(self.target_crs)
        resolution = tuple(map(abs, reprojected_raw.rio.resolution()))
        hx, hy = np.array(resolution) / 2
        x_keep = (reprojected_raw.x - hx >= x_min) & (reprojected_raw.x + hx <= x_max)
        y_keep = (reprojected_raw.y - hy >= y_min) & (reprojected_raw.y + hy <= y_max)
        reprojected = reprojected_raw.isel(x=x_keep, y=y_keep)
        
        return reprojected
    

    def divide(self, data: xr.DataArray) -> List[xr.DataArray]:
        full_height, full_width = data.shape
        chunk_height, chunk_width = self.target_resolution
        chunks = []

        for h in range(0, full_height, chunk_height):
            for w in range(0, full_width, chunk_width):
                # reassigning h and w if the chunk would go out of bounds, to not get chunks with dimensions smaller than (chunk_height, chunk_width))
                h = min(h, full_height - chunk_height)
                w = min(w, full_width - chunk_width)
                chunks.append(data.isel(
                    y=slice(h, h + chunk_height),
                    x=slice(w, w + chunk_width),
                ))
                
        return chunks


    def write(self, data: List[xr.DataArray], output_path: Path):
        N = len(data)
        train_end = round(N * self.data_division.train) - 1
        val_end = train_end + 1 + round(N * self.data_division.val) - 1

        for i, chunk in enumerate(tqdm(data, desc="Writing chunks")):
            split = "test"
            if i <= train_end:
                split = "train"
            elif i <= val_end:
                split = "val"

            out_file = output_path / split / "LR" / f"copernicus_{chunk.rio.bounds()[0]:.0f}_{chunk.rio.bounds()[1]:.0f}.tif"

            if out_file.exists():
                continue

            chunk.rio.to_raster(out_file)


    def write_merge(self, data: xr.DataArray, output_path: Path, filename: str):
        """Writes the merged data to a single file. Used for testing and debugging."""
        out_file = output_path / filename
        
        if out_file.exists():
            return

        data.rio.to_raster(out_file)


def main():
    print("Downloading and processing Copernicus DEM data...")
    
    midtjylland = BoundingBoxDegree(lon_min=9.0, lat_min=55.000277777777775, lon_max=9.999583333333334, lat_max=57.0)
    resolution = (512, 512)
    copernicus = Copernicus(aoi=midtjylland, target_resolution=resolution)
    
    current_dir = Path(__file__).parent
    output_path = current_dir.parent / "data"
    copernicus.get_data(output_path)

    print("Done.")


if __name__ == "__main__":
    main()