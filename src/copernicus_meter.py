from typing import Iterable, List
import pystac
import pystac_client
import planetary_computer
import rioxarray
import xarray as xr
from planetary_computer import sign
from pathlib import Path
from tqdm import tqdm
import numpy as np

class Copernicus:
    DATAFORSYNINGEN_LIMITS = (441000, 6049000, 894000, 6403000)

    # predetermined bounding box for the area of interest
    LON_MIN = 9.0
    LAT_MIN = 55.000277777777775
    LON_MAX = 9.999583333333334
    LAT_MAX = 57.0

    def __init__(self, 
                 target_resolution: tuple[int, int], 
                 target_crs: str = "EPSG:25832",
                 catalog: pystac_client.Client | None = None):
        """
        Args:
            target_resolution: (height, width) of the chunks when dividing the data.
            target_crs: CRS to reproject DEM data to before chunking.
            catalog: Client connection to STAC API
        """
        self.catalog = catalog or pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                                            modifier=planetary_computer.sign_inplace)
        self.target_resolution = target_resolution
        self.target_crs = target_crs

    def get_data(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        items = self.search()
        data = self.merge(items)
        data_reprojected = self.reproject(data)
        divided_data = list(self.divide(data_reprojected))
        self.write(divided_data, output_path)

        # self.write_merge(data.rio.reproject(self.target_crs), output_path, "copernicus_meter_merged.tif")
        # self.write_merge(data, output_path, "copernicus_degree_merged.tif")

    def search(self, collection_id: str = "cop-dem-glo-30") -> List[pystac.Item]:
        search = self.catalog.search(
            collections=[collection_id],
            bbox=[self.LON_MIN, self.LAT_MIN, self.LON_MAX, self.LAT_MAX],
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
        reprojected_raw = data.rio.reproject(self.target_crs)

        # bound til dataforsyningen limits
        xmin, ymin, xmax, ymax = self.DATAFORSYNINGEN_LIMITS
        resolution = tuple(map(abs, reprojected_raw.rio.resolution()))
        
        # meter difference from center to pixel corner
        hx, hy = np.array(resolution) / 2
        x_keep = (reprojected_raw.x - hx >= xmin) & (reprojected_raw.x + hx <= xmax)
        y_keep = (reprojected_raw.y - hy >= ymin) & (reprojected_raw.y + hy <= ymax)

        reprojected = reprojected_raw.isel(x=x_keep, y=y_keep)
        
        return reprojected

    def divide(self, data: xr.DataArray) -> Iterable[xr.DataArray]:
        full_height, full_width = data.shape
        chunk_height, chunk_width = self.target_resolution

        for h in range(0, full_height, chunk_height):
            for w in range(0, full_width, chunk_width):
                # reassigning h and w if the chunk would go out of bounds, to not get chunks with dimensions smaller than (chunk_height, chunk_width))
                h = min(h, full_height - chunk_height)
                w = min(w, full_width - chunk_width)
                yield data.isel(
                    y=slice(h, h + chunk_height),
                    x=slice(w, w + chunk_width),
                )

    def write(self, data: List[xr.DataArray], output_path: Path):
        for chunk in tqdm(data, desc="Writing chunks"):
            x_min, y_min, _, _ = chunk.rio.bounds()
            out_file = output_path / f"copernicus_{x_min:.2f}_{y_min:.2f}_meter.tif"

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
    current_dir = Path(__file__).parent
    output_path = current_dir.parent / "data" / "copernicus"
    resolution = (512, 512)
    copernicus = Copernicus(target_resolution=resolution)
    copernicus.get_data(output_path)
    print("Done.")

if __name__ == "__main__":
    main()