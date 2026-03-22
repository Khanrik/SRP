from typing import Iterable, List
import pystac
import pystac_client
import planetary_computer
import rioxarray
import xarray as xr
from planetary_computer import sign
from pathlib import Path
from tqdm import tqdm
from helpers import BoundingBoxDegree

class Copernicus:
    def __init__(self,
                 aoi: BoundingBoxDegree,
                 target_resolution: tuple[int, int], 
                 catalog: pystac_client.Client | None = None):
        """
        Args:
            aoi: Area of interest defined by bounding box (lon_min, lat_min, lon_max, lat_max).
            target_resolution: (height, width) of the chunks when dividing the data.
            catalog: Client connection to STAC API
        """
        self.aoi = aoi
        self.catalog = catalog or pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                                            modifier=planetary_computer.sign_inplace)
        self.target_resolution = target_resolution

    def get_data(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        items = self.search()
        data = self.merge(items)
        divided_data = list(self.divide(data))
        self.write(divided_data, output_path)

    def search(self, collection_id: str = "cop-dem-glo-30") -> List[pystac.Item]:
        search = self.catalog.search(
            collections=[collection_id],
            bbox=[self.aoi.lon_min, self.aoi.lat_min, self.aoi.lon_max, self.aoi.lat_max],
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
            lon_min, lat_min, _, _ = chunk.rio.bounds()
            out_file = output_path / f"copernicus_{lon_min:.5f}_{lat_min:.5f}_degree.tif"
            
            if out_file.exists():
                continue

            chunk.rio.to_raster(out_file)

def main():
    print("Downloading and processing Copernicus DEM data...")
    
    midtjylland = BoundingBoxDegree(lon_min=9.0, lat_min=55.000277777777775, lon_max=9.999583333333334, lat_max=57.0)
    resolution = (512, 512)
    copernicus = Copernicus(aoi=midtjylland, target_resolution=resolution)
    
    current_dir = Path(__file__).parent
    output_path = current_dir.parent / "data" / "copernicus"
    copernicus.get_data(output_path)

    print("Done.")

if __name__ == "__main__":
    main()