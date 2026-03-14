from typing import List
import pystac
import pystac_client
import planetary_computer
import rioxarray
import xarray as xr
from planetary_computer import sign
from pathlib import Path
from tqdm import tqdm

class Copernicus:
    # predetermined bounding box for the area of interest
    LON_MIN = 9.0
    LAT_MIN = 55.000277777777775
    LON_MAX = 9.999583333333334
    LAT_MAX = 57.0

    def __init__(self, target_resolution: tuple[int, int], catalog: pystac_client.Client | None = None):
        """
        Args:
            target_resolution: (height, width) of the output data.
            catalog: Client connection to STAC API
        """
        self.catalog = catalog or pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                                            modifier=planetary_computer.sign_inplace)
        self.target_resolution = target_resolution

    def get_data(self, output_path: str | Path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        items = self.search()
        data = self.merge(items)
        divided_data = self.divide(data)
        self.write(divided_data, output_path)

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
            da = rioxarray.open_rasterio(signed.href)
            rasters.append(da)
            
        merged = xr.combine_by_coords(rasters)

        return merged.squeeze().drop_vars("band")

    def divide(self, data: xr.DataArray) -> List[xr.DataArray]:
        full_height, full_width = data.shape

        data_list = []
        for h in range(0, full_height, self.target_resolution[0]):
            for w in range(0, full_width, self.target_resolution[1]):
                data_list.append(data[h:h + self.target_resolution[0],
                                          w:w + self.target_resolution[1]])
        return data_list

    def write(self, data: List[xr.DataArray], output_path: Path):
        for chunk in tqdm(data, desc="Writing chunks"):
            lon_min, lat_min, _, _ = chunk.rio.bounds()
            out_file = output_path / f"copernicus_{lon_min:.5f}_{lat_min:.5f}.tif"
            chunk.rio.to_raster(out_file)

def main():
    current_dir = Path(__file__).parent
    output_path = current_dir.parent / "data" / "copernicus"
    resolution = (512, 512)
    copernicus = Copernicus(target_resolution=resolution)
    copernicus.get_data(output_path)

if __name__ == "__main__":
    main()