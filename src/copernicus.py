from __future__ import annotations
from typing import List
import pystac
import pystac_client
import planetary_computer
import rioxarray
import xarray as xr
from planetary_computer import sign
from pathlib import Path
import numpy as np
from data_distributor import BoundingBoxDegree, BoundingBoxMeter
from pyproj import Transformer
from tqdm import tqdm


class Copernicus:
    """Class for fetching and processing DEM data from the Copernicus dataset using the Microsoft Planetary Computer STAC API. The class allows for searching, merging, reprojecting, dividing, and writing DEM data based on a specified area of interest (AOI) and target resolution. It also includes functionality to accommodate for the limits of the Dataforsyningen dataset when reprojecting the data.
    Args:
        aoi (BoundingBoxDegree): Area of interest defined by bounding box (lon_min, lat_min, lon_max, lat_max).
        target_crs (str): CRS to reproject DEM data to before chunking.
        dataforsyningen (bool): Whether to accomodate for dataforsyningen limits.
    """
    # bounding box limits for dataforsyningen in EPSG:25832
    LIMITS = BoundingBoxMeter(x_min=441000, y_min=6049000, x_max=894000, y_max=6403000)

    # how many degrees each edge of a copernicus chunk is in (lon, lat) - used to pad extra data
    COP_CHUNK_SIZE_DEG = (1, 1)

    # sum of pixel values in a chunk must be above this to be included to filter out chunks with mostly water
    WATER_THRESHOLD = 1000 

    def __init__(self,
                 aoi: BoundingBoxDegree, 
                 target_crs: str = "EPSG:25832",
                 dataforsyningen: bool = True):
        """
        Args:
            aoi: Area of interest defined by bounding box (lon_min, lat_min, lon_max, lat_max).
            target_crs: CRS to reproject DEM data to before chunking.
            dataforsyningen: Whether to accomodate for dataforsyningen limits.
        """
        self.aoi = aoi
        self.target_crs = target_crs
        self.dataforsyningen = dataforsyningen

    def get_data(self, target_resolution: tuple[int, int]) -> tuple[List[xr.DataArray], xr.DataArray]:
        """main function to get data from copernicus
        Args:
            target_resolution (tuple[int, int]): The desired resolution of the output chunks in pixels (height, width)

        Returns:
            divided_data,merged_data:
            List of DEM chunks for area of interest, and full DEM before chunking.
        """
        items = self.search()
        merged_data = self.merge(items)
        
        if self.target_crs != str.join(":", merged_data.rio.crs.to_authority()):
            merged_data = self.reproject(merged_data)

        divided_data = self.divide(merged_data, target_resolution)

        return divided_data, merged_data


    def search(self, collection_id: str = "cop-dem-glo-30") -> List[pystac.Item]:
        """Searches the Microsoft Planetary Computer STAC API for items in the specified collection that intersect with the area of interest (AOI) and have a ground sample distance (GSD) of 30 meters. The search results are returned as a list of pystac.Item objects.
        Args:
            collection_id (str, optional): The ID of the collection to search within the STAC API. Defaults to "cop-dem-glo-30".
        Returns:
            List[pystac.Item]: A list of pystac.Item objects that match the search criteria.
        """
        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier=planetary_computer.sign_inplace)
        search = catalog.search(
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
        """Merges the DEM data from the provided list of pystac.Item objects into a single xarray.DataArray. Each item is accessed using its signed URL, and the resulting rasters are combined using xarray's combine_by_coords function with an outer join.
        Args:
            items (List[pystac.Item]): A list of pystac.Item objects containing DEM data.
        Returns:
            xr.DataArray: A single xarray.DataArray containing the merged DEM data.
        """
        rasters = []

        for item in tqdm(items, desc="Collecting chunks to merge"):
            signed = sign(item.assets["data"])
            with rioxarray.open_rasterio(signed.href) as data:
                rasters.append(data.squeeze().drop_vars("band").load())
        
        print("Merging chunks...")
        merged = xr.combine_by_coords(rasters, join="outer")

        return merged
    
    
    def reproject(self, data: xr.DataArray) -> xr.DataArray:
        """Reprojects the provided xarray.DataArray to the target CRS specified in the class instance, while also accommodating for the limits of the Dataforsyningen dataset. The function first reprojects the data to the target CRS, then filters out any areas that fall outside the bounds of the Dataforsyningen dataset (if applicable) to ensure that the resulting data is suitable for comparison with Dataforsyningen data.
        Args:
            data (xr.DataArray): The input xarray.DataArray to be reprojected.
        Returns:
            xr.DataArray: The reprojected xarray.DataArray.
        """
        # bound til dataforsyningen limits
        transformer = Transformer.from_crs("EPSG:4326", self.target_crs, always_xy=True).transform
        aoi_meter = BoundingBoxMeter(*transformer(self.aoi.lon_min, self.aoi.lat_min), *transformer(self.aoi.lon_max, self.aoi.lat_max))
        x_min = max(self.LIMITS.x_min, aoi_meter.x_min) if self.dataforsyningen else aoi_meter.x_min
        y_min = max(self.LIMITS.y_min, aoi_meter.y_min) if self.dataforsyningen else aoi_meter.y_min
        x_max = min(self.LIMITS.x_max, aoi_meter.x_max) if self.dataforsyningen else aoi_meter.x_max
        y_max = min(self.LIMITS.y_max, aoi_meter.y_max) if self.dataforsyningen else aoi_meter.y_max

        # meter difference from center to pixel corner
        reprojected_raw = data.rio.reproject(self.target_crs)
        resolution = tuple(map(abs, reprojected_raw.rio.resolution()))
        hx, hy = np.array(resolution) / 2
        x_keep = (reprojected_raw.x - hx >= x_min) & (reprojected_raw.x + hx <= x_max)
        y_keep = (reprojected_raw.y - hy >= y_min) & (reprojected_raw.y + hy <= y_max)
        reprojected = reprojected_raw.isel(x=x_keep, y=y_keep)
        
        return reprojected
    

    def divide(self, data: xr.DataArray, target_resolution: tuple[int, int]) -> List[xr.DataArray]:
        """Divides the provided xarray.DataArray into smaller chunks based on the specified target resolution in pixels. The function iterates through the data in steps of the target resolution, creating chunks of the specified size. It also filters out any chunks that have a sum of pixel values below a certain threshold, which is used to exclude areas that are likely to be water (i.e., have very low elevation values).
        Args:
            data (xr.DataArray): The input xarray.DataArray to be divided.
            target_resolution (tuple[int, int]): The desired resolution of the output chunks in pixels (height, width).
        Returns:
            List[xr.DataArray]: A list of xarray.DataArray objects representing the divided chunks.
        """
        full_height, full_width = data.shape
        chunk_height, chunk_width = target_resolution
        chunks = []

        for h in range(0, full_height, chunk_height):
            for w in range(0, full_width, chunk_width):
                # reassigning h and w if the chunk would go out of bounds, to not get chunks with dimensions smaller than (chunk_height, chunk_width))
                h = min(h, full_height - chunk_height)
                w = min(w, full_width - chunk_width)
                chunk = data.isel(
                    y=slice(h, h + chunk_height),
                    x=slice(w, w + chunk_width),
                )

                # filter out chunks with very little data (likely water)
                if np.sum(chunk.values) < self.WATER_THRESHOLD:
                    continue

                chunks.append(chunk)
                
        return chunks


    def write(self, data: List[xr.DataArray], output_path: Path):
        """Writes the divided data to separate files."""
        output_path.mkdir(parents=True, exist_ok=True)

        for chunk in tqdm(data, desc="Writing chunks"):
            out_file = output_path / f"copernicus_{chunk.rio.bounds()[0]:.0f}_{chunk.rio.bounds()[1]:.0f}.tif"

            if out_file.exists():
                continue

            chunk.rio.to_raster(out_file)


    def write_merge(self, data: xr.DataArray, output_path: Path, filename: str):
        """Writes the merged data to a single file."""
        output_path.mkdir(parents=True, exist_ok=True)

        out_file = output_path / filename
        
        if out_file.exists():
            return

        data.rio.to_raster(out_file)