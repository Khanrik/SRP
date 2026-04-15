from __future__ import annotations
from helpers import BoundingBoxDegree
from dataforsyningen import Dataforsyningen
from dataclasses import dataclass
from copernicus import Copernicus
from pathlib import Path
import xarray as xr

@dataclass
class cop_data_group:
    copernicus: Copernicus = None
    divided_data: list[xr.Dataset | xr.DataArray] = None

def get_ethiopia_data(output_path: Path,
                      target_resolution: tuple[int, int]):
    # ethiopia_bbox = what area???
    return
    ethiopia = Copernicus(aoi=ethiopia_bbox)
    divided_data, merged_data = ethiopia.get_data(target_resolution=target_resolution)
    ethiopia.write(divided_data, output_path=output_path / "LR", data_division=DataDivision(no_division=True))
    ethiopia.write_merge(merged_data, output_path=output_path, filename="ethiopia_merged.tif")

def get_denmark_data(output_path: Path, 
                     lr_target_resolution: tuple[int, int],
                     hr_target_resolution: int,
                     include_merge: bool = False):
    # bounding boxes found with this website https://bboxfinder.com/
    dataforsyningen = Dataforsyningen(target_resolution=hr_target_resolution)
    west_jutland_bbox = BoundingBoxDegree(lon_min=8.047, lat_min=54.908, lon_max=9.673, lat_max=57.169)
    east_jutland_bbox = BoundingBoxDegree(lon_min=9.674, lat_min=55.618, lon_max=10.953, lat_max=57.774)
    funen_bbox = BoundingBoxDegree(lon_min=9.689, lat_min=54.711, lon_max=10.975, lat_max=55.627)
    zealand_bbox = BoundingBoxDegree(lon_min=11.019, lat_min=54.619, lon_max=12.612, lat_max=56.130)
    
    denmark_data: dict[str, cop_data_group] = {
        "west_jutland": cop_data_group(copernicus=Copernicus(aoi=west_jutland_bbox)),
        "east_jutland": cop_data_group(copernicus=Copernicus(aoi=east_jutland_bbox)),
        "funen": cop_data_group(copernicus=Copernicus(aoi=funen_bbox)),
        "zealand": cop_data_group(copernicus=Copernicus(aoi=zealand_bbox))
    }

    for region in denmark_data.keys():
        print(f"Processing {region} for Copernicus...")
        divided_data, merged_data = denmark_data[region].copernicus.get_data(target_resolution=lr_target_resolution)
        denmark_data[region].divided_data = divided_data

        if include_merge:
            denmark_data[region].copernicus.write_merge(merged_data, output_path=output_path, filename=f"{region}_merged.tif")

        region_name = "jutland" if "jutland" in region else region

        denmark_data[region].copernicus.write(divided_data, output_path=output_path / "copernicus" / region_name)
        dataforsyningen.get_data(output_path / "dataforsyningen" / region_name)

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    lr_target_resolution = (128, 128)
    hr_target_resolution = 10
    
    print("Downloading and processing Denmark data...")
    get_denmark_data(output_path=data_dir, 
                     lr_target_resolution=lr_target_resolution, 
                     hr_target_resolution=hr_target_resolution,
                     include_merge=True)
    
    # print("Downloading and processing Ethiopia data...")
    # get_ethiopia_data(output_path=data_dir / "ethiopia", 
    #                   target_resolution=lr_target_resolution)
    
    print("Done fetching data!")

if __name__ == "__main__":
    main()