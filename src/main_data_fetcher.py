from helpers import BoundingBoxDegree, DataDivision
from dataforsyningen import Dataforsyningen
from dataclasses import dataclass
from copernicus import Copernicus
from pathlib import Path
import xarray as xr
import shutil

@dataclass
class cop_data_group:
    copernicus: Copernicus = None
    divided_data: list[xr.Dataset | xr.DataArray] = None
    data_divsion: DataDivision = None

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
                     include_funen: bool,
                     include_merge: bool = False):
    # bounding boxes found with this website https://bboxfinder.com/
    west_jutland_bbox = BoundingBoxDegree(lon_min=8.047, lat_min=54.908, lon_max=9.673, lat_max=57.169)
    east_jutland_bbox = BoundingBoxDegree(lon_min=9.674, lat_min=55.618, lon_max=10.953, lat_max=57.774)
    zealand_bbox = BoundingBoxDegree(lon_min=11.019, lat_min=54.619, lon_max=12.612, lat_max=56.130)
    funen_bbox = BoundingBoxDegree(lon_min=9.689, lat_min=54.711, lon_max=10.975, lat_max=55.627)
    
    denmark_data: dict[str, cop_data_group] = {
        "west_jutland": cop_data_group(copernicus=Copernicus(aoi=west_jutland_bbox)),
        "east_jutland": cop_data_group(copernicus=Copernicus(aoi=east_jutland_bbox)),
        "zealand": cop_data_group(copernicus=Copernicus(aoi=zealand_bbox))
    }
    if include_funen:
        denmark_data["funen"] = cop_data_group(copernicus=Copernicus(aoi=funen_bbox))

    for region, group in denmark_data.items():
        print(f"Processing {region} for Copernicus...")
        divided_data, merged_data = group.copernicus.get_data(target_resolution=lr_target_resolution)
        denmark_data[region].divided_data = divided_data

        if include_merge:
            group.copernicus.write_merge(merged_data, output_path=output_path, filename=f"{region}_merged.tif")

    if include_funen:
        denmark_data["funen"].data_divsion = DataDivision(train=0, val=1, test=0)
        denmark_data["west_jutland"].data_divsion = DataDivision(train=1, val=0, test=0)
        denmark_data["east_jutland"].data_divsion = DataDivision(train=1, val=0, test=0)
    else:
        # Calculate proportions for data division so the amount of validation and test data are the same
        jutland_len = len(denmark_data["west_jutland"].divided_data) + len(denmark_data["east_jutland"].divided_data)
        zealand_len = len(denmark_data["zealand"].divided_data)
        val_proportion = zealand_len // (jutland_len + zealand_len)
        denmark_data["west_jutland"].data_divsion = DataDivision(train=1-val_proportion, val=val_proportion, test=0)
        denmark_data["east_jutland"].data_divsion = DataDivision(train=1-val_proportion, val=val_proportion, test=0)
    denmark_data["zealand"].data_divsion = DataDivision(train=0, val=0, test=1)

    for group in denmark_data.values():
        group.copernicus.write(group.divided_data, output_path=output_path, data_division=group.data_divsion)
    
    dataforsyningen = Dataforsyningen(target_resolution=hr_target_resolution)
    dataforsyningen.get_data(output_path)

def main():
    data_dir = Path(__file__).parent.parent / "data"
    lr_target_resolution = (128, 128)
    hr_target_resolution = 10

    print("Downloading and processing Denmark data without Funen...")
    get_denmark_data(output_path=data_dir / "no_fyn", 
                     lr_target_resolution=lr_target_resolution,
                     hr_target_resolution=hr_target_resolution,
                     include_funen=False,
                     include_merge=False)
    
    # funen is just used as validation data instead of a fraction of the jutland data, so we can reuse what we just downloaded
    shutil.copytree(data_dir / "no_fyn" / "train", data_dir / "with_fyn" / "train", dirs_exist_ok=True)
    shutil.copytree(data_dir / "no_fyn" / "val", data_dir / "with_fyn" / "train", dirs_exist_ok=True)
    shutil.copytree(data_dir / "no_fyn" / "test", data_dir / "with_fyn" / "test", dirs_exist_ok=True)
    
    print("Downloading and processing Denmark data with Funen...")
    get_denmark_data(output_path=data_dir / "with_fyn", 
                     lr_target_resolution=lr_target_resolution, 
                     hr_target_resolution=hr_target_resolution,
                     include_funen=True,
                     include_merge=True)
    
    # print("Downloading and processing Ethiopia data...")
    # get_ethiopia_data(output_path=data_dir / "ethiopia", 
    #                   target_resolution=lr_target_resolution)
    
    print("Done fetching data!")

if __name__ == "__main__":
    main()