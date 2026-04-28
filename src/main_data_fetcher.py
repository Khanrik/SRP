from __future__ import annotations
from helpers import BoundingBoxDegree
from dataforsyningen import Dataforsyningen
from dataclasses import dataclass
from copernicus import Copernicus
from pathlib import Path
import xarray as xr
import shutil
import os
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
    bornholm_bbox = BoundingBoxDegree(lon_min=14.674, lat_min=54.982, lon_max=15.180, lat_max=55.305)
    
    denmark_data: dict[str, cop_data_group] = {
        "west_jutland": cop_data_group(copernicus=Copernicus(aoi=west_jutland_bbox)),
        "east_jutland": cop_data_group(copernicus=Copernicus(aoi=east_jutland_bbox)),
        "funen": cop_data_group(copernicus=Copernicus(aoi=funen_bbox)),
        "zealand": cop_data_group(copernicus=Copernicus(aoi=zealand_bbox)),
        "bornholm": cop_data_group(copernicus=Copernicus(aoi=bornholm_bbox))
    }

    for region in denmark_data.keys():
        # TODO: slet den her check igen efter bornholm er downloadet
        if region != "bornholm":
            continue
        print(f"Processing {region} for Copernicus...")
        divided_data, merged_data = denmark_data[region].copernicus.get_data(target_resolution=lr_target_resolution)
        denmark_data[region].divided_data = divided_data

        if include_merge:
            denmark_data[region].copernicus.write_merge(merged_data, output_path=output_path, filename=f"{region}_merged.tif")

        region_name = "jutland" if "jutland" in region else region

        denmark_data[region].copernicus.write(divided_data, output_path=output_path / "copernicus" / region_name)
        dataforsyningen.get_data(output_path / "dataforsyningen" / region_name)

def move_to_selected(output_path: Path):
    lr_file_path = output_path / "selected" / "lr"
    hr_file_path = output_path / "selected" / "hr"
    lr_file_path.parent.mkdir(parents=True, exist_ok=True)
    hr_file_path.parent.mkdir(parents=True, exist_ok=True)

    files_to_move = {
        "jutland": [
            "572995_6223578", # aarhus
            "471890_6212915", # herborg
            "550091_6200674", # møllehøj
        ],
        "zealand": [
            "720615_6172713", # københavn
            "684063_6126692", # paradishaven
            "695568_6126692", # faxe kalkbrud
            "710908_6130528", # random sjælland mark
        ],
        "bornholm": [
            "874677_6120476", # lilleborg
            "863435_6120476", # rønne
            "882171_6112982", # Ringborgen Rispebjerg (random bornholm mark)
        ]
    }

    for region, coords_list in files_to_move.items():
        for coords in coords_list:
            lr_file = output_path / "copernicus" / region / f"copernicus_{coords}.tif"
            hr_file = output_path / "dataforsyningen" / region / f"dataforsyningen_{coords}.tif"
            
            if not lr_file.exists() or not hr_file.exists():
                print(f"Warning: Missing file for {region} with coords {coords}. LR exists: {lr_file.exists()}, HR exists: {hr_file.exists()}")
                continue
            
            shutil.move(str(lr_file), str(lr_file_path))
            shutil.move(str(hr_file), str(hr_file_path))

            if lr_file.exists():
                os.remove(lr_file)
            if hr_file.exists():
                os.remove(hr_file)

def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    lr_target_resolution = (128, 128)
    hr_target_resolution = 10
    
    print("Downloading and processing Denmark data...")
    get_denmark_data(output_path=data_dir, 
                     lr_target_resolution=lr_target_resolution, 
                     hr_target_resolution=hr_target_resolution,
                     include_merge=True)
    
    print("Moving select files to 'selected' directory...")
    move_to_selected(output_path=data_dir)
    
    # print("Downloading and processing Ethiopia data...")
    # get_ethiopia_data(output_path=data_dir / "ethiopia", 
    #                   target_resolution=lr_target_resolution)
    
    print("Done fetching data!")

if __name__ == "__main__":
    main()