from __future__ import annotations
from data_distributor import BoundingBoxDegree
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
                      target_resolution: tuple[int, int],
                      include_merge: bool = False):
    ethiopia_bbox = BoundingBoxDegree(lon_min=38.2788857234929907, lat_min=6.8128541402981115, lon_max=38.7260934049450327, lat_max=7.2317745888872231)
    ethiopia = Copernicus(aoi=ethiopia_bbox, target_crs="EPSG:20138", dataforsyningen=False) # UTM zone 38N - meter based Ethiopia CRS
    divided_data, merged_data = ethiopia.get_data(target_resolution=target_resolution)

    if include_merge:
        ethiopia.write_merge(merged_data, output_path=output_path, filename=f"ethiopia_merged.tif")

    ethiopia.write(divided_data, output_path=output_path / "ethiopia")

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
        print(f"Processing {region} for Copernicus...")
        divided_data, merged_data = denmark_data[region].copernicus.get_data(target_resolution=lr_target_resolution)
        denmark_data[region].divided_data = divided_data

        if include_merge:
            denmark_data[region].copernicus.write_merge(merged_data, output_path=output_path, filename=f"{region}_merged.tif")

        region_name = "jutland" if "jutland" in region else region

        denmark_data[region].copernicus.write(divided_data, output_path=output_path / "copernicus" / region_name)
        dataforsyningen.get_data(output_path / "dataforsyningen" / region_name)

def move_to_selected(output_path: Path):
    files_to_move = {
        "jutland": [
            "572995_6223578", # Aarhus
            "471890_6212915", # Herborg
            "550091_6200674", # Møllehøj
        ],
        "zealand": [
            "720615_6172713", # København
            "695568_6126692", # Faxe kalkbrud
            "710908_6130528", # tilfældig Sjælland mark
        ],
        "bornholm": [
            "874344_6123316", # Lilleborg
            "862935_6119513", # Rønne
            "885754_6111907", # Vibegård (tilfældig Bornholm mark)
        ],
        "ethiopia": [
            "-212769_784181", # near Honse, flat area with water
            "-204859_764408", # mountainous area near Hogiso
            "-236496_764408", # lightly populous area near Yirba
        ]
    }

    for region, coords_list in files_to_move.items():
        lr_file_path = output_path / "selected" / "lr" / region
        hr_file_path = output_path / "selected" / "hr" / region
        lr_file_path.mkdir(parents=True, exist_ok=True)
        hr_file_path.mkdir(parents=True, exist_ok=True)

        for coords in coords_list:
            if region != "ethiopia":
                lr_file = output_path / "copernicus" / region / f"copernicus_{coords}.tif"
                hr_file = output_path / "dataforsyningen" / region / f"dataforsyningen_{coords}.tif"

                if (not (lr_file.exists() or not hr_file.exists()) and (lr_file_path.joinpath(f"copernicus_{coords}.tif").exists() and hr_file_path.joinpath(f"dataforsyningen_{coords}.tif").exists())):
                    print(f"Warning: Missing file for {region} with coords {coords}. LR exists: {lr_file.exists()}, HR exists: {hr_file.exists()}")
                    continue
                
                if not lr_file_path.joinpath(f"copernicus_{coords}.tif").exists():
                    shutil.move(str(lr_file), str(lr_file_path / f"copernicus_{coords}.tif"))
                if not hr_file_path.joinpath(f"dataforsyningen_{coords}.tif").exists():
                    shutil.move(str(hr_file), str(hr_file_path / f"dataforsyningen_{coords}.tif"))
            else:
                lr_file = output_path / "ethiopia" / f"copernicus_{coords}.tif"

                if not lr_file.exists() and (lr_file_path.joinpath(f"copernicus_{coords}.tif").exists() and hr_file_path.joinpath(f"dataforsyningen_{coords}.tif").exists()):
                    print(f"Warning: Missing Ethiopia file for {region} with coords {coords}.")
                    continue
                
                # as there is no HR data for Ethiopia, we will just move the LR file to both the LR and HR folders in the selected directory, so that it can be used for visualization and testing with the understanding that the GT is the same as the LR.
                if not lr_file_path.joinpath(f"copernicus_{coords}.tif").exists():
                    shutil.copy(str(lr_file), str(lr_file_path / f"copernicus_{coords}.tif"))
                if not hr_file_path.joinpath(f"dataforsyningen_{coords}.tif").exists():
                    shutil.copy(str(lr_file), str(hr_file_path / f"dataforsyningen_{coords}.tif"))

