from dataclasses import dataclass
from pathlib import Path
from typing import Literal, List
import xarray as xr
from tqdm import tqdm

@dataclass
class BoundingBoxDegree:
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float

@dataclass
class BoundingBoxMeter:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

@dataclass
class DataDivision:
    train: float
    val: float
    test: float

    def __post_init__(self):
        total = sum((self.train, self.val, self.test))
        if not abs(total - 1.0) < 1e-6:
            raise ValueError("The sum of train, val, and test proportions must equal 1.")

def make_folders(output_path: Path, resolution: Literal["LR", "HR"]):
    for split in ["train", "val", "test"]:
        (output_path / split / resolution).mkdir(parents=True, exist_ok=True)

def divide(data: xr.DataArray, chunk_shape: tuple[int, int]) -> List[xr.DataArray]:
    full_height, full_width = data.shape
    chunk_height, chunk_width = chunk_shape
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

def write(data: List[xr.DataArray],
          output_path: Path,
          data_division: DataDivision,
          resolution: Literal["LR", "HR"],
          dataset: str,
          unit: str):
    N = len(data)
    train_end = round(N * data_division.train) - 1
    val_end = train_end + 1 + round(N * data_division.val) - 1

    for i, chunk in enumerate(tqdm(data, desc="Writing chunks")):
        split = "test"
        if i <= train_end:
            split = "train"
        elif i <= val_end:
            split = "val"

        out_file = output_path / split / resolution / f"{dataset}_{i}_{unit}.tif"

        if out_file.exists():
            continue

        chunk.rio.to_raster(out_file)