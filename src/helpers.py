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