from dataclasses import dataclass
from pathlib import Path
from helpers import DataDivision
from random import Random
import rioxarray

DATA_DIR = Path(__file__).parent.parent / "data"

@dataclass
class DataPair:
    lr: rioxarray.DataArray
    hr: rioxarray.DataArray

def _pair_files(lr_data_dir_list: list[Path],
                hr_data_dir_list: list[Path]) -> list[tuple[Path, Path]]:
    lr_files = [f for directory in lr_data_dir_list for f in directory.glob("*.tif")]
    hr_files = [f for directory in hr_data_dir_list for f in directory.glob("*.tif")]

    hr_by_key: dict[str, Path] = {}
    for hr_file in hr_files:
        key = "_".join(hr_file.stem.split("_")[-2:])
        hr_by_key[key] = hr_file

    pairs: list[tuple[Path, Path]] = []
    for lr_file in lr_files:
        key = "_".join(lr_file.stem.split("_")[-2:])
        hr_file = hr_by_key[key]
        pairs.append((lr_file, hr_file))

    return pairs

def _load_pairs(pairs: list[tuple[Path, Path]]) -> list[DataPair]:
    data_pairs = []
    for lr_file, hr_file in pairs:
        with rioxarray.open_rasterio(lr_file) as raw_lr:
            lr_data = raw_lr.squeeze().drop_vars("band").load()
        with rioxarray.open_rasterio(hr_file) as raw_hr:
            hr_data = raw_hr.squeeze().drop_vars("band").load()
        data_pairs.append(DataPair(lr=lr_data, hr=hr_data))

    return data_pairs

def get_base_dataset(lr_data_dir_list: list[Path],
                     hr_data_dir_list: list[Path],
                     division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1)):
    all_pairs = _pair_files(lr_data_dir_list, hr_data_dir_list)
    N = len(all_pairs)

    Random().shuffle(all_pairs)

    train_count = int(N * division.train)
    val_count = int(N * division.val)

    train_end = train_count
    val_end = train_count + val_count

    train = (all_pairs[:train_end], [])
    val = (all_pairs[train_end:val_end], [])
    test = (all_pairs[val_end:], [])

    for pairs in [train, val, test]:
        pairs[1].extend(_load_pairs(pairs[0]))

    return train[1], val[1], test[1]

def get_augmented_dataset(lr_data_dir_list: list[Path],
                          hr_data_dir_list: list[Path],
                          division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1)):
    pass

if __name__ == "__main__":
    lr_dirs = [DATA_DIR / "copernicus" / region for region in ["jutland", "funen"]]
    hr_dirs = [DATA_DIR / "dataforsyningen" / region for region in ["jutland", "funen"]]
    
    division = DataDivision(train=0.8, val=0.1, test=0.1)
    train, val, test = get_base_dataset(lr_dirs, hr_dirs, division=division)

    print(f"Train: {len(train)} pairs, Val: {len(val)} pairs, Test: {len(test)} pairs")