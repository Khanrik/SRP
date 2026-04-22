from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from helpers import DataDivision
from random import Random

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

@dataclass
class DataPair:
    lr: Path
    hr: Path

@dataclass
class DataSplits:
    """Container for training, validation, and test splits of the dataset.

    DataPair objects contain lr and hr file paths.
    
    Attributes
    ---------
        train : list[DataPair]
            Data for training.
        val : list[DataPair]
            Data for validation.
        test : list[DataPair]
            Data for testing.
        dataset : list[DataPair]
            The entire dataset (sum of all splits).
    """
    train: list[DataPair]
    val: list[DataPair]
    test: list[DataPair]
    
    @property
    def dataset(self):
        return self.train + self.val + self.test

def _pair_files(lr_data_dir_list: list[Path],
                hr_data_dir_list: list[Path]) -> list[DataPair]:
    lr_files = [f for directory in lr_data_dir_list for f in directory.glob("*.tif")]
    hr_files = [f for directory in hr_data_dir_list for f in directory.glob("*.tif")]

    hr_by_key: dict[str, Path] = {}
    for hr_file in hr_files:
        key = "_".join(hr_file.stem.split("_")[-2:])
        hr_by_key[key] = hr_file

    pairs: list[DataPair] = []
    for lr_file in lr_files:
        key = "_".join(lr_file.stem.split("_")[-2:])
        hr_file = hr_by_key[key]
        pairs.append(DataPair(lr=lr_file, hr=hr_file))

    return pairs

def get_base_dataset(lr_data_dir_list: list[Path],
                     hr_data_dir_list: list[Path],
                     division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1)) -> DataSplits:
    all_pairs = _pair_files(lr_data_dir_list, hr_data_dir_list)
    Random().shuffle(all_pairs)
    N = len(all_pairs)

    train_count = int(N * division.train)
    val_count = int(N * division.val)
    train_end = train_count
    val_end = train_count + val_count
    
    data_splits = DataSplits(
        train=all_pairs[:train_end], 
        val=all_pairs[train_end:val_end], 
        test=all_pairs[val_end:]
    )

    return data_splits

def get_augmented_dataset(lr_data_dir_list: list[Path],
                          hr_data_dir_list: list[Path],
                          division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1)) -> DataSplits:
    pass

if __name__ == "__main__":
    lr_dirs = [DATA_DIR / "copernicus" / region for region in ["jutland", "funen"]]
    hr_dirs = [DATA_DIR / "dataforsyningen" / region for region in ["jutland", "funen"]]
    
    division = DataDivision(train=0.8, val=0.1, test=0.1)
    data_splits = get_base_dataset(lr_dirs, hr_dirs, division=division)

    print(f"Train: {len(data_splits.train)} pairs, Val: {len(data_splits.val)} pairs, Test: {len(data_splits.test)} pairs, Total: {len(data_splits.dataset)} pairs")