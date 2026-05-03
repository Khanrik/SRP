from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from helpers import DataDivision, DatasetInterface
from random import Random
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        train : DatasetInterface
            Data for training.
        val : DatasetInterface
            Data for validation.
        test : DatasetInterface
            Data for testing.
        dataset : DatasetInterface
            The entire dataset (sum of all splits).
    """
    train: DatasetInterface
    val: DatasetInterface
    test: DatasetInterface

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

def get_augmented_dataset(lr_data_dir_list: list[Path],
                          hr_data_dir_list: list[Path],
                          division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1)) -> DataSplits:
    pass



def prepare_dataloader(dataset, batch_size, pin_memory, num_workers=0, shuffle_bool=True):
    # This function initializes the iterable over the datasets for training, validation, and testing 
    # with the specified batch size and the Trainer's num_workers and pin_memory settings.
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        batch_size=batch_size,
        persistent_workers=False,
        shuffle=shuffle_bool,
    )
    
    return dataloader

def get_base_dataset(lr_data_dir_list: list[Path],
                     hr_data_dir_list: list[Path],
                     batch_size: int,
                     cuda: bool,
                     division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1),
                     randomize: bool = True) :
    if division.train + division.val + division.test != 1.0:
        raise ValueError(f"Data division proportions must sum to 1.0. Got {division.train} + {division.val} + {division.test} = {division.train + division.val + division.test}")
    all_pairs = _pair_files(lr_data_dir_list, hr_data_dir_list)
    if randomize:
        Random().shuffle(all_pairs)
    N = len(all_pairs)

    train_count = int(N * division.train)
    val_count = int(N * division.val)
    train_end = train_count
    val_end = train_count + val_count
    
    data_splits = DataSplits(
        train=DatasetInterface(all_pairs[:train_end], loading_description="Loading training data"),
        val=DatasetInterface(all_pairs[train_end:val_end], loading_description="Loading validation data"),
        test=DatasetInterface(all_pairs[val_end:], loading_description="Loading test data")
    )

    train_dataset, val_dataset, test_dataset = data_splits.train, data_splits.val, data_splits.test

    if train_count != 0:
        train_dataloader = prepare_dataloader(train_dataset, batch_size, cuda)
    else:
        train_dataloader = None
    if val_count != 0:
        val_dataloader = prepare_dataloader(val_dataset, batch_size, cuda)
    else:
        val_dataloader = None
    test_dataloader = prepare_dataloader(test_dataset, batch_size, cuda)
    
    min_pixel_value, max_pixel_value = compute_extremal_pixel_value(train_dataset, batch_size)

    if ((train_dataloader is None or val_dataloader is None) and test_dataloader is None):
        raise ValueError(
            f"Dataloaders not properly initialized. Train samples: {len(train_dataset)}, "
            f"Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
        )

    if division.train!= 0 and division.val != 0 and division.test != 0:
        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
            raise ValueError(
                f"Empty dataset split detected. Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
            )

        num_train_batches = len(train_dataloader)
        num_val_batches = len(val_dataloader)
        if num_train_batches == 0 or num_val_batches == 0:
            raise ValueError(
                f"Empty dataloader detected. Train batches: {num_train_batches}, "
                f"Val batches: {num_val_batches}"
            )
    
    return [train_dataloader, val_dataloader, test_dataloader, min_pixel_value, max_pixel_value]


def compute_extremal_pixel_value(dataset: DatasetInterface, batch_size: int) -> tuple[float, float]:
    """Computes the minimum and maximum pixel values across the entire dataset
    
    Returns:
        (min_pixel_value, max_pixel_value): The minimum and maximum pixel values found in the dataset
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    min_pixel_value = float('inf')
    max_pixel_value = float('-inf')
    for lr, hr in tqdm(loader, desc="Computing extremal pixel values"):
        relative_min_pixel_value = min(lr.min().item(), hr.min().item())
        relative_max_pixel_value = max(lr.max().item(), hr.max().item())
        min_pixel_value = min(min_pixel_value, relative_min_pixel_value)
        max_pixel_value = max(max_pixel_value, relative_max_pixel_value)

    return min_pixel_value, max_pixel_value




if __name__ == "__main__":
    lr_dirs = [DATA_DIR / "copernicus" / region for region in ["jutland", "funen"]]
    hr_dirs = [DATA_DIR / "dataforsyningen" / region for region in ["jutland", "funen"]]
    
    division = DataDivision(train=0.8, val=0.1, test=0.1)
    data_splits = get_base_dataset(lr_dirs, hr_dirs, division=division)

    print(f"Train: {len(data_splits.train)} pairs, Val: {len(data_splits.val)} pairs, Test: {len(data_splits.test)} pairs, Total: {len(data_splits.dataset)} pairs")
