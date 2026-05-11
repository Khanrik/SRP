from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from helpers import DataDivision, DatasetInterface
from random import Random
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

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
    
    min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value = compute_extremal_pixel_value(train_dataset, batch_size) if train_dataloader is not None else (0.0, 0.0, 0.0, 0.0)

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
    
    return [train_dataloader, val_dataloader, test_dataloader, min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value]


def filter_min_outliers(data, z_threshold=3):
    """Filters out lower-tail outliers from the data using the Z-score method.
    
    Args:
        data: A list of numerical values.
        z_threshold: The Z-score threshold to identify outliers. Default is 3.
    
    Returns:
        A list of values with outliers removed and the amount of values disregarded.
    """
    if len(data) == 0:
        return [0.0], 0
    z_scores = stats.zscore(data)
    values_disregarded_amount = 0
    filtered_data = []
    for i, z_score in enumerate(z_scores):  
        if z_score < -z_threshold:  
            values_disregarded_amount += 1  
        else:  
            filtered_data.append(data[i])  
    return filtered_data, values_disregarded_amount


def compute_extremal_pixel_value(dataset: DatasetInterface, batch_size: int) -> tuple[float, float, float, float]:
    """Computes the minimum and maximum pixel values across the entire dataset
    
    Returns:
        (min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value): The minimum and maximum pixel values found in the dataset
    """
    if len(dataset) < 1:  
        return 0, 0, 0, 0 
    loader = DataLoader(dataset, batch_size=batch_size)
    min_pixel_values_lr = []
    max_pixel_values_lr = []
    min_pixel_values_hr = []
    max_pixel_values_hr = []
    mean_pixel_value = 0.0
    std_pixel_value = 0.0
    total_pixels = 0

    for lr, hr in tqdm(loader, desc="Computing test dataset pixel value statistics"):
        min_pixel_values_lr.append(lr.min().item())
        max_pixel_values_lr.append(lr.max().item())
        min_pixel_values_hr.append(hr.min().item())
        max_pixel_values_hr.append(hr.max().item())

        # Update mean and std
        batch_pixels = lr.numel() + hr.numel()
        total_pixels += batch_pixels
        mean_pixel_value += (lr.mean().item() * lr.numel() + hr.mean().item() * hr.numel())
        std_pixel_value += (lr.std().item() ** 2 * lr.numel() + hr.std().item() ** 2 * hr.numel())

    mean_pixel_value /= total_pixels
    std_pixel_value = (std_pixel_value / total_pixels) ** 0.5
    
    #combining the min and max pixel values from lr and hr to get the overall min and max pixel values across the dataset
    filtered_min_pixel_values_lr, amount_of_min_values_disregarded_lr = filter_min_outliers(min_pixel_values_lr, z_threshold=3)
    filtered_min_pixel_values_hr, amount_of_min_values_disregarded_hr = filter_min_outliers(min_pixel_values_hr, z_threshold=3)
    amount_of_min_values_disregarded = amount_of_min_values_disregarded_lr + amount_of_min_values_disregarded_hr
    
    min_pixel_values = filtered_min_pixel_values_lr + filtered_min_pixel_values_hr
    max_pixel_values = max_pixel_values_lr + max_pixel_values_hr

    dataset_min_pixel_value = min(min_pixel_values)
    dataset_max_pixel_value = max(max_pixel_values) if max_pixel_values else 0.0
    
    # plot the box plot of the max and min pixel values across the dataset, lr beside hr, in the same figure with two subplots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot([filtered_min_pixel_values_lr, filtered_min_pixel_values_hr], labels=['LR Min Pixel Values', 'HR Min Pixel Values'])
    plt.ylabel('Pixel Value')
    plt.title('Histogram of Minimum Pixel Values')
    plt.subplot(1, 2, 2)
    plt.boxplot([max_pixel_values_lr, max_pixel_values_hr], labels=['LR Max Pixel Values', 'HR Max Pixel Values'])
    plt.ylabel('Pixel Value')
    plt.title('Histogram of Maximum Pixel Values')
    plt.suptitle(f'Box Plots of Minimum and Maximum Pixel Values in the Dataset\n(Disregarding {amount_of_min_values_disregarded} values under -800 for min pixel value)')

    plt.tight_layout()
    plt.show()

    print (f"Dataset pixel value statistics - \nMin: {round(dataset_min_pixel_value, 4)}, Max: {round(dataset_max_pixel_value, 4)}, Mean: {round(mean_pixel_value, 4)}, Std: {round(std_pixel_value, 4)}\n")
    print(f"Disregarding {amount_of_min_values_disregarded} minimum pixel values under -800 for the dataset min pixel value calculation.")

    return dataset_min_pixel_value, dataset_max_pixel_value, mean_pixel_value, std_pixel_value




if __name__ == "__main__":
    lr_dirs = [DATA_DIR / "copernicus" / region for region in ["jutland", "funen"]]
    hr_dirs = [DATA_DIR / "dataforsyningen" / region for region in ["jutland", "funen"]]
    
    division = DataDivision(train=0.8, val=0.1, test=0.1)
    data_splits = get_base_dataset(lr_dirs, hr_dirs, division=division)

    print(f"Train: {len(data_splits.train)} pairs, Val: {len(data_splits.val)} pairs, Test: {len(data_splits.test)} pairs, Total: {len(data_splits.dataset)} pairs")
