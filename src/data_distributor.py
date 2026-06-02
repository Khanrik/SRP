from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from random import Random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import copy
import logging
import rasterio
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

@dataclass
class DataPair:
    lr: Path
    hr: Path

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
    """A class to define the division of data into training, validation, and test sets.

    All floats must be between 0 and 1 and their sum must equal 1.
    
    It is possible to set a paramter to 0, meaning that no data will be assigned to that set. 

    If `no_division` is set to True, all data will be written to a single folder (`output_path`) instead of being divided into train, val, and test folders.
    """
    train: float = 0
    val: float = 0
    test: float = 0
    no_division: bool = False

    def __post_init__(self):
        total = sum(self.__dict__.values())
        if self.no_division:
            return
        if abs(total - 1.0) > 1e-6:
            raise ValueError("The sum of non bool train, val, and test proportions must equal 1.")
        

class DatasetInterface(Dataset):
    def __init__(self,
                 data_pairs: list[DataPair],
                 lr_target_size: tuple[int, int] = (128, 128),
                 category: str = ""):
        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(lr_target_size)
        ])
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((lr_target_size[0] * 3, lr_target_size[1] * 3))
        ])

        self.lr: list[Image.Image] = []
        self.hr: list[Image.Image] = []
        self.bboxes: list[rasterio.coords.BoundingBox] = []
        for pair in tqdm(data_pairs, desc=f"loading {category} data"):
            self.lr.append(Image.open(pair.lr))
            self.hr.append(Image.open(pair.hr))
            
            # saving bbox for visualization maps
            with rasterio.open(pair.lr) as src:
                bounds = src.bounds
                self.crs = src.crs
            self.bboxes.append(bounds)
        self.category = category

    def _use_lr_transform_for_hr(self, idx: int) -> bool:
        lr_filename = str(getattr(self.lr[idx], "filename", "")).lower()
        hr_filename = str(getattr(self.hr[idx], "filename", "")).lower()
        return "ethiopia" in lr_filename and "ethiopia" in hr_filename

    def __add__(self, other):
        combined = copy.deepcopy(self)
        combined.lr += other.lr 
        combined.hr += other.hr
        combined.bboxes += other.bboxes
        
        return combined

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx: int):
        hr_transform = self.lr_transform if self._use_lr_transform_for_hr(idx) else self.hr_transform
        return  self.lr_transform(self.lr[idx]).float(), \
            hr_transform(self.hr[idx]).float()
    
    def get_bbox(self, idx: int):
        return self.bboxes[idx]

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
                          division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1)) :
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

def unshuffle_dataloader(dataloader: DataLoader) -> DataLoader:
    """Returns a dataloader with the same data as the input dataloader but with shuffle set to False."""
    unshuffled_dataloader = prepare_dataloader(dataloader.dataset, dataloader.batch_size, dataloader.pin_memory, num_workers=dataloader.num_workers, shuffle_bool=False)
    return unshuffled_dataloader

def get_base_dataset(lr_data_dir_list: list[Path],
                     hr_data_dir_list: list[Path],
                     batch_size: int,
                     cuda: bool,
                     division: DataDivision = DataDivision(train=0.8, val=0.1, test=0.1),
                     randomize: bool = True,
                     seed: int = 12345678,
                     category: str = None,
                     include_plot: bool = False,
                     logger: logging.Logger = None,
                     compute_extremals: bool = True
                     ) -> tuple[DataLoader, DataLoader, DataLoader, float, float, float, float]:
    if division.train + division.val + division.test != 1.0:
        raise ValueError(f"Data division proportions must sum to 1.0. Got {division.train} + {division.val} + {division.test} = {division.train + division.val + division.test}")
    all_pairs = _pair_files(lr_data_dir_list, hr_data_dir_list)
    if randomize:
        Random(seed).shuffle(all_pairs)
    N = len(all_pairs)

    train_count = int(N * division.train)
    val_count = int(N * division.val)
    train_end = train_count
    val_end = train_count + val_count

    train_dataset = DatasetInterface(all_pairs[:train_end], category=category or "training")
    val_dataset = DatasetInterface(all_pairs[train_end:val_end], category=category or "validation")
    test_dataset = DatasetInterface(all_pairs[val_end:], category=category or "test")

    train_dataloader = None
    val_dataloader = None
    if train_count != 0:
        train_dataloader = prepare_dataloader(train_dataset, batch_size, cuda, shuffle_bool=randomize)
    if val_count != 0:
        val_dataloader = prepare_dataloader(val_dataset, batch_size, cuda, shuffle_bool=randomize)
    test_dataloader = prepare_dataloader(test_dataset, batch_size, cuda, shuffle_bool=randomize)

    if compute_extremals:
        min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value = compute_extremal_pixel_value(train_dataloader, include_plot=include_plot, logger=logger) if train_dataloader is not None else compute_extremal_pixel_value(test_dataloader, include_plot=include_plot, logger=logger)
    else:
        min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value = 0.0, 0.0, 0.0, 0.0
    
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
    
    return (train_dataloader, val_dataloader, test_dataloader, min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value)


def loader_to_downsampled_loader(loader: DataLoader, downsample_factor: int = 3, shuffle_bool: bool = True) -> DataLoader:
    """Returns a dataloader where the LR images are replaced by the HR images downsampled by `downsample_factor` using the bicubic method"""
    downsampled_dataset = copy.deepcopy(loader.dataset)
    downsampled_dataset.lr = [img.resize((img.width // downsample_factor, img.height // downsample_factor), resample=Image.Resampling.BICUBIC) for img in tqdm(downsampled_dataset.hr, desc="Downsampling HR images")]
    downsampled_loader = prepare_dataloader(downsampled_dataset, loader.batch_size, loader.pin_memory, num_workers=loader.num_workers, shuffle_bool=shuffle_bool)
    return downsampled_loader


def dataset_to_downsampled_dataset(dataset: tuple[DataLoader, DataLoader, DataLoader, float, float, float, float], downsample_factor: int = 3, logger: logging.Logger = None) -> tuple[DataLoader, DataLoader, DataLoader, float, float, float, float]:
    """Returns a dataset where the LR images are replaced by the HR images downsampled by `downsample_factor` using the bicubic method"""
    train_loader, val_loader, test_loader, min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value = dataset
    downsampled_train_loader = loader_to_downsampled_loader(train_loader, downsample_factor)
    downsampled_val_loader = loader_to_downsampled_loader(val_loader, downsample_factor)
    downsampled_test_loader = loader_to_downsampled_loader(test_loader, downsample_factor)
    min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value = compute_extremal_pixel_value(downsampled_train_loader, include_plot=False, logger=logger)
    return (downsampled_train_loader, downsampled_val_loader, downsampled_test_loader, min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value)

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


def compute_extremal_pixel_value(loader: DataLoader,
                                 include_plot: bool = False,
                                 logger: logging.Logger = None) -> tuple[float, float, float, float]:
    """Computes the minimum and maximum pixel values across the entire dataset
    
    Returns:
        (min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value): The minimum and maximum pixel values found in the dataset
    """
    if len(loader.dataset) < 1:
        return 0, 0, 0, 0
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
        total_pixels += lr.numel() + hr.numel()
        mean_pixel_value += lr.sum().item() + hr.sum().item()
        std_pixel_value += (lr.std().item() ** 2 * lr.numel() + hr.std().item() ** 2 * hr.numel())
        # Update histograms
        flat_lr = lr.flatten()
        flat_hr = hr.flatten()

    
    mean_pixel_value /= total_pixels
    std_pixel_value = (std_pixel_value / total_pixels) ** 0.5
    
    #combining the min and max pixel values from lr and hr to get the overall min and max pixel values across the dataset
    filtered_min_pixel_values_lr, amount_of_min_values_disregarded_lr = filter_min_outliers(min_pixel_values_lr, z_threshold=3)
    filtered_min_pixel_values_hr, amount_of_min_values_disregarded_hr = filter_min_outliers(min_pixel_values_hr, z_threshold=3)

    amount_of_min_values_disregarded = amount_of_min_values_disregarded_lr + amount_of_min_values_disregarded_hr
    
    min_pixel_values = filtered_min_pixel_values_lr + filtered_min_pixel_values_hr
    max_pixel_values = max_pixel_values_lr + max_pixel_values_hr

    dataset_min_pixel_value = min(min_pixel_values) if min_pixel_values else 0.0
    dataset_max_pixel_value = max(max_pixel_values) if max_pixel_values else 0.0
    
    if include_plot:
        
        bins=25

        hist_lr=torch.zeros(bins)
        hist_hr=torch.zeros(bins)
        for lr, hr in tqdm(loader, desc="Creating histogram"):
            flat_lr = lr.flatten()
            flat_hr = hr.flatten()
            hist_lr += torch.histc(flat_lr, bins=bins, min=dataset_min_pixel_value, max=dataset_max_pixel_value)
            hist_hr += torch.histc(flat_hr, bins=bins, min=dataset_min_pixel_value, max=dataset_max_pixel_value)

        bin_edges = torch.linspace(dataset_min_pixel_value, dataset_max_pixel_value, bins + 1)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # plot the box plot of the max and min pixel values across the dataset, lr beside hr, in the same figure with two subplots
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 4, 1)
        plt.boxplot([filtered_min_pixel_values_lr, filtered_min_pixel_values_hr], labels=['LR Min Pixel Values', 'HR Min Pixel Values'])
        plt.ylabel('Pixel Value')
        plt.title('Boxplot of Minimum Pixel Values')
        
        plt.subplot(1, 4, 2)
        plt.boxplot([max_pixel_values_lr, max_pixel_values_hr], labels=['LR Max Pixel Values', 'HR Max Pixel Values'])
        plt.ylabel('Pixel Value')
        plt.title('Boxplot of Maximum Pixel Values')
        plt.suptitle(f'(Disregarding {amount_of_min_values_disregarded} values under -800 for min pixel value)')
        
        plt.subplot(1, 4, 3)
        plt.plot(centers, hist_lr.numpy(), label='LR Pixel Value Distribution')
        # place a bar at the point equal to the value 0 on the x-axis to indicate where the value 0 is in the distribution, since there are many pixel values under 0 in the dataset and it is important to see where they are in the distribution
        plt.axvline(x=(0 - dataset_min_pixel_value) / (dataset_max_pixel_value - dataset_min_pixel_value), color='red', linestyle='--', label='Value 0')
        plt.legend()
        plt.xlabel('Normalized Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Pixel Value Distribution For LR')

        plt.subplot(1, 4, 4)
        plt.plot(centers, hist_hr.numpy(), label='HR Pixel Value Distribution')
        plt.axvline(x=0, color='red', linestyle='--', label='Value 0')
        plt.legend()
        plt.xlabel('Normalized Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Pixel Value Distribution For HR')
        
        plt.tight_layout()
        plt.show()

    if logger is not None:
        logger.info(f"Dataset pixel value statistics - \nMin: {round(dataset_min_pixel_value, 4)}, Max: {round(dataset_max_pixel_value, 4)}, Mean: {round(mean_pixel_value, 4)}, Std: {round(std_pixel_value, 4)}\n")
        logger.info(f"Disregarding {amount_of_min_values_disregarded} minimum pixel values under -800 for the dataset min pixel value calculation.")

    return dataset_min_pixel_value, dataset_max_pixel_value, mean_pixel_value, std_pixel_value




if __name__ == "__main__":
    import time
    current_dir = Path(__file__).resolve().parent
    logfile = current_dir.parent / "checkpoints" / "logs" / f"data_distributor_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(logfile),
                        format='%(asctime)s %(levelname)s: %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    
    lr_dirs = [DATA_DIR / "copernicus" / region for region in ["jutland", "funen"]]
    hr_dirs = [DATA_DIR / "dataforsyningen" / region for region in ["jutland", "funen"]]
    
    division = DataDivision(train=0.8, val=0.1, test=0.1)
    data_splits = get_base_dataset(lr_dirs, hr_dirs, division=division, logger=logger)

    logger.info(f"Train: {len(data_splits.train)} pairs, Val: {len(data_splits.val)} pairs, Test: {len(data_splits.test)} pairs, Total: {len(data_splits.dataset)} pairs")
