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
    same_as_lr: bool = False

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
        """Initializes the dataset by loading the low resolution and high resolution images from the file paths specified in the data_pairs list, applying the necessary transformations, and storing them in memory for later retrieval.
        Args:
            data_pairs (list[DataPair]): A list of DataPair objects containing the file paths for the low resolution and high resolution images, as well as flags indicating whether the HR image is the same as the LR image.
            lr_target_size (tuple[int, int], optional): The target size for the low resolution images (width, height) in pixels. The high resolution images will be resized to 3 times this size. Default is (128, 128).
            category (str, optional): A string to be used in logging and progress bars to indicate the category of the dataset (e.g., "training", "validation", "test"). Default is an empty string.
        """
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
        self.same_as_lr_flags: list[bool] = []
        self.bboxes: list[rasterio.coords.BoundingBox] = []
        for pair in tqdm(data_pairs, desc=f"loading {category} data"):
            lr_img = Image .open(pair.lr)
            self.lr.append(lr_img)
            self.hr.append(lr_img if pair.same_as_lr else Image.open(pair.hr))
            self.same_as_lr_flags.append(pair.same_as_lr)
            
            # saving bbox for visualization maps
            with rasterio.open(pair.lr) as src:
                bounds = src.bounds
                self.crs = src.crs
            self.bboxes.append(bounds)
        self.category = category

    def __add__(self, other):
        combined = copy.deepcopy(self)
        combined.lr += other.lr 
        combined.hr += other.hr
        combined.same_as_lr_flags += other.same_as_lr_flags
        combined.bboxes += other.bboxes
        
        return combined

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx: int):
        hr_transform = self.lr_transform if self.same_as_lr_flags[idx] else self.hr_transform
        return (
            self.lr_transform(self.lr[idx]).float(),
            hr_transform(self.hr[idx]).float(),
            idx
        )
    
    def delete_item(self, idx: int):
        del self.lr[idx]
        del self.hr[idx]
        del self.same_as_lr_flags[idx]
        del self.bboxes[idx]
    
    def get_bbox(self, idx: int):
        return self.bboxes[idx]

def _pair_files(lr_data_dir_list: list[Path],
                hr_data_dir_list: list[Path],
                same_as_lr_regions: set[str] | None = None) -> list[DataPair]:
    """Pairs low resolution and high resolution image files based on their filenames and returns a list of DataPair objects containing the file paths and same_as_lr flags.
    Args:
        lr_data_dir_list (list[Path]): A list of directories containing the low resolution images.
        hr_data_dir_list (list[Path]): A list of directories containing the high resolution images.
        same_as_lr_regions (set[str] | None, optional): A set of region names (derived from file paths) for which the HR image is the same as the LR image. This is used to set the same_as_lr flag in the DataPair objects. Default is None.
    Returns:
        list[DataPair]: A list of DataPair objects containing the file paths and same_as_lr flags.
    """
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
        region_name = lr_file.parent.name.lower()
        pairs.append(DataPair(
            lr=lr_file,
            hr=hr_file,
            same_as_lr=region_name in same_as_lr_regions if same_as_lr_regions is not None else False,
        ))

    return pairs




def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, pin_memory: bool, num_workers: int = 0, shuffle_bool: bool = True)-> DataLoader:
    """Prepares a DataLoader from a given dataset with the specified parameters.
    Args:
        dataset (Dataset): The dataset to be loaded into the DataLoader.
        batch_size (int): The batch size to be used for loading the data.
        pin_memory (bool): A boolean indicating whether to use pinned memory for faster data transfer to CUDA-enabled GPUs.
        num_workers (int, optional): The number of subprocesses to use for data loading. Default is 0, which means that the data will be loaded in the main process.
        shuffle_bool (bool, optional): A boolean indicating whether to shuffle the data at every epoch. Default is True.
    Returns:
        DataLoader: The prepared DataLoader.
    """
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
                     same_as_lr_regions: set[str] | None = None
                     ) -> tuple[DataLoader, DataLoader, DataLoader, float, float, float, float]:
    """Gets the base dataset for training, validation, and testing.
    
    Args:
        lr_data_dir_list (list[Path]): A list of directories containing the low resolution images.
        hr_data_dir_list (list[Path]): A list of directories containing the high resolution images.
        batch_size (int): The batch size to be used for the dataloaders.
        cuda (bool): A boolean indicating whether to use CUDA for the dataloaders.
        division (DataDivision, optional): A DataDivision object defining the proportions of data to be used for training, validation, and testing. Default is DataDivision(train=0.8, val=0.1, test=0.1).
        randomize (bool, optional): A boolean indicating whether to randomize the order of the data before splitting it into training, validation, and test sets. Default is True.
        seed (int, optional): The seed to be used for randomization if `randomize` is True. Default is 12345678.
        category (str, optional): A string to be used in logging and progress bars to indicate the category of the dataset (e.g., "training", "validation", "test"). Default is None.
        include_plot (bool, optional): A boolean indicating whether to include a plot of pixel value distribution in the output. Default is False.
        logger (logging.Logger, optional): Alogging.Logger instance for logging messages. If None, no logging will be performed. Default is None.
        same_as_lr_regions (set[str] | None, optional): A set of region names (derived from file paths) for which the HR image is the same as the LR image. This is used to apply different transformations to these images in the DatasetInterface. Default is None.
    Returns:
        tuple[DataLoader, DataLoader, DataLoader, float, float, float, float]: A tuple containing the training, validation, and test dataloaders, as well as the minimum pixel value, maximum pixel value, mean pixel value, and standard deviation of pixel values across the dataset.
    
    """
    if division.train + division.val + division.test != 1.0:
        raise ValueError(f"Data division proportions must sum to 1.0. Got {division.train} + {division.val} + {division.test} = {division.train + division.val + division.test}")
    all_pairs = _pair_files(lr_data_dir_list, hr_data_dir_list, same_as_lr_regions=same_as_lr_regions)
    if randomize:
        Random(seed).shuffle(all_pairs)
    N = len(all_pairs)

    train_count = int(N * division.train)
    val_count = int(N * division.val)
    test_count = N - train_count - val_count
    train_end = train_count
    val_end = train_count + val_count

    train_dataset = DatasetInterface(all_pairs[:train_end], category=category or "training") if train_count != 0 else None
    val_dataset = DatasetInterface(all_pairs[train_end:val_end], category=category or "validation") if val_count != 0 else None
    test_dataset = DatasetInterface(all_pairs[val_end:], category=category or "test") if test_count != 0 else None    
    
    train_dataloader = prepare_dataloader(train_dataset, batch_size, cuda, num_workers=0, shuffle_bool=randomize) if train_count != 0 else None
    val_dataloader = prepare_dataloader(val_dataset, batch_size, cuda, num_workers=0, shuffle_bool=randomize) if val_count != 0 else None
    test_dataloader = prepare_dataloader(test_dataset, batch_size, cuda, num_workers=0, shuffle_bool=randomize) if test_count != 0 else None

    if train_dataloader is not None:
        min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value, train_dataloader, _ = compute_extremal_pixel_value(train_dataloader, include_plot=include_plot, logger=logger)
        _, _, _, _, val_dataloader, _ = compute_extremal_pixel_value(val_dataloader, include_plot=False, logger=logger)
        _, _, _, _, test_dataloader, _ = compute_extremal_pixel_value(test_dataloader, include_plot=False, logger=logger)
    else:
        min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value, test_dataloader, _ = compute_extremal_pixel_value(test_dataloader, include_plot=include_plot, logger=logger)

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
    downsampled_train_loader = loader_to_downsampled_loader(train_loader, downsample_factor) if train_loader is not None else None
    downsampled_val_loader = loader_to_downsampled_loader(val_loader, downsample_factor) if val_loader is not None else None
    downsampled_test_loader = loader_to_downsampled_loader(test_loader, downsample_factor) if test_loader is not None else None
    min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value, downsampled_train_loader, _ = compute_extremal_pixel_value(downsampled_train_loader, include_plot=False, logger=logger) if downsampled_train_loader is not None else (min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value, None, None)
    _, _, _, _, downsampled_val_loader, _ = compute_extremal_pixel_value(downsampled_val_loader, include_plot=False, logger=logger) if downsampled_val_loader is not None else (None, None, None, None, None, None)
    _, _, _, _, downsampled_test_loader, _ = compute_extremal_pixel_value(downsampled_test_loader, include_plot=False, logger=logger) if downsampled_test_loader is not None else (None, None, None, None, None, None)
    return (downsampled_train_loader, downsampled_val_loader, downsampled_test_loader, min_pixel_value, max_pixel_value, mean_pixel_value, std_pixel_value)


def compute_extremal_pixel_value(loader: DataLoader,
                                 include_plot: bool = False,
                                 logger: logging.Logger = None,
                                 threshold: float = -800.0,
                                 bins: int = 25) -> tuple[float, float, float, float, DataLoader, DataLoader]:
    """Computes the minimum and maximum pixel values across the entire dataset
    
    Args:
        loader (DataLoader): A DataLoader containing the dataset for which to compute the extremal pixel values.
        include_plot (bool, optional): A boolean indicating whether to include a plot of pixel value distribution in the output. Default is False.
        logger (logging.Logger, optional): A logging.Logger instance for logging messages. If None, no logging will be performed. Default is None.
        threshold (float, optional): A threshold value to filter out minimum outlier values. Meant for NAN filtering. Default is -800.0.
        bins (int, optional): The number of bins to use for the pixel value distribution histogram. Default is 25.
    Returns:
        tuple[float, float, float, float, DataLoader, DataLoader]: The minimum, maximum, mean, and standard deviation of pixel values found in the dataset, along with the filtered and unfiltered DataLoaders
    """
    if len(loader.dataset) < 1:
        return 0, 0, 0, 0, loader, loader
    
    mean_pixel_value = 0.0
    std_pixel_value = 0.0
    total_pixels = 0
    hist_lr=torch.zeros(bins)
    hist_hr=torch.zeros(bins)
    max_pixel_values_lr = []
    max_pixel_values_hr = []
    filtered_min_pixel_values_lr = []
    filtered_min_pixel_values_hr = []
    dataset_min_pixel_value = float('inf')
    dataset_max_pixel_value = float('-inf')
    amount_of_min_values_disregarded = 0
    filtered_dataloader = copy.deepcopy(loader)
    bad_value_indices = []
    
    for lr, hr, indices in tqdm(loader, desc="Computing test dataset pixel value statistics", total=len(loader)):
        for batch, dataset_idx in enumerate(indices.tolist()):
            sample_lr = lr[batch]
            sample_hr = hr[batch]
            lr_min = sample_lr.min().item()
            hr_min = sample_hr.min().item()
            lr_max = sample_lr.max().item()
            hr_max = sample_hr.max().item()

            max_pixel_values_lr.append(lr_max)
            max_pixel_values_hr.append(hr_max)

            if lr_min < threshold or hr_min < threshold:
                amount_of_min_values_disregarded += 1
                bad_value_indices.append(dataset_idx)
                continue
            
            filtered_min_pixel_values_lr.append(lr_min)
            filtered_min_pixel_values_hr.append(hr_min)
            hist_lr += torch.histc(sample_lr.flatten(), bins=bins, min=lr_min, max=lr_max)
            hist_hr += torch.histc(sample_hr.flatten(), bins=bins, min=hr_min, max=hr_max)

            # Update mean and std
            total_pixels += sample_lr.numel() + sample_hr.numel()
            mean_pixel_value += sample_lr.sum().item() + sample_hr.sum().item()
            std_pixel_value += (sample_lr.std().item() ** 2 * sample_lr.numel() + sample_hr.std().item() ** 2 * sample_hr.numel())

            dataset_min_pixel_value = min(dataset_min_pixel_value, lr_min, hr_min)
            dataset_max_pixel_value = max(dataset_max_pixel_value, lr_max, hr_max)
    
    for idx in sorted(bad_value_indices, reverse=True):
        filtered_dataloader.dataset.delete_item(idx)

    mean_pixel_value /= total_pixels
    std_pixel_value = (std_pixel_value / total_pixels) ** 0.5
    
    if include_plot:
        plot_outlier_plots(hist_lr, hist_hr, dataset_min_pixel_value, dataset_max_pixel_value, filtered_min_pixel_values_lr, filtered_min_pixel_values_hr, max_pixel_values_lr, max_pixel_values_hr, amount_of_min_values_disregarded, bins)

    if logger is not None:
        logger.info(f"Dataset pixel value statistics - \nMin: {round(dataset_min_pixel_value, 4)}, Max: {round(dataset_max_pixel_value, 4)}, Mean: {round(mean_pixel_value, 4)}, Std: {round(std_pixel_value, 4)}\n")
        logger.info(f"Disregarding {amount_of_min_values_disregarded} minimum pixel values under -800 for the dataset min pixel value calculation.")

    return dataset_min_pixel_value, dataset_max_pixel_value, mean_pixel_value, std_pixel_value, filtered_dataloader, loader


def plot_outlier_plots(hist_lr: torch.Tensor, hist_hr: torch.Tensor, dataset_min_pixel_value: float, dataset_max_pixel_value: float, filtered_min_pixel_values_lr: list, filtered_min_pixel_values_hr: list, max_pixel_values_lr: list, max_pixel_values_hr: list, amount_of_min_values_disregarded: int, bins: int):
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
    plt.close('all')
        

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
