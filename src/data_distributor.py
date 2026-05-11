from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from random import Random
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
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

        self.lr = []
        self.hr = []
        self.bboxes = []
        for pair in tqdm(data_pairs, desc=f"loading {category} data"):
            self.lr.append(Image.open(pair.lr))
            self.hr.append(Image.open(pair.hr))
            
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
        combined.bboxes += other.bboxes
        
        return combined

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx: int):
        return  self.lr_transform(self.lr[idx]).float(), \
                self.hr_transform(self.hr[idx]).float()
    
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
                     randomize: bool = True,
                     seed: int = None,
                     category: str = None) -> tuple[DataLoader, DataLoader, DataLoader, float, float]:
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
        train_dataloader = prepare_dataloader(train_dataset, batch_size, cuda)
    if val_count != 0:
        val_dataloader = prepare_dataloader(val_dataset, batch_size, cuda)
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
    
    return (train_dataloader, val_dataloader, test_dataloader, min_pixel_value, max_pixel_value)


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
