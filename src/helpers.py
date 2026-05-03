from __future__ import annotations

import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from contextlib import nullcontext

from dataclasses import dataclass
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING
from tqdm import tqdm

if TYPE_CHECKING:
    from data_distributor import DataPair

@dataclass
class results:
    image: torch.Tensor
    name: str
    metrics: list[tuple[str, float]]

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
                 loading_description: str = "Loading dataset"):
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
        for pair in tqdm(data_pairs, desc=loading_description):
            self.lr.append(Image.open(pair.lr))
            self.hr.append(Image.open(pair.hr))

    def __add__(self, other):
        combined = copy.deepcopy(self)
        combined.lr += other.lr 
        combined.hr += other.hr
        return combined

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx: int):
        return  self.lr_transform(self.lr[idx]).float(), \
                self.hr_transform(self.hr[idx]).float()


def compute_max_pixel_value(dataset: DatasetInterface, batch_size: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size)
    max_pixel_value = float('-inf')
    for _, hr in tqdm(loader, desc="Computing max pixel value"):
        max_pixel_value = max(max_pixel_value, hr.max().item())
    return max_pixel_value

def compute_target_norm_stats(dataset: DatasetInterface, 
                              stats_batch_size: int, 
                              num_workers: int = 0, 
                              target_norm_eps: float = 1e-6) -> tuple[float, float]:
    # Compute train-target mean/std once so normalization can be toggled on safely.
    stats_loader = DataLoader(
        dataset=dataset,
        batch_size=stats_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    with torch.no_grad():
        for _, hr in tqdm(stats_loader, desc="Computing normalization stats"):
            hr = hr.float()
            total_sum += hr.sum().item()
            total_sq_sum += (hr * hr).sum().item()
            total_count += hr.numel()

    if total_count == 0:
        raise ValueError("Cannot compute normalization stats from an empty training dataset.")

    mean = total_sum / total_count
    variance = max((total_sq_sum / total_count) - (mean * mean), 0.0)
    std = max(variance ** 0.5, target_norm_eps) # Avoid zero std with epsilon floor.

    target_mean = mean
    target_std = std
    print(f"[norm] target_mean={target_mean:.6f} target_std={target_std:.6f}")
    return target_mean, target_std


def normalize_target(target, target_mean, target_std) -> torch.Tensor:
    return (target - target_mean) / target_std

def denormalize_target(target, target_mean, target_std) -> torch.Tensor:
    return target * target_std + target_mean

def tensor_mb(tensor) -> float:
    # Calculate the approximate memory usage of a tensor in megabytes. 
    # This is for avoiding OOM errors and does not account for additional memory used (autograd, optimizer states, etc).
    return tensor.nelement() * tensor.element_size() / (1024 ** 2)

def validate_batch_shapes(LR, target, prediction, max_pixels_per_image=1024*1024):
    # Ensuring that the input image, target, and prediction have reasonable shapes and sizes 
    # to avoid OOM errors and shape mismatches during training.
    LR_pixels = LR.shape[-2] * LR.shape[-1]
    target_pixels = target.shape[-2] * target.shape[-1]
    if LR_pixels > max_pixels_per_image:
        raise ValueError(
            f"Input image too large: {tuple(LR.shape)} ({LR_pixels} pixels/image). "
            f"Lower resolution or increase max_pixels_per_image."
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: pred={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}."
        )
    if target_pixels > max_pixels_per_image * 9:
        raise ValueError(
            f"Target image is very large: {tuple(target.shape)} ({target_pixels} pixels/image). "
            f"Use smaller lr_resize_to/hr_resize_to values."
        )

def log_shape_and_memory(stage: str, 
                         epoch: int, 
                         batch_idx: int, 
                         LR: torch.Tensor, 
                         target: torch.Tensor, 
                         prediction: torch.Tensor, 
                         cuda: bool):
    # Log the shapes, data types, value ranges of the input, target, and prediction tensors,
    # as well as the approximate memory usage of the batch tensors and CUDA memory stats if using GPU. 
    # This is for debugging and analysis of training dynamics and OOM issues.
    batch_memory_mb = (
        tensor_mb(LR) + tensor_mb(target) + tensor_mb(prediction)
    )
    print(
        f"[{stage}] epoch={epoch + 1} batch={batch_idx + 1} "
        f"LR={tuple(LR.shape)} target={tuple(target.shape)} pred={tuple(prediction.shape)} "
        f"dtype={LR.dtype} LR_minmax=({LR.min().item():.4f},{LR.max().item():.4f}) "
        f"target_minmax=({target.min().item():.4f},{target.max().item():.4f}) "
        f"pred_minmax=({prediction.min().item():.4f},{prediction.max().item():.4f})"
    )
    if cuda:
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(
            f"[{stage}] memory: batch_tensors={batch_memory_mb:.2f}MB "
            f"cuda_allocated={allocated_mb:.2f}MB cuda_reserved={reserved_mb:.2f}MB "
            f"cuda_peak_allocated={peak_allocated_mb:.2f}MB"
        )
    else:
        print(f"[{stage}] memory: batch_tensors={batch_memory_mb:.2f}MB (cpu)")

def profile_layer_activations(model: nn.Module, 
                              sample_batch: torch.Tensor, 
                              use_amp: bool, 
                              profile_layers_once: bool = True):
    # This function registers forward hooks on convolutional and pooling layers
    # to log the shape and approximate memory usage of their outputs during a forward pass with a sample batch.

    if not profile_layers_once:
        return

    hooks = [] 
    activations=[] # For plotting.

    def hook_fn(name):
        def _hook(_module, _inp, out):
            if isinstance(out, torch.Tensor): # Only log if output is a tensor (some layers may output tuples, dicts, etc.)
                out_mb = tensor_mb(out)
                print(f"[layer] {name}: shape={tuple(out.shape)} approx={out_mb:.2f}MB")
        return _hook 
    
    def save_activation_hook(module, input, output):
        activations.append(output.cpu().detach().numpy())

    # Going through all layers and registering hooks on conv and pooling layers 
    # to log their output shapes and memory usage during the first forward pass
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d)): # We can add more layer types if needed
            hooks.append(module.register_forward_hook(hook_fn(name)))
            hooks.append(module.register_forward_hook(save_activation_hook))

    # Run a forward pass with the sample batch to trigger the hooks and log layer outputs
    model.eval() 

    with torch.no_grad(): 
        # Enables autocasting for the forward pass if using mixed precision (use_amp=true), 
        # which can reduce memory usage and is important to profile accurately.
        autocast_ctx = ( 
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with autocast_ctx:
            _ = model(sample_batch)

    for hook in hooks:
        hook.remove()
    
    
    
    for i, activation in enumerate(activations[0][0]):  # Visualize the first batch
        plt.subplot(4, 8, i+1)
        plt.title(f"Layer {i+1}")
        plt.imshow(activation, cmap='viridis')
        plt.axis('off')

    plt.show()
    profile_layers_once = False

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