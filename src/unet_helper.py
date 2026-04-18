import copy
import os
import random
import shutil
import zipfile
from math import atan2, cos, sin, sqrt, pi, log
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import rioxarray
import numpy as np
from rasterio.enums import Resampling
from data_distributor import DataPair
from contextlib import nullcontext
from datetime import datetime

# model
class SmoothGradLoss(nn.Module):
    def __init__(self, beta=1.0, lambda_grad=0.2):
        super().__init__()
        """Returns: Self. Custom loss func.
        Args:
            beta: The beta parameter for the SmoothL1Loss, controls the transition point between L1 and L2 loss. (Default 1.0, values 0.0-1.0 ).
            lambda_grad: The weight for the gradient loss component, encourages smoothness in the predictions. (Default 0.2, values 0.0-1.0 ).
        """
        self.pixel = nn.SmoothL1Loss(beta=beta)
        self.grad = nn.L1Loss()
        self.lambda_grad = lambda_grad

    def forward(self, pred, target):
        """Returns: Loss value
        Args:
            pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as pred.
        """
        pixel_loss = self.pixel(pred, target)
    
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        grad_loss = self.grad(pred_dx, tgt_dx) + self.grad(pred_dy, tgt_dy)
        return pixel_loss + self.lambda_grad * grad_loss

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)
    

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        #print(f"DownSample: down shape: {down.shape}, pool shape: {p.shape}")
        return down, p
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        #print(f"UpSample: x shape: {x.shape}")
        return self.conv(x)
    
class UpSample_last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=3, stride=3)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        #self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(256,512)

        #self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        self.up_convolution_5 = UpSample(64, 32)
        self.up_convolution_6 = UpSample(32,16)
        
        self.last_up_convolution = UpSample_last(64, 32)
        self.first_up_convolution_1 = nn.ConvTranspose2d(in_channels, 32, kernel_size=3, stride=3)

        self.out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)

    def forward(self, x):

        first_1 = self.first_up_convolution_1(x)

        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        #down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p3)

        #up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(b, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)
        up_5 = self.last_up_convolution(up_4,first_1)

        out = self.out(up_5)
        return out


# data and metrics
class plotter:
    def plot_val_and_train_loss(self, train_losses, train_maes, train_rmses, train_psnrs, val_losses, val_maes, val_rmses, val_psnrs, display_plots=False):
        """Returns: Self.
        Args:
            train_losses: List of training losses per epoch.
            train_maes: List of training MAEs per epoch.
            train_rmses: List of training RMSEs per epoch.
            train_psnrs: List of training PSNRs per epoch.
            val_losses: List of validation losses per epoch.
            val_maes: List of validation MAEs per epoch.
            val_rmses: List of validation RMSEs per epoch.
            val_psnrs: List of validation PSNRs per epoch.
        """
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_maes, label='Train MAE')
        plt.plot(epochs, val_maes, label='Val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training and Validation MAE')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_rmses, label='Train RMSE')
        plt.plot(epochs, val_rmses, label='Val RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('Root Mean Squared Error')
        plt.title('Training and Validation RMSE')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(epochs, train_psnrs, label='Train PSNR')
        plt.plot(epochs, val_psnrs, label='Val PSNR')
        plt.xlabel('Epoch')
        plt.ylabel('Peak Signal-to-Noise Ratio')
        plt.title('Training and Validation PSNR')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'train_val_metrics_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
        if display_plots:
            plt.show()

    def plot_training_images(self, LR, HR, prediction, train_loss, train_mae, train_rmse, train_psnr, display_images=False):
        """Returns: Self.
        Args:
            LR: Low-resolution input image tensor.
            HR: High-resolution target image tensor.
            prediction: Model's predicted high-resolution image tensor.
            train_loss: Current training loss.
            train_mae: Current training mean absolute error.
            train_rmse: Current training root mean squared error.
            train_psnr: Current training peak signal-to-noise ratio.
        """
        difference_tensor = torch.abs(prediction - HR)
        def _to_plot_array(tensor):
            # Handling the batch dimension and channel dimension for plotting.
            tensor = tensor.detach().cpu()
            if tensor.ndim == 4:
                tensor = tensor[0]  # Display first image in the batch.
            if tensor.ndim == 3:
                if tensor.shape[0] == 1:
                    tensor = tensor[0]
                elif tensor.shape[0] in (3, 4):
                    tensor = tensor.permute(1, 2, 0)
                else:
                    tensor = tensor[0]
            if tensor.ndim != 2 and tensor.ndim != 3:
                raise ValueError(f"Unsupported tensor shape for plotting: {tuple(tensor.shape)}")
            return tensor.numpy()

        LR_img = _to_plot_array(LR)
        HR_img = _to_plot_array(HR)
        pred_img = _to_plot_array(prediction)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Low-Resolution Input')
        plt.imshow(LR_img, cmap='gray' if LR_img.ndim == 2 else None)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('High-Resolution Target')
        plt.imshow(HR_img, cmap='gray' if HR_img.ndim == 2 else None)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Model Prediction')
        plt.imshow(pred_img, cmap='gray' if pred_img.ndim == 2 else None)
        plt.axis('off')

        plt.suptitle(f"Train Loss: {sum(train_loss):.4f}, Train MAE: {sum(train_mae):.4f}, Train RMSE: {sum(train_rmse):.4f}, Train PSNR: {sum(train_psnr):.4f}")
        plt.tight_layout()
        plt.savefig(f'training_images_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
        if display_images:
            plt.show()

class DatasetInterface(Dataset):
    def __init__(self,
                 data_pairs: list[DataPair],
                 lr_target_size: tuple[int, int] = (128, 128)):
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
        for pair in data_pairs:
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

def mean_absolute_error(prediction, target) -> float:
    """Calculates the MAE between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
    """
    mae_tensor = torch.mean(torch.abs(prediction - target))
    return mae_tensor.item()

def mean_squared_error(prediction, target) -> float:
    """Calculates the MSE between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
    """
    mse_tensor = torch.mean((prediction - target) ** 2)
    return mse_tensor.item()

def root_mean_squared_error(prediction, target, mse=None) -> float:
    """Calculates the RMSE between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
    """
    if mse is None:
        mse = mean_squared_error(prediction, target)
    rmse = np.sqrt(mse)
    return rmse

def peak_signal_to_noise_ratio(prediction, target, max_pixel_value, mse=None) -> float:
    """Calculates the PSNR between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
        max_pixel_value: The maximum pixel value in the data.
        mse: The mean squared error between the predicted and target tensors.
    """
    if mse is None:
        mse = mean_squared_error(prediction, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(max_pixel_value**2 / mse)
    return psnr

def compute_max_pixel_value(dataset: DatasetInterface, batch_size: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size)
    max_pixel_value = float('-inf')
    for _, hr in loader:
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
        for _, hr in stats_loader:
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

def prepare_dataloader(dataset, batch_size, pin_memory, num_workers=0):
    # This function initializes the iterable over the datasets for training, validation, and testing 
    # with the specified batch size and the Trainer's num_workers and pin_memory settings.
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        batch_size=batch_size,
        persistent_workers=False,
        shuffle=True,
    )
    
    return dataloader