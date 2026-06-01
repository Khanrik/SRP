import torch
import numpy as np
from pytorch_msssim import ssim as py_SSIM, ms_ssim as py_MSSSIM  # type: ignore

def MAE(prediction, target) -> float:
    """Calculates the MAE between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
    """
    mae_tensor = torch.mean(torch.abs(prediction - target))
    return mae_tensor.item()

def MSE(prediction, target) -> float:
    """Calculates the MSE between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
    """
    mse_tensor = torch.mean((prediction - target) ** 2)
    return mse_tensor.item()

def RMSE(prediction, target) -> float:
    """Calculates the RMSE between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
    """
    mse = MSE(prediction, target)
    rmse = np.sqrt(mse)
    return rmse

def PSNR(prediction, target, max_value=1.0) -> float:
    """Calculates the PSNR between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
        max_value: The maximum possible value of the images (default is 1.0).
    """
    mse = MSE(prediction, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(max_value**2 / mse)
    return psnr

def SSIM(prediction, target, data_range=1.0) -> float:
    """Calculates the SSIM between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
        data_range: The dynamic range of the images (i.e., the difference between the maximum and minimum possible values).
    """
    metric_value = py_SSIM(prediction, target, data_range=data_range, win_size=11, win_sigma=1.5, size_average=True)
    if torch.is_tensor(metric_value):
        metric_value = metric_value.detach().cpu().item()
    else:
        metric_value = float(metric_value)
    return metric_value

def MS_SSIM(prediction, target, data_range=1.0) -> float:
    """Calculates the MS-SSIM between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
        data_range: The dynamic range of the images (i.e., the difference between the maximum and minimum possible values).
    """
    metric_value = py_MSSSIM(prediction, target, data_range=data_range, win_size=11, win_sigma=1.5, size_average=True)
    if torch.is_tensor(metric_value):
        metric_value = metric_value.detach().cpu().item()
    else:
        metric_value = float(metric_value)
    return metric_value