import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

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

def PSNR(prediction, target, data_range=1.0) -> float:
    """Calculates the PSNR between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
        data_range: The dynamic range of the images (i.e., the difference between the maximum and minimum possible values).
    """
    mse = MSE(prediction, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(data_range**2 / mse)
    return psnr

def SSIM(prediction, target, data_range=1.0) -> float:
    """Calculates the SSIM between the predicted and target tensors
    Args:
        prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
        target: The ground truth target tensor of the same shape as prediction.
        data_range: The dynamic range of the images (i.e., the difference between the maximum and minimum possible values).
    """
    # Convert tensors to numpy images and handle channel ordering for skimage
    p = prediction.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    # remove batch dim if present
    if p.ndim == 4:
        p = p[0]
    if t.ndim == 4:
        t = t[0]

    # If channels-first (C,H,W) -> transpose to (H,W,C)
    if p.ndim == 3 and p.shape[0] in (1, 3, 4):
        p = p.transpose(1, 2, 0)
        t = t.transpose(1, 2, 0)

    # Now p and t should be HxW or HxWxC
    h, w = p.shape[0], p.shape[1]
    min_side = min(h, w)
    # skimage default win_size is 7; ensure odd and <= min_side
    win_size = 7
    if min_side < win_size:
        win_size = min_side if (min_side % 2 == 1) else max(1, min_side - 1)
    if win_size < 3:
        return float('nan')

    # Try calling structural_similarity with channel_axis (newer skimage) or multichannel (older)
    try:
        if p.ndim == 3:
            return float(ssim(p, t, data_range=data_range, channel_axis=2, win_size=win_size))
        else:
            return float(ssim(p, t, data_range=data_range, win_size=win_size))
    except TypeError:
        # fallback for older skimage versions
        if p.ndim == 3:
            return float(ssim(p, t, data_range=data_range, multichannel=True, win_size=win_size))
        else:
            return float(ssim(p, t, data_range=data_range, win_size=win_size))