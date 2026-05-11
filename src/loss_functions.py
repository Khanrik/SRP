import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Union
from pytorch_msssim import ms_ssim as MSSSIM, ssim as SSIM  # type: ignore

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
        """Returns: Loss value.
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


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        """Returns: Self. Custom loss func.
        Args: None.
        """
        self.grad = nn.L1Loss()

    def forward(self, pred, target):
        """Returns: Loss value.
        Args:
            pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as pred.
        """
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        grad_loss = self.grad(pred_dx, tgt_dx) + self.grad(pred_dy, tgt_dy)
        return grad_loss


class CombinedGradL1Loss(nn.Module):
    def __init__(self, lambda_grad=0.5, lambda_l1=1.0):
        super().__init__()
        """Returns: Self. Combined L1 + Gradient loss to prevent output explosion.
        Args:
            lambda_grad: Weight for gradient loss component.
            lambda_l1: Weight for L1 pixel loss component.
        """
        self.grad_loss = GradLoss()
        self.l1_loss = nn.L1Loss()
        self.lambda_grad = lambda_grad
        self.lambda_l1 = lambda_l1

    def forward(self, pred, target):
        """Returns: Combined loss value.
        Args:
            pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as pred.
        """
        grad_loss = self.grad_loss(pred, target)
        l1_loss = self.l1_loss(pred, target)
        return self.lambda_l1 * l1_loss + self.lambda_grad * grad_loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, k1=0.01, k2=0.03):
        super().__init__()
        """Returns: Self. Differentiable SSIM loss.
        Args:
            window_size: Gaussian window size used for local statistics.
            sigma: Standard deviation for the Gaussian window.
            data_range: Expected dynamic range of the input tensors.
            k1: SSIM stability constant for the luminance term.
            k2: SSIM stability constant for the contrast/structure term.
        """
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

    # def _create_gaussian_window(self, channels, window_size, device, dtype):
    #     coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    #     gauss_1d = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
    #     gauss_1d = gauss_1d / gauss_1d.sum()
    #     window_2d = torch.outer(gauss_1d, gauss_1d)
    #     window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    #     return window_2d.repeat(channels, 1, 1, 1)

    # def forward(self, pred, target):
    #     """Returns: 1 - SSIM score.
    #     Args:
    #         pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
    #         target: The ground truth target tensor of the same shape as pred.
    #     """
    #     if pred.ndim == 3:
    #         pred = pred.unsqueeze(0)
    #     if target.ndim == 3:
    #         target = target.unsqueeze(0)

    #     pred = torch.nan_to_num(pred.float(), nan=0.0, posinf=1.0, neginf=0.0)
    #     target = torch.nan_to_num(target.float(), nan=0.0, posinf=1.0, neginf=0.0)

    #     _, channels, height, width = pred.shape
    #     window_size = min(self.window_size, height, width)
    #     if window_size % 2 == 0:
    #         window_size -= 1
    #     if window_size < 3:
    #         raise ValueError("SSIMLoss requires input images of at least 3x3 pixels.")

    #     padding = window_size // 2
    #     window = self._create_gaussian_window(channels, window_size, pred.device, pred.dtype)

    #     pred_padded = F.pad(pred, (padding, padding, padding, padding), mode="reflect")
    #     target_padded = F.pad(target, (padding, padding, padding, padding), mode="reflect")

    #     mu_pred = F.conv2d(pred_padded, window, groups=channels)
    #     mu_target = F.conv2d(target_padded, window, groups=channels)

    #     mu_pred_sq = mu_pred ** 2
    #     mu_target_sq = mu_target ** 2
    #     mu_pred_target = mu_pred * mu_target

    #     sigma_pred_sq = F.conv2d(pred_padded * pred_padded, window, groups=channels) - mu_pred_sq
    #     sigma_target_sq = F.conv2d(target_padded * target_padded, window, groups=channels) - mu_target_sq
    #     sigma_pred_target = F.conv2d(pred_padded * target_padded, window, groups=channels) - mu_pred_target

    #     sigma_pred_sq = torch.clamp(sigma_pred_sq, min=0.0)
    #     sigma_target_sq = torch.clamp(sigma_target_sq, min=0.0)

    #     c1 = (self.k1 * self.data_range) ** 2
    #     c2 = (self.k2 * self.data_range) ** 2

    #     numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    #     denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    #     denominator = denominator.clamp_min(1e-12)
    #     ssim_map = numerator / denominator
    #     ssim_map = torch.nan_to_num(ssim_map, nan=0.0, posinf=1.0, neginf=-1.0)

    #     return 1 - ssim_map.mean()
    def forward(self, pred, target):
        """Returns: 1 - SSIM score.
        Args:
            pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as pred.
        """

        ssim_score = SSIM(pred, target, data_range=self.data_range, win_size=self.window_size, win_sigma=self.sigma,size_average=True)
        return 1 - ssim_score


class MSSSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, data_range=1.0, k1=0.01, k2=0.03):
        super().__init__()
        """Returns: Self. Differentiable MSSSIM loss.
        Args:
            window_size: Gaussian window size used for local statistics.
            sigma: Standard deviation for the Gaussian window.
            data_range: Expected dynamic range of the input tensors.
            k1: SSIM stability constant for the luminance term.
            k2: SSIM stability constant for the contrast/structure term.
        """
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

    def forward(self, pred, target):
        """Returns: 1 - MSSSIM score.
        Args:
            pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as pred.
        """

        msssim_score = MSSSIM(pred, target, data_range=self.data_range, win_size=self.window_size, win_sigma=self.sigma,size_average=True)
        return 1 - msssim_score