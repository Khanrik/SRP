import torch
import torch.nn as nn
from torch.nn.functional import conv2d

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
    
class GradLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        """Returns: Self. Custom loss func.
        Args: None.
        """
        self.grad = nn.L1Loss()

    def forward(self, pred, target):
        """Returns: Loss value
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
    
# class SSIM(nn.Module):
#     def __init__(self, window_size=11, sigma=1.5):
#         super().__init__()
#         """Returns: Self. Custom loss func.
#         Args:
#             window_size: The size of the Gaussian window used for computing local statistics. (Default 11, typically an odd number).
#             sigma: The standard deviation of the Gaussian window, controls the amount of smoothing applied to the local statistics. (Default 1.5).
#         """
#         self.window_size = window_size
#         self.sigma = sigma

#     def forward(self, pred, target):
#         """Returns: SSIM value
#         Args:
#             pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
#             target: The ground truth target tensor of the same shape as pred.
#         """
#         pred = pred.float()
#         target = target.float()

#         _, channels, height, width = pred.shape
#         window_size = min(self.window_size, height, width)
#         if window_size % 2 == 0:
#             window_size -= 1
#         if window_size < 3:
#             window_size = 3

#         # Using the current data range for SSIM calculation.
#         data_min = torch.min(torch.stack([pred.min(), target.min()]))
#         data_max = torch.max(torch.stack([pred.max(), target.max()]))
#         data_range = torch.clamp(data_max - data_min, min=1e-6)
#         C1 = (0.01 * data_range) ** 2
#         C2 = (0.03 * data_range) ** 2
#         eps = 1e-6

#         # Use a Gaussian blur to compute local means and variances for SSIM.
#         kernel = self.create_gaussian_kernel(window_size, self.sigma, channels).to(pred.device, dtype=pred.dtype)
#         mu_pred = self.gaussian_blur(pred, kernel)
#         mu_target = self.gaussian_blur(target, kernel)

#         sigma_pred = self.gaussian_blur(pred * pred, kernel) - mu_pred ** 2
#         sigma_target = self.gaussian_blur(target * target, kernel) - mu_target ** 2
#         sigma_cross = self.gaussian_blur(pred * target, kernel) - mu_pred * mu_target

#         # Ensure variances stay non-negative and denominators stay well-conditioned.
#         sigma_pred = torch.clamp(sigma_pred, min=0)
#         sigma_target = torch.clamp(sigma_target, min=0)
#         ssim_numerator = (2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)
#         ssim_denominator = (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2)
#         ssim_map = ssim_numerator / (ssim_denominator + eps)
        
#         return 1 - ssim_map.mean()

#     def gaussian_blur(self, x, kernel):
#         padding = kernel.shape[-1] // 2
#         return conv2d(x, kernel, padding=padding, groups=x.shape[1])

#     def create_gaussian_kernel(self, kernel_size, sigma, channels):
#         coords = torch.arange(kernel_size).float() - kernel_size // 2
#         grid = coords.repeat(kernel_size).view(kernel_size, kernel_size)
#         x_grid = grid
#         y_grid = grid.t()

#         kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
#         kernel = kernel / kernel.sum()

#         kernel = kernel.view(1, 1, kernel_size, kernel_size)
#         kernel = kernel.repeat(channels, 1, 1, 1)

#         return kernel