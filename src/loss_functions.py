import torch.nn as nn
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

class SmoothLoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        """Returns: Self. Custom loss func.
        Args:
            beta: The beta parameter for the SmoothL1Loss, controls the transition point between L1 and L2 loss. (Default 1.0, values 0.0-1.0 ).
        """
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred, target):
        """Returns: Loss value.
        Args:
            pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as pred.
        """
        return self.smooth_l1(pred, target)

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
    def forward(self, pred, target):
        """Returns: 1 - SSIM score.
        Args:
            pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as pred.
        """

        ssim_score = SSIM(pred, target, data_range=self.data_range, win_size=self.window_size, win_sigma=self.sigma,size_average=True)
        return 1 - ssim_score
    

class MSESSIMLoss(nn.Module):
    def __init__(self, alpha=0.5, window_size=11, sigma=1.5, data_range=1.0, k1=0.01, k2=0.03):
        super().__init__()
        """Returns: Self. Combined MSE and SSIM loss.
        Args:
            alpha: Weighting factor to balance MSE and SSIM components (default 0.5).
            window_size: Gaussian window size used for local statistics in SSIM.
            sigma: Standard deviation for the Gaussian window in SSIM.
            data_range: Expected dynamic range of the input tensors for SSIM.
            k1: SSIM stability constant for the luminance term.
            k2: SSIM stability constant for the contrast/structure term.
        """
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size, sigma, data_range, k1, k2)

    def forward(self, pred, target):
        """Returns: Combined loss value.
        Args:
            pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as pred.
        """
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

class MAESSIMLoss(nn.Module):
    def __init__(self, alpha=0.5, window_size=11, sigma=1.5, data_range=1.0, k1=0.01, k2=0.03):
        super().__init__()
        """Returns: Self. Combined MAE and SSIM loss.
        Args:
            alpha: Weighting factor to balance MAE and SSIM components (default 0.5).
            window_size: Gaussian window size used for local statistics in SSIM.
            sigma: Standard deviation for the Gaussian window in SSIM.
            data_range: Expected dynamic range of the input tensors for SSIM.
            k1: SSIM stability constant for the luminance term.
            k2: SSIM stability constant for the contrast/structure term.
        """
        self.alpha = alpha
        self.mae = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size, sigma, data_range, k1, k2)

    def forward(self, pred, target):
        """Returns: Combined loss value.
        Args:
            pred: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as pred.
        """
        mae_loss = self.mae(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        return self.alpha * mae_loss + (1 - self.alpha) * ssim_loss


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