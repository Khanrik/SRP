from __future__ import annotations

import logging
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from contextlib import nullcontext
from dataclasses import dataclass
from inspect import signature

@dataclass
class results:
    image: torch.Tensor
    name: str
    metrics: list[tuple[str, float]]

def tensor_mb(tensor: torch.Tensor) -> float:
    """Calculates the approximate memory usage of a tensor in megabytes (MB)."""
    return tensor.nelement() * tensor.element_size() / (1024 ** 2)

def validate_batch_shapes(LR: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor, max_pixels_per_image: int = 1024 * 1024):
    """Validates the shapes of the input tensors for a batch of images and raises errors if they are not compatible.
    Args:
        LR (torch.Tensor): The low-resolution input tensor of shape (batch_size, channels, height, width).
        target (torch.Tensor): The ground truth high-resolution tensor of shape (batch_size, channels, height, width).
        prediction (torch.Tensor): The predicted high-resolution tensor of shape (batch_size, channels, height, width).
        max_pixels_per_image (int, optional): The maximum number of pixels allowed per image to prevent excessive memory usage. Default is 1024*1024 (1 megapixel).
    Raises:
        ValueError: If the input image is too large, if the prediction and target shapes differ, or if the target image is very large.
    """
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

def profile_layer_activations(model: nn.Module, 
                              sample_batch: torch.Tensor, 
                              use_amp: bool, 
                              profile_layers_once: bool = True,
                              logger: logging.Logger | None = None):
    """Profiles the activations of convolutional and pooling layers in a PyTorch model during a forward pass with a sample batch of data. Logs the output shapes and approximate memory usage of each layer, and visualizes the activations of the first batch.
    Args:
        model (nn.Module): The PyTorch model to be profiled.
        sample_batch (torch.Tensor): A sample batch of input data to be passed through the model
        use_amp (bool): A boolean indicating whether to use automatic mixed precision (AMP) during the forward pass, which can reduce memory usage and is important to profile accurately.
        profile_layers_once (bool, optional): A boolean indicating whether to profile the layers only once during the first forward pass. If True, the profiling will only be performed once and subsequent calls to this function will not re-profile the layers. Default is True.
        logger (logging.Logger, optional): A logging.Logger instance for logging messages. If None, no logging will be performed. Default is None.
    Returns:
        None
    """
    if not profile_layers_once:
        return

    hooks = [] 
    activations=[] # For plotting.

    def hook_fn(name):
        def _hook(_module, _inp, out):
            if isinstance(out, torch.Tensor): # Only log if output is a tensor (some layers may output tuples, dicts, etc.)
                out_mb = tensor_mb(out)
                if logger is not None:
                    logger.info(f"[layer] {name}: shape={tuple(out.shape)} approx={out_mb:.2f}MB")
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

def normalize_targets(targets: list[torch.Tensor], mean: float, std: float) -> list[torch.Tensor]:
    """Normalizes a tensor with Z-score normalization

    Args:
        targets (list[torch.Tensor]): A list of target tensors to be normalized.
        mean (float): The mean pixel value in the dataset.
        std (float): The standard deviation of pixel values in the dataset.

    Returns:
        list[torch.Tensor]: A list of normalized tensors.
    """
    if not isinstance(targets, list):
        targets = [targets]

    normalized_targets = []

    for target in targets:
        normalized_targets.append((target - mean) / std)

    return normalized_targets if len(normalized_targets) > 1 else normalized_targets[0]

def denormalize_target(target: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """ Denormalizes a tensor that was normalized with Z-score normalization back to its original scale.
    Args:
        target (torch.Tensor): The normalized target tensor that is normalized
        mean (float): The mean pixel value used for normalization.
        std (float): The standard deviation of pixel values used for normalization.
    Returns:
        torch.Tensor: The denormalized target tensor.
    """
    return target * std + mean

def metric_items(prediction: torch.Tensor, target: torch.Tensor, metrics: dict, min_val: float = 0.0, max_val: float = 1.0)-> list[tuple[str, float]]:
    """Calculates the specified metrics for a given prediction and target tensor, and returns a list of metric names and their corresponding values.
    Args:
        prediction (torch.Tensor): The predicted output tensor from the model.
        target (torch.Tensor): The ground truth target tensor.
        metrics (dict): A dictionary where keys are metric names (str) and values are metric functions that take prediction and target tensors as input and return a scalar metric value. The metric functions can optionally accept additional parameters such as data_range or max_value.
        min_val (float, optional): The minimum possible value of the images, used for metrics that require a data range. Default is 0.0.
        max_val (float, optional): The maximum possible value of the images, used for metrics that require a data range. Default is 1.0.
    Returns:
        list[tuple[str, float]]: A list of tuples, where each tuple contains a metric name and its corresponding calculated value for the given prediction and target tensors.
    """
    if prediction.shape != target.shape:
        return []
    metric_items = []
    for metric_name, metric_func in metrics.items():
        input_parameters = signature(metric_func).parameters.keys()
        if "data_range" in input_parameters:
            metric_value = metric_func(prediction.float(), target, data_range=max_val - min_val)
        elif "max_value" in input_parameters:
            metric_value = metric_func(prediction.float(), target, max_value=max_val)
        else:
            metric_value = metric_func(prediction.float(), target)
        metric_items.append((metric_name, metric_value))
    return metric_items