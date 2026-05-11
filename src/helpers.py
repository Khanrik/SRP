from __future__ import annotations

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from contextlib import nullcontext
from dataclasses import dataclass

@dataclass
class results:
    image: torch.Tensor
    name: str
    metrics: list[tuple[str, float]]

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

def normalize_targets(targets: list[torch.Tensor], mean: float, std: float) -> list[torch.Tensor]:
    """Normalizes a tensor with Z-score normalization

    Args:
        targets: A list of target tensors to be normalized.
        mean: The mean pixel value in the dataset.
        std: The standard deviation of pixel values in the dataset.

    Returns:
        A list of normalized tensors.
    """
    if not isinstance(targets, list):
        targets = [targets]

    normalized_targets = []

    for target in targets:
        normalized_targets.append((target - mean) / std)

    return normalized_targets if len(normalized_targets) > 1 else normalized_targets[0]

def denormalize_target(target: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    Args:
        target: The normalized target tensor that is normalized
        mean: The mean pixel value used for normalization.
        std: The standard deviation of pixel values used for normalization.
    Returns:
        
    """
    return target * std + mean