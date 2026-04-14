import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from contextlib import nullcontext
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from unet_helper import UNet
from data_distributor import DataPair, get_base_dataset

def diff_in_height_coefficient(prediction, target) :
        """Returns: Difference as tensor.
        Args:
            prediction: The predicted output from the model, expected to be a tensor of shape (batch_size, channels, height, width).
            target: The ground truth target tensor of the same shape as prediction.
        """
        # For height prediction, we can use mean absolute error as a simple "difference in height coefficient".
        difference = torch.mean(torch.abs(prediction - target))
        return difference

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

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx: int):
        return  self.lr_transform(self.lr[idx]).float(), \
                self.hr_transform(self.hr[idx]).float()

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        learning_rate=3e-4,
        max_pixels_per_image=1024*1024,
        profile_layers_once=True,
        normalize_targets=False,
        target_norm_eps=1e-6,
    ):
        """ Returns: Self. Initializes the Trainer with the model, optimizer, loss function, device, and learning rate.
        Args:
            model: The neural network model.
            optimizer: The chosen optimizer for training the model.
            criterion: The chosen loss function.
            device: The device to run the model on, eg. "cuda" or "cpu".
            learning_rate: The learning rate for the optimizer. (Default 3e-4).
            max_pixels_per_image: Max num of pixels allowed in input/target images to avoid OOM errors. (Default 1024x1024).
            profile_layers_once: Whether to profile layers in start of training. (Default True).
        """
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion
        self.device = device
        self.pin_memory = device == "cuda"
        self.num_workers = 0
        self.use_amp = device == "cuda"
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.max_pixels_per_image = max_pixels_per_image
        self.profile_layers_once = profile_layers_once
        self.normalize_targets = normalize_targets
        self.target_norm_eps = target_norm_eps
        self.target_mean = None
        self.target_std = None

    def _compute_target_norm_stats(self, dataset, stats_batch_size):
        # Compute train-target mean/std once so normalization can be toggled on safely.
        stats_loader = DataLoader(
            dataset=dataset,
            batch_size=stats_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
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
        std = max(variance ** 0.5, self.target_norm_eps) # Avoid zero std with epsilon floor.

        self.target_mean = mean
        self.target_std = std
        print(f"[norm] target_mean={self.target_mean:.6f} target_std={self.target_std:.6f}")

    def _normalize_target(self, target):
        if not self.normalize_targets:
            return target
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Normalization is enabled but target stats are not initialized.")
        return (target - self.target_mean) / self.target_std

    def _denormalize_target(self, target):
        if not self.normalize_targets:
            return target
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Normalization is enabled but target stats are not initialized.")
        return target * self.target_std + self.target_mean

    def _tensor_mb(self, tensor):
        # Calculate the approximate memory usage of a tensor in megabytes. 
        # This is for avoiding OOM errors and does not account for additional memory used (autograd, optimizer states, etc).
        return tensor.nelement() * tensor.element_size() / (1024 ** 2)

    def _validate_batch_shapes(self, LR, target, prediction):
        # Ensuring that the input image, target, and prediction have reasonable shapes and sizes 
        # to avoid OOM errors and shape mismatches during training.
        LR_pixels = LR.shape[-2] * LR.shape[-1]
        target_pixels = target.shape[-2] * target.shape[-1]
        if LR_pixels > self.max_pixels_per_image:
            raise ValueError(
                f"Input image too large: {tuple(LR.shape)} ({LR_pixels} pixels/image). "
                f"Lower resolution or increase max_pixels_per_image."
            )
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: pred={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}."
            )
        if target_pixels > self.max_pixels_per_image * 9:
            raise ValueError(
                f"Target image is very large: {tuple(target.shape)} ({target_pixels} pixels/image). "
                f"Use smaller lr_resize_to/hr_resize_to values."
            )

    def _log_shape_and_memory(self, stage, epoch, batch_idx, LR, target, prediction):
        # Log the shapes, data types, value ranges of the input, target, and prediction tensors,
        # as well as the approximate memory usage of the batch tensors and CUDA memory stats if using GPU. 
        # This is for debugging and analysis of training dynamics and OOM issues.
        batch_memory_mb = (
            self._tensor_mb(LR) + self._tensor_mb(target) + self._tensor_mb(prediction)
        )
        print(
            f"[{stage}] epoch={epoch + 1} batch={batch_idx + 1} "
            f"LR={tuple(LR.shape)} target={tuple(target.shape)} pred={tuple(prediction.shape)} "
            f"dtype={LR.dtype} LR_minmax=({LR.min().item():.4f},{LR.max().item():.4f}) "
            f"target_minmax=({target.min().item():.4f},{target.max().item():.4f}) "
            f"pred_minmax=({prediction.min().item():.4f},{prediction.max().item():.4f})"
        )
        if self.device == "cuda":
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

    def _profile_layer_activations(self, sample_batch):
        # This function registers forward hooks on convolutional and pooling layers
        # to log the shape and approximate memory usage of their outputs during a forward pass with a sample batch.

        if not self.profile_layers_once:
            return

        hooks = [] 
        activations=[] # For plotting.

        def hook_fn(name):
            def _hook(_module, _inp, out):
                if isinstance(out, torch.Tensor): # Only log if output is a tensor (some layers may output tuples, dicts, etc.)
                    out_mb = self._tensor_mb(out)
                    print(f"[layer] {name}: shape={tuple(out.shape)} approx={out_mb:.2f}MB")
            return _hook 
        
        def save_activation_hook(module, input, output):
            activations.append(output.cpu().detach().numpy())

        # Going through all layers and registering hooks on conv and pooling layers 
        # to log their output shapes and memory usage during the first forward pass
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d)): # We can add more layer types if needed
                hooks.append(module.register_forward_hook(hook_fn(name)))
                hooks.append(module.register_forward_hook(save_activation_hook))

        # Run a forward pass with the sample batch to trigger the hooks and log layer outputs
        self.model.eval() 

        with torch.no_grad(): 
            # Enables autocasting for the forward pass if using mixed precision (use_amp=true), 
            # which can reduce memory usage and is important to profile accurately.
            autocast_ctx = ( 
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp
                else nullcontext()
            )
            with autocast_ctx:
                _ = self.model(sample_batch)

        for hook in hooks:
            hook.remove()
        
        
        
        for i, activation in enumerate(activations[0][0]):  # Visualize the first batch
            plt.subplot(4, 8, i+1)
            plt.title(f"Layer {i+1}")
            plt.imshow(activation, cmap='viridis')
            plt.axis('off')

        plt.show()
        self.profile_layers_once = False

    def _prepare_dataloaders(self, train_dataset, val_dataset, test_dataset, batch_size):
        # This function initializes the iterable over the datasets for training, validation, and testing 
        # with the specified batch size and the Trainer's num_workers and pin_memory settings.
        train_dataloader = DataLoader(
            dataset=train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            persistent_workers=False,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            persistent_workers=False,
            shuffle=True,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            persistent_workers=False,
            shuffle=True,
        )

        self.test_dataloader = test_dataloader # Storing test dataloader for later use in testing after training is complete
        return train_dataloader, val_dataloader, test_dataloader

    def train(self, train_dataset, val_dataset, test_dataset, num_epochs, batch_size):
        """Returns: training and validation losses and difference in height coefficients per epoch for analysis and debugging.
        Args:
            train_dataset: Dataset for training.
            val_dataset: Dataset for validation.
            test_dataset: Dataset for testing.
            num_epochs: Number of epochs to train.
            batch_size: Batch size for training and validation.
        
        """
        train_dataloader, val_dataloader, test_dataloader = \
            self._prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)

        if self.normalize_targets:
            self._compute_target_norm_stats(train_dataset, batch_size)
        
        if train_dataloader is None or val_dataloader is None or test_dataloader is None:
            raise ValueError(
                f"Dataloaders not properly initialized. Train samples: {len(train_dataset)}, "
                f"Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
            )

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


        train_losses = []
        train_dcs = []
        val_losses = []
        val_dcs = []

        for epoch in tqdm(range(num_epochs)):
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats() #Keeping track of peak memory usage per epoch for debugging OOM errors

            self.model.train()
            train_running_loss = 0
            train_running_dc = 0
            
            for idx, LR_HR in enumerate(tqdm(train_dataloader, position=0, leave=True)):
                # creating LR and HR tensors for the batch and moving them to the correct device.
                LR = LR_HR[0].float().to(self.device) 
                HR = LR_HR[1].float().to(self.device)
                HR_for_loss = self._normalize_target(HR)

                if epoch == 0 and idx == 0:
                    # Profiling for memory usage stats to avoid OOM errors.
                    self._profile_layer_activations(LR) 

                # Using autocasting for the forward pass and loss calculation to reduce memory usage.
                # If use_amp is false, this will just be a nullcontext and have no effect.
                autocast_ctx = (
                    torch.autocast(device_type=self.device, dtype=torch.float16)
                    if self.use_amp
                    else nullcontext()
                )

                try:
                    with autocast_ctx:
                        # Forward pass through the model to get predictions, and calculating the loss with the criterion.
                        y_pred = self.model(LR)
                        self._validate_batch_shapes(LR, HR_for_loss, y_pred)
                        loss = self.criterion(y_pred, HR_for_loss)
                except RuntimeError as err:
                    if "not enough memory" in str(err).lower():
                        # catching OOM errors during the forward pass or loss calculation.
                        if self.device == "cuda":
                            torch.cuda.empty_cache() 
                        raise RuntimeError(
                            "OOM during forward/loss. Reduce BATCH_SIZE, use smaller resize_to, or simplify model."
                        ) from err
                    raise
                
                # Using set_to_none=True to reduce memory usage by freeing gradients immediately after backward pass.
                self.optimizer.zero_grad(set_to_none=True) 
                y_pred_eval = self._denormalize_target(y_pred)
                dc = diff_in_height_coefficient(y_pred_eval.float(), HR)

                train_running_loss += loss.item()
                train_running_dc += dc.item()

                # Scales the loss for mixed precision (autocast) training to prevent underflow in gradients, 
                # and performs the backward pass.
                self.scaler.scale(loss).backward() 
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if idx == 0:
                    self._log_shape_and_memory("train", epoch, idx, LR, HR, y_pred_eval)

            train_loss = train_running_loss / (idx + 1)
            train_dc = train_running_dc / (idx + 1)
            
            train_losses.append(train_loss)
            train_dcs.append(train_dc)

            # Validation loop, i.e. training loop but without backpropagation and with torch.no_grad() to save memory and computations.
            self.model.eval()
            val_running_loss = 0
            val_running_dc = 0
            
            with torch.no_grad():
                for idx, LR_HR in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                    LR = LR_HR[0].float().to(self.device)
                    HR = LR_HR[1].float().to(self.device)
                    HR_for_loss = self._normalize_target(HR)

                    autocast_ctx = (
                        torch.autocast(device_type=self.device, dtype=torch.float16)
                        if self.use_amp
                        else nullcontext()
                    )
                    with autocast_ctx:
                        y_pred = self.model(LR)
                        self._validate_batch_shapes(LR, HR_for_loss, y_pred)
                        loss = self.criterion(y_pred, HR_for_loss)
                    y_pred_eval = self._denormalize_target(y_pred)
                    dc = diff_in_height_coefficient(y_pred_eval, HR)
                    
                    val_running_loss += loss.item()
                    val_running_dc += dc.item()

                    if idx == 0:
                        self._log_shape_and_memory("val", epoch, idx, LR, HR, y_pred_eval)

                val_loss = val_running_loss / (idx + 1)
                val_dc = val_running_dc / (idx + 1)
            
            val_losses.append(val_loss)
            val_dcs.append(val_dc)

            print("-" * 30)
            print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
            print(f"Training Difference EPOCH {epoch + 1}: {train_dc:.4f}")
            print("\n")
            print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
            print(f"Validation Difference EPOCH {epoch + 1}: {val_dc:.4f}")
            print("-" * 30)

        # Saving the model
        path = 'checkpoints'
        os.makedirs(path, exist_ok = True) 
        torch.save(self.model.state_dict(), os.path.join(path, 'my_checkpoint.pth'))
        return train_losses, train_dcs, val_losses, val_dcs
    
class plotter:
    def __init__(self):
        """Returns: Self.
        Args:
            None
        
        """
        
    
    def plot_val_and_train_loss(self, train_losses, train_dcs, val_losses, val_dcs):
        """Returns: Self.
        Args:
            train_losses: List of training losses per epoch.
            train_dcs: List of training difference coefficients per epoch.
            val_losses: List of validation losses per epoch.
            val_dcs: List of validation difference coefficients per epoch.
        
        """
        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_dcs, label='Train Diff Coeff')
        plt.plot(epochs, val_dcs, label='Val Diff Coeff')
        plt.xlabel('Epoch')
        plt.ylabel('Difference Coefficient')
        plt.title('Training and Validation Difference Coefficient')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_training_images(self, LR, HR, prediction, train_loss, train_dc):
        """Returns: Self.
        Args:
            LR: Low-resolution input image tensor.
            HR: High-resolution target image tensor.
            prediction: Model's predicted high-resolution image tensor.
            train_loss: Current training loss.
            train_dc: Current training difference coefficient.

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

        plt.suptitle(f"Train Loss: {train_loss:.4f}, Train Diff Coeff: {train_dc:.4f}")
        plt.tight_layout()
        plt.show()

class Tester:
    def __init__(
        self,
        model,
        device,
        criterion=nn.SmoothL1Loss(),
        normalize_targets=False,
        target_mean=None,
        target_std=None,
    ):
        """Returns: Self. Initializes the Tester with the trained model and device.
        Args:
            model: The trained neural network model to be tested.
            device: The device to run the model on, eg. "cuda" or "cpu".
            criterion: The loss function to be used for testing.
        
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.normalize_targets = normalize_targets
        self.target_mean = target_mean
        self.target_std = target_std

    def _denormalize_target(self, target):
        if not self.normalize_targets:
            return target
        if self.target_mean is None or self.target_std is None:
            raise ValueError("Normalization is enabled but target_mean/target_std are not set.")
        return target * self.target_std + self.target_mean

    
    def test(self, test_dataloader):
        """Returns: test loss and difference coefficient for the test dataset.
        Args:
            test_dataloader: DataLoader for the test dataset.
        
        """
        self.model.eval()
        test_running_loss = 0
        test_running_dc = 0
        first_prediction = None

        with torch.no_grad():
            for idx, LR_HR in enumerate(tqdm(test_dataloader, position=0, leave=True)):
                LR = LR_HR[0].float().to(self.device)
                HR = LR_HR[1].float().to(self.device)

                y_pred = self.model(LR)
                y_pred_eval = self._denormalize_target(y_pred)
                loss = self.criterion(y_pred_eval, HR)
                dc = diff_in_height_coefficient(y_pred_eval, HR)
                
                test_running_loss += loss.item()
                test_running_dc += dc.item()
                if idx == 0:
                    first_prediction = y_pred_eval.cpu().detach().numpy()
                    plotter().plot_training_images(LR, HR, y_pred_eval, test_running_loss, test_running_dc)

        test_loss = test_running_loss / (idx + 1)
        test_dc = test_running_dc / (idx + 1)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Difference Coefficient: {test_dc:.4f}")
        return test_loss, test_dc


def main():
    current_dir = Path(__file__).parent
    data_root = current_dir.parent / "data"  # Contains train/, val/, test/
    regions = ["jutland", "funen"]
    data = get_base_dataset(
        lr_data_dir_list=[data_root / "copernicus" / region for region in regions],
        hr_data_dir_list=[data_root / "dataforsyningen" / region for region in regions],
    )
    
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 3
    NORMALIZE_TARGETS = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.AdamW
    criterion = SmoothGradLoss(beta=1.0, lambda_grad=0.2)

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        learning_rate=LEARNING_RATE,
        normalize_targets=NORMALIZE_TARGETS,
    )

    # flattens out at about 38 epochs
    train_losses, train_dcs, val_losses, val_dcs = \
        trainer.train(DatasetInterface(data.train), 
                      DatasetInterface(data.val), 
                      DatasetInterface(data.test), 
                      num_epochs=38, batch_size=BATCH_SIZE) 
    
    print(f"Training complete. \n Final training loss and dc: {train_losses[-1]:.4f}, {train_dcs[-1]:.4f}")
    print(f"Validation loss and dc: {val_losses[-1]:.4f}, {val_dcs[-1]:.4f}")
    plotter().plot_val_and_train_loss(train_losses, train_dcs, val_losses, val_dcs)

    model_pth = current_dir.parent / "checkpoints" / "my_checkpoint.pth"
    trained_model = UNet(in_channels=1, num_classes=1).to(device)
    trained_model.load_state_dict(torch.load(model_pth, map_location=torch.device(device),weights_only=True))

    tester = Tester(
        trained_model,
        device,
        criterion,
        normalize_targets=trainer.normalize_targets,
        target_mean=trainer.target_mean,
        target_std=trainer.target_std,
    )
    tester.test(trainer.test_dataloader)

if __name__ == "__main__":
    main()

