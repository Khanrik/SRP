import os
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from unet_helper import *
from data_distributor import get_base_dataset
from typing import Literal

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str,
        learning_rate: float = 3e-4,
        max_pixels_per_image: int = 1024*1024,
        profile_layers_once: bool = True,
        normalize_targets: bool = False,
        target_norm_eps: float = 1e-6,
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
        self.cuda = device == "cuda"
        self.num_workers = 0
        self.max_pixels_per_image = max_pixels_per_image
        self.profile_layers_once = profile_layers_once
        self.normalize_targets = normalize_targets
        self.target_norm_eps = target_norm_eps

        # this enables compatibility for older python 3.8.10 i think or maybe linux
        try:
            self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.cuda)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cuda)

    def _run_loop(self,
                   dataloader: DataLoader,
                   training_state: Literal["train", "val", "test"],
                   epoch: int):
        running = {
            "Loss": [],
            "MAE": [],
            "RMSE": [],
            "PSNR": []
        }
        first_prediction = None

        for idx, (LR, HR) in enumerate(tqdm(dataloader, position=0, leave=True)):
            # creating LR and HR tensors for the batch and moving them to the correct device.
            LR = LR.float().to(self.device)
            HR = HR.float().to(self.device)
            HR_for_loss = normalize_target(HR, self.target_mean, self.target_std) if self.normalize_targets else HR

            if epoch == 0 and idx == 0:
                # Profiling for memory usage stats to avoid OOM errors.
                profile_layer_activations(self.model, LR, self.cuda, self.profile_layers_once) 

            # Using autocasting for the forward pass and loss calculation to reduce memory usage.
            # If cuda is false, this will just be a nullcontext and have no effect.
            autocast_ctx = (
                torch.autocast(device_type=self.device, dtype=torch.float16)
                if self.cuda 
                else nullcontext()
            )

            try:
                with autocast_ctx:
                    # Forward pass through the model to get predictions, and calculating the loss with the criterion.
                    y_pred = self.model(LR)
                    validate_batch_shapes(LR, HR_for_loss, y_pred, self.max_pixels_per_image)
                    loss = self.criterion(y_pred, HR_for_loss)
            except RuntimeError as err:
                if "not enough memory" in str(err).lower():
                    # catching OOM errors during the forward pass or loss calculation.
                    if self.cuda:
                        torch.cuda.empty_cache() 
                    raise RuntimeError(
                        "OOM during forward/loss. Reduce BATCH_SIZE, use smaller resize_to, or simplify model."
                    ) from err
                raise
            y_pred_eval = denormalize_target(y_pred, self.target_mean, self.target_std) if self.normalize_targets else y_pred

            mse = mean_squared_error(y_pred_eval.float(), HR)
            running["Loss"].append(loss.item())
            running["MAE"].append(mean_absolute_error(y_pred_eval.float(), HR))
            running["RMSE"].append(root_mean_squared_error(y_pred_eval.float(), HR, mse))
            running["PSNR"].append(peak_signal_to_noise_ratio(y_pred_eval.float(), HR, self.max_pixel_value, mse))

            if training_state == "train":
                # Using set_to_none=True to reduce memory usage by freeing gradients immediately after backward pass.
                self.optimizer.zero_grad(set_to_none=True)

                # Scales the loss for mixed precision (autocast) training to prevent underflow in gradients, 
                # and performs the backward pass.
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if idx == 0:
                if training_state == "test":
                    first_prediction = y_pred_eval.cpu().detach().numpy()
                    plotter().plot_training_images(LR, HR, y_pred_eval, running["Loss"], running["MAE"], running["RMSE"], running["PSNR"])
                else:
                    log_shape_and_memory(training_state, epoch, idx, LR, HR, y_pred_eval, self.cuda)

        return [np.mean(running[key]) for key in ("Loss", "MAE", "RMSE", "PSNR")]

    def prepare_data(self,
                      train_dataset: DatasetInterface, 
                      val_dataset: DatasetInterface, 
                      test_dataset: DatasetInterface, 
                      batch_size: int):
        self.train_dataloader, self.val_dataloader, self.test_dataloader = \
            prepare_dataloader(train_dataset, batch_size, self.cuda), \
            prepare_dataloader(val_dataset, batch_size, self.cuda), \
            prepare_dataloader(test_dataset, batch_size, self.cuda)

        if self.normalize_targets:
            self.target_mean, self.target_std = compute_target_norm_stats(train_dataset, batch_size, self.num_workers, self.target_norm_eps)
        self.max_pixel_value = compute_max_pixel_value(train_dataset + val_dataset + test_dataset, batch_size)

        if self.train_dataloader is None or self.val_dataloader is None or self.test_dataloader is None:
            raise ValueError(
                f"Dataloaders not properly initialized. Train samples: {len(train_dataset)}, "
                f"Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
            )

        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
            raise ValueError(
                f"Empty dataset split detected. Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
            )

        num_train_batches = len(self.train_dataloader)
        num_val_batches = len(self.val_dataloader)
        if num_train_batches == 0 or num_val_batches == 0:
            raise ValueError(
                f"Empty dataloader detected. Train batches: {num_train_batches}, "
                f"Val batches: {num_val_batches}"
            )

    def train(self, num_epochs: int):
        """Returns: training and validation losses and difference in height coefficients per epoch for analysis and debugging.
        Args:
            train_dataset: Dataset for training.
            val_dataset: Dataset for validation.
            test_dataset: Dataset for testing.
            num_epochs: Number of epochs to train.
            batch_size: Batch size for training and validation.
        
        """
        train_losses = []
        train_maes = []
        train_rmses = []
        train_psnrs = []
        val_losses = []
        val_maes = []
        val_rmses = []
        val_psnrs = []

        if self.cuda:
            torch.cuda.reset_peak_memory_stats() #Keeping track of peak memory usage per epoch for debugging OOM errors

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            train_loss, train_mae, train_rmse, train_psnr = self._run_loop(self.train_dataloader, training_state="train", epoch=epoch)

            train_losses.append(train_loss)
            train_maes.append(train_mae)
            train_rmses.append(train_rmse)
            train_psnrs.append(train_psnr)

            # Validation loop, i.e. training loop but without backpropagation and with torch.no_grad() to save memory and computations.
            self.model.eval()
            self.profile_layers_once = False  # profile layers is only relevant for training
            with torch.no_grad():
                val_loss, val_mae, val_rmse, val_psnr = self._run_loop(self.val_dataloader, training_state="val", epoch=epoch)

            val_losses.append(val_loss)
            val_maes.append(val_mae)
            val_rmses.append(val_rmse)
            val_psnrs.append(val_psnr)

            print("-" * 30)
            print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
            print(f"Training MAE EPOCH {epoch + 1}: {train_mae:.4f}")
            print(f"Training RMSE EPOCH {epoch + 1}: {train_rmse:.4f}")
            print(f"Training PSNR EPOCH {epoch + 1}: {train_psnr:.4f}")
            print("\n")
            print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
            print(f"Validation MAE EPOCH {epoch + 1}: {val_mae:.4f}")
            print(f"Validation RMSE EPOCH {epoch + 1}: {val_rmse:.4f}")
            print(f"Validation PSNR EPOCH {epoch + 1}: {val_psnr:.4f}")
            print("-" * 30)

        # Saving the model
        path = 'checkpoints'
        os.makedirs(path, exist_ok = True) 
        torch.save(self.model.state_dict(), os.path.join(path, 'my_checkpoint.pth'))
        return train_losses, train_maes, train_rmses, train_psnrs, val_losses, val_maes, val_rmses, val_psnrs
    
    def test(self):
        """Returns: test loss and difference coefficient for the test dataset.
        """
        self.model.eval()
        test_loss, test_mae, test_rmse, test_psnr = self._run_loop(self.test_dataloader, training_state="test", epoch=0)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test PSNR: {test_psnr:.4f}")
        return test_loss, test_mae, test_rmse, test_psnr

def main():
    current_dir = Path(__file__).resolve().parent
    data_root = current_dir.parent / "data"  # Contains train/, val/, test/
    regions = ["jutland", "funen"]
    data = get_base_dataset(
        lr_data_dir_list=[data_root / "copernicus" / region for region in regions],
        hr_data_dir_list=[data_root / "dataforsyningen" / region for region in regions],
    )

    print(f"Training with dataset sizes - Train: {len(data.train)}, Val: {len(data.val)}, Test: {len(data.test)}")
    
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 3
    EPOCHS = 2
    NORMALIZE_TARGETS = True
    PROFILE_LAYERS_ONCE = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using cuda cores of GPU: {torch.cuda.get_device_name(0)}")
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
        profile_layers_once=PROFILE_LAYERS_ONCE
    )

    trainer.prepare_data(DatasetInterface(data.train), 
                         DatasetInterface(data.val), 
                         DatasetInterface(data.test), 
                         BATCH_SIZE)

    # flattens out at about 38 epochs
    train_losses, train_maes, train_rmses, train_psnrs, val_losses, val_maes, val_rmses, val_psnrs = trainer.train(num_epochs=EPOCHS) 
    
    print(f"Training complete. \n Final training metrics: \nloss: {train_losses[-1]:.4f}, MAE: {train_maes[-1]:.4f}, RMSE: {train_rmses[-1]:.4f}, PSNR: {train_psnrs[-1]:.4f}")
    print(f"Validation metrics: \nloss: {val_losses[-1]:.4f}, MAE: {val_maes[-1]:.4f}, RMSE: {val_rmses[-1]:.4f}, PSNR: {val_psnrs[-1]:.4f}")
    plotter().plot_val_and_train_loss(train_losses, train_maes, train_rmses, train_psnrs, val_losses, val_maes, val_rmses, val_psnrs)

    model_pth = current_dir.parent / "checkpoints" / "my_checkpoint.pth"
    trained_model = UNet(in_channels=1, num_classes=1).to(device)
    trained_model.load_state_dict(torch.load(model_pth, map_location=torch.device(device),weights_only=True))
    trainer.test()

if __name__ == "__main__":
    main()