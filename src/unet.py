import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from contextlib import nullcontext
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from unet_helper import *
from data_distributor import DataPair, get_base_dataset

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
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.cuda)
        self.max_pixels_per_image = max_pixels_per_image
        self.profile_layers_once = profile_layers_once
        self.normalize_targets = normalize_targets
        self.target_norm_eps = target_norm_eps

    def train(self, 
              train_dataset: DatasetInterface, 
              val_dataset: DatasetInterface, 
              test_dataset: DatasetInterface, 
              num_epochs: int, 
              batch_size: int):
        """Returns: training and validation losses and difference in height coefficients per epoch for analysis and debugging.
        Args:
            train_dataset: Dataset for training.
            val_dataset: Dataset for validation.
            test_dataset: Dataset for testing.
            num_epochs: Number of epochs to train.
            batch_size: Batch size for training and validation.
        
        """
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

        train_losses = []
        train_maes = []
        train_rmses = []
        train_psnrs = []
        val_losses = []
        val_maes = []
        val_rmses = []
        val_psnrs = []

        for epoch in tqdm(range(num_epochs)):
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats() #Keeping track of peak memory usage per epoch for debugging OOM errors

            self.model.train()
            train_running_loss = []
            train_running_mae = []
            train_running_rmse = []
            train_running_psnr = []
            for idx, LR_HR in enumerate(tqdm(self.train_dataloader, position=0, leave=True)):
                # creating LR and HR tensors for the batch and moving them to the correct device.
                LR = LR_HR[0].float().to(self.device) 
                HR = LR_HR[1].float().to(self.device)

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
                        if self.device == "cuda":
                            torch.cuda.empty_cache() 
                        raise RuntimeError(
                            "OOM during forward/loss. Reduce BATCH_SIZE, use smaller resize_to, or simplify model."
                        ) from err
                    raise
                
                # Using set_to_none=True to reduce memory usage by freeing gradients immediately after backward pass.
                self.optimizer.zero_grad(set_to_none=True) 
                y_pred_eval = denormalize_target(y_pred, self.target_mean, self.target_std) if self.normalize_targets else y_pred

                mse = mean_squared_error(y_pred_eval.float(), HR)
                train_running_loss.append(loss.item())
                train_running_mae.append(mean_absolute_error(y_pred_eval.float(), HR))
                train_running_rmse.append(root_mean_squared_error(y_pred_eval.float(), HR, mse))
                train_running_psnr.append(peak_signal_to_noise_ratio(y_pred_eval.float(), HR, self.max_pixel_value, mse))
                # Scales the loss for mixed precision (autocast) training to prevent underflow in gradients, 
                # and performs the backward pass.
                self.scaler.scale(loss).backward() 
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if idx == 0:
                    log_shape_and_memory("train", epoch, idx, LR, HR, y_pred_eval, self.cuda)

            train_loss = np.mean(train_running_loss)
            train_mae = np.mean(train_running_mae)
            train_rmse = np.mean(train_running_rmse)
            train_psnr = np.mean(train_running_psnr)

            train_losses.append(train_loss)
            train_maes.append(train_mae)
            train_rmses.append(train_rmse)
            train_psnrs.append(train_psnr)

            # Validation loop, i.e. training loop but without backpropagation and with torch.no_grad() to save memory and computations.
            self.model.eval()
            val_running_loss = []
            val_running_mae = []
            val_running_rmse = []
            val_running_psnr = []

            with torch.no_grad():
                for idx, LR_HR in enumerate(tqdm(self.val_dataloader, position=0, leave=True)):
                    LR = LR_HR[0].float().to(self.device)
                    HR = LR_HR[1].float().to(self.device)
                    HR_for_loss = normalize_target(HR, self.target_mean, self.target_std) if self.normalize_targets else HR

                    autocast_ctx = (
                        torch.autocast(device_type=self.device, dtype=torch.float16)
                        if self.cuda
                        else nullcontext()
                    )
                    with autocast_ctx:
                        y_pred = self.model(LR)
                        validate_batch_shapes(LR, HR_for_loss, y_pred, self.max_pixels_per_image)
                        loss = self.criterion(y_pred, HR_for_loss)
                    y_pred_eval = denormalize_target(y_pred, self.target_mean, self.target_std) if self.normalize_targets else y_pred
                    
                    mse = mean_squared_error(y_pred_eval.float(), HR)
                    val_running_loss.append(loss.item())
                    val_running_mae.append(mean_absolute_error(y_pred_eval.float(), HR))
                    val_running_rmse.append(root_mean_squared_error(y_pred_eval.float(), HR, mse))
                    val_running_psnr.append(peak_signal_to_noise_ratio(y_pred_eval.float(), HR, self.max_pixel_value, mse))

                    if idx == 0:
                        log_shape_and_memory("val", epoch, idx, LR, HR, y_pred_eval, self.cuda)

                val_loss = np.mean(val_running_loss)
                val_mae = np.mean(val_running_mae)
                val_rmse = np.mean(val_running_rmse)
                val_psnr = np.mean(val_running_psnr)
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

class Tester:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module = nn.SmoothL1Loss(),
        normalize_targets: bool = False,
        target_mean: float = None,
        target_std: float = None,
        max_pixel_value: float = None
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
        self.max_pixel_value = max_pixel_value

    def test(self, test_dataloader):
        """Returns: test loss and difference coefficient for the test dataset.
        Args:
            test_dataloader: DataLoader for the test dataset.
        
        """
        self.model.eval()
        test_running_loss = []
        test_running_mae = []
        test_running_rmse = []
        test_running_psnr = []
        first_prediction = None

        with torch.no_grad():
            for idx, LR_HR in enumerate(tqdm(test_dataloader, position=0, leave=True)):
                LR = LR_HR[0].float().to(self.device)
                HR = LR_HR[1].float().to(self.device)

                y_pred = self.model(LR)
                y_pred_eval = denormalize_target(y_pred, self.target_mean, self.target_std) if self.normalize_targets else y_pred

                loss = self.criterion(y_pred_eval, HR)
                mse = mean_squared_error(y_pred_eval.float(), HR)
                mae = mean_absolute_error(y_pred_eval, HR)
                rmse = root_mean_squared_error(y_pred_eval, HR, mse)
                psnr = peak_signal_to_noise_ratio(y_pred_eval, HR, self.max_pixel_value, mse)

                test_running_loss.append(loss.item())
                test_running_mae.append(mae)
                test_running_rmse.append(rmse)
                test_running_psnr.append(psnr)

                if idx == 0:
                    first_prediction = y_pred_eval.cpu().detach().numpy()
                    plotter().plot_training_images(LR, HR, y_pred_eval, test_running_loss, test_running_mae, test_running_rmse, test_running_psnr)

        test_loss = np.mean(test_running_loss)
        test_mae = np.mean(test_running_mae)
        test_rmse = np.mean(test_running_rmse)
        test_psnr = np.mean(test_running_psnr)

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
    EPOCHS = 32
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

    # flattens out at about 38 epochs
    train_losses, train_maes, train_rmses, train_psnrs, val_losses, val_maes, val_rmses, val_psnrs = \
        trainer.train(DatasetInterface(data.train), 
                      DatasetInterface(data.val), 
                      DatasetInterface(data.test), 
                      num_epochs=EPOCHS, batch_size=BATCH_SIZE) 
    
    print(f"Training complete. \n Final training metrics: \nloss: {train_losses[-1]:.4f}, MAE: {train_maes[-1]:.4f}, RMSE: {train_rmses[-1]:.4f}, PSNR: {train_psnrs[-1]:.4f}")
    print(f"Validation metrics: \nloss: {val_losses[-1]:.4f}, MAE: {val_maes[-1]:.4f}, RMSE: {val_rmses[-1]:.4f}, PSNR: {val_psnrs[-1]:.4f}")
    plotter().plot_val_and_train_loss(train_losses, train_maes, train_rmses, train_psnrs, val_losses, val_maes, val_rmses, val_psnrs)

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
        max_pixel_value=trainer.max_pixel_value
    )
    tester.test(trainer.test_dataloader)

if __name__ == "__main__":
    main()

