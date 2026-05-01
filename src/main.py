import os
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from unet import *
from helpers import *
from plotter import plotter
from data_distributor import get_base_dataset
from typing import Literal, Union
import time
from loss_functions import *
from visualiser import visualiser


class ModelPipeline:
    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        plotter: plotter, 
        max_pixels_per_image: int = 1024*1024,
        target_norm_eps: float = 1e-6
    ):
        """ Returns: Self. Initializes the ModelPipeline with the model, optimizer, loss function, device, and learning rate.
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
        self.optimizer = model_config["OPTIMIZER"](model.parameters(), lr=model_config["LEARNING_RATE"])
        self.criterion = model_config["CRITERION"]
        self.device = model_config["DEVICE"]
        self.cuda = self.device == "cuda"
        self.num_workers = 0
        self.max_pixels_per_image = max_pixels_per_image
        self.profile_layers_once = model_config["PROFILE_LAYERS_ONCE"]
        self.normalize_targets = model_config["NORMALIZE_TARGETS"]
        self.target_norm_eps = target_norm_eps
        self.plotter = plotter
        self.epochs = model_config["EPOCHS"]

        # this enables compatibility for older python 3.8.10 i think or maybe linux
        try:
            self.scaler = torch.amp.GradScaler(
                device="cuda", enabled=self.cuda)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cuda)

    def _run_loop(self,
                  dataloader: DataLoader,
                  training_state: Literal["train", "val", "test"],
                  epoch: int,
                  use_amp: bool,
                  test_images: list[torch.Tensor] = None):
        running = {
            "Loss": [],
            "MAE": [],
            "RMSE": [],
            "PSNR": []
        }

        for idx, (LR, HR) in enumerate(tqdm(dataloader, position=0, leave=True)):
            # creating LR and HR tensors for the batch and moving them to the correct device.
            LR = LR.float().to(self.device)
            HR = HR.float().to(self.device)
            HR_for_loss = normalize_target(
                HR, self.target_mean, self.target_std) if self.normalize_targets else HR

            if epoch == 0 and idx == 0:
                # Profiling for memory usage stats to avoid OOM errors.
                profile_layer_activations(
                    self.model, LR, self.cuda, self.profile_layers_once)

            # Using autocasting for the forward pass and loss calculation to reduce memory usage.
            # If cuda is false, this will just be a nullcontext and have no effect.
            autocast_ctx = (
                torch.autocast(device_type=self.device, dtype=torch.float16)
                if use_amp
                else nullcontext()
            )

            try:
                with autocast_ctx:
                    # Forward pass through the model to get predictions, and calculating the loss with the criterion.
                    y_pred = self.model(LR)
                    validate_batch_shapes(
                        LR, HR_for_loss, y_pred, self.max_pixels_per_image)
                with torch.autocast(device_type=self.device, enabled=False):
                    loss = self.criterion(y_pred.float(), HR_for_loss.float())
            except RuntimeError as err:
                if "not enough memory" in str(err).lower():
                    # catching OOM errors during the forward pass or loss calculation.
                    if self.cuda:
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "OOM during forward/loss. Reduce BATCH_SIZE, use smaller resize_to, or simplify model."
                    ) from err
                raise

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected during {training_state} at epoch {epoch + 1}, batch {idx + 1}. "
                    f"LR range=({LR.min().item():.4f}, {LR.max().item():.4f}), "
                    f"HR range=({HR.min().item():.4f}, {HR.max().item():.4f}), "
                    f"pred range=({y_pred.min().item():.4f}, {y_pred.max().item():.4f})"
                )

            y_pred_eval = denormalize_target(
                y_pred, self.target_mean, self.target_std) if self.normalize_targets else y_pred

            mse = mean_squared_error(y_pred_eval.float(), HR)
            running["Loss"].append(loss.item())
            running["MAE"].append(mean_absolute_error(y_pred_eval.float(), HR))
            running["RMSE"].append(root_mean_squared_error(
                y_pred_eval.float(), HR, mse))
            running["PSNR"].append(peak_signal_to_noise_ratio(
                y_pred_eval.float(), HR, self.max_pixel_value, mse))

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
                    self.plotter.plot_training_images(LR=LR, HR=HR, prediction=y_pred_eval,train_loss=running["Loss"], train_mae=running["MAE"], train_rmse=running["RMSE"], train_psnr=running["PSNR"])
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
            self.target_mean, self.target_std = compute_target_norm_stats(
                train_dataset, batch_size, self.num_workers, self.target_norm_eps)
        self.max_pixel_value = compute_max_pixel_value(
            train_dataset + val_dataset + test_dataset, batch_size)

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

    def train(self, retrain=False):
        """Returns: training and validation losses and difference in height coefficients per epoch for analysis and debugging.
        Args:

        """
        if retrain:
            train_losses = []
            train_maes = []
            train_rmses = []
            train_psnrs = []
            val_losses = []
            val_maes = []
            val_rmses = []
            val_psnrs = []

            if self.cuda:
                # Keeping track of peak memory usage per epoch for debugging OOM errors
                torch.cuda.reset_peak_memory_stats()

            for epoch in tqdm(range(self.epochs)):
                self.model.train()
                train_loss, train_mae, train_rmse, train_psnr = self._run_loop(
                    self.train_dataloader, training_state="train", epoch=epoch, use_amp=self.cuda)

                train_losses.append(train_loss)
                train_maes.append(train_mae)
                train_rmses.append(train_rmse)
                train_psnrs.append(train_psnr)

                # Validation loop, i.e. training loop but without backpropagation and with torch.no_grad() to save memory and computations.
                self.model.eval()
                self.profile_layers_once = False  # profile layers is only relevant for training
                with torch.no_grad():
                    val_loss, val_mae, val_rmse, val_psnr = self._run_loop(
                        self.val_dataloader, training_state="val", epoch=epoch, use_amp=self.cuda)

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
                # stopping training if metrics are the same for 10 epochs in a row
                if len(train_losses) > 10 and all(abs(train_losses[-i] - train_losses[-i-1]) < train_losses[-i]*0.2 for i in range(1, 10)):
                    print(f"Training loss has not improved for 10 epochs. Stopping training at epoch {epoch + 1}.")
                    break

            # Saving the model for the current run and timestamping it for archival purposes
            path = 'checkpoints'
            os.makedirs(path, exist_ok=True)
            torch.save(self.model.state_dict(),
                    os.path.join(path, f'{self.model.__class__.__name__}.pth'))
            
            archives=os.path.join(path,'archives')
            os.makedirs(archives, exist_ok=True)

            torch.save(self.model.state_dict(),
                    os.path.join(archives, f'{self.model.__class__.__name__}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth'))

            self.plotter.plot_val_and_train_loss(train_losses, train_maes, train_rmses,
                                                train_psnrs, val_losses, val_maes, val_rmses, val_psnrs)
            
        else:
            if not (Path(__file__).resolve().parent.parent / "checkpoints" / f"{self.model.__class__.__name__}.pth").exists():
                print(f"No existing model weights found for {self.model.__class__.__name__}. Cannot skip retraining.")
                self.train(retrain=True)
                return
            print("Skipping retraining and using existing model weights.")
            model_pth = Path(__file__).resolve().parent.parent / "checkpoints" / f"{self.model.__class__.__name__}.pth"
            self.model.load_state_dict(torch.load(model_pth, map_location=torch.device(self.device), weights_only=True))
            
        
        
    def test(self):
        """Returns: test loss and difference coefficient for the test dataset.
        """
        self.model.eval()
        with torch.no_grad():
            test_loss, test_mae, test_rmse, test_psnr = self._run_loop(self.test_dataloader, training_state="test", epoch=0, use_amp=False)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test PSNR: {test_psnr:.4f}")
        return 

                    




def main():
    current_dir = Path(__file__).resolve().parent
    data_root = current_dir.parent / "data"
    regions = ["jutland", "funen"]
    data = get_base_dataset(
        lr_data_dir_list=[data_root / "copernicus" /
                          region for region in regions],
        hr_data_dir_list=[data_root / "dataforsyningen" /
                          region for region in regions],
    )

    model_config = {
        "LEARNING_RATE": 2e-4,
        "BATCH_SIZE": 3,
        "EPOCHS": 38,
        "PROFILE_LAYERS_ONCE": False,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "OPTIMIZER": optim.AdamW,
        "CRITERION": GradLoss(),
        "NORMALIZE_TARGETS": True
    }
    plotter_instance = plotter(save_dir=current_dir.parent / "checkpoints" / "plots", show_plots=True, save_plots=True)

    unet = UNet(in_channels=1, num_classes=1).to(model_config["DEVICE"])
    unet_pipeline = ModelPipeline(
        unet,
        model_config,
        plotter=plotter_instance
    )

    unet_pipeline.prepare_data(data.train,
                               data.val,
                               data.test,
                               model_config["BATCH_SIZE"])

    # flattens out at about 38 epochs
    unet_pipeline.train(retrain=False)

    unet_pipeline.test()

    regions = ["jutland", "zealand", "bornholm"]
    visualization_data = get_base_dataset(
        lr_data_dir_list=[data_root / "selected" / "lr" / region for region in regions],
        hr_data_dir_list=[data_root / "selected" / "hr" / region for region in regions],
        division=DataDivision(train=0.0, val=0.0, test=1.0),
        randomize=False
    ).test

    visualiser([unet_pipeline], plotter_instance, visualization_data, model_config["DEVICE"], max_pixel_value=unet_pipeline.max_pixel_value)

    print("finished main")

if __name__ == "__main__":
    main()
