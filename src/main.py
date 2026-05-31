import os
import copy
import logging
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from torch import optim
from tqdm import tqdm
from pathlib import Path
from unet import UNet
from helpers import *  # noqa: F403
from plotter import plotter
from data_distributor import get_base_dataset, DataDivision, dataset_to_downsampled_dataset, loader_to_downsampled_loader
from typing import Literal
import time
from loss_functions import *  # noqa: F403
from visualiser import visualiser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import *  # noqa: F403
from inspect import signature
from logsrn import LoGSRN
from cnn import CNN



class ModelPipeline:
    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        plotter: plotter,
        logger: logging.Logger,
        max_pixels_per_image: int = 1024 * 1024,
        target_norm_eps: float = 1e-6,
        criterion: nn.Module = GradLoss()
    ):
        """Returns: Self. Initializes the ModelPipeline with the model, optimizer, loss function, device, and learning rate.
        Args:
            model: The neural network model.
            model_config: A dictionary containing model configuration parameters such as learning rate, device, optimizer, etc.
            plotter: An instance of the plotter class for visualization.
            max_pixels_per_image: An integer specifying the maximum number of pixels per image to prevent OOM errors.
            target_norm_eps: A small float value to prevent division by zero during normalization of targets.
            criterion: The loss function to be used for training the model. Defaults to GradLoss.
        """
        self.model = copy.deepcopy(model)
        self.model = self.model.to(model_config["DEVICE"])
        self.optimizer = model_config["OPTIMIZER"](
            self.model.parameters(), lr=model_config["LEARNING_RATE"]
        )
        self.criterion = criterion
        self.device = model_config["DEVICE"]
        self.cuda = self.device == "cuda"
        self.num_workers = 0
        self.max_pixels_per_image = max_pixels_per_image
        self.profile_layers_once = model_config["PROFILE_LAYERS_ONCE"]
        self.target_norm_eps = target_norm_eps
        self.plotter = plotter
        self.epochs = model_config["EPOCHS"]
        self.metrics = model_config["METRICS"]
        self.global_norm = model_config["GLOBAL_NORMALIZATION"]
        self.timer = model_config["TIMER"]
        self.train_dataloader = model_config["data"][0]
        self.val_dataloader = model_config["data"][1]
        self.test_dataloader = model_config["data"][2]
        self.min_pixel_value = model_config["data"][3]
        self.max_pixel_value = model_config["data"][4]
        self.mean_pixel_value = model_config["data"][5]
        self.std_pixel_value = model_config["data"][6]
        self.train_time = 0
        self.val_time = 0

        self.logger = logger
        self.logger.info(f"\n\nInitialized ModelPipeline for {self.model.__class__.__name__}_{self.criterion.__class__.__name__}_{self.optimizer.__class__.__name__}")

        # Use ReduceLROnPlateau to adapt LR based on validation loss
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        ) if model_config["DYNAMIC_LR"] else None
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_count = 0
        self.patience_limit = 5

        # this enables compatibility for older python 3.8.10 i think or maybe linux
        try:
            self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.cuda)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cuda)

    def _run_loop(
        self,
        dataloader,
        training_state: Literal["train", "val", "test"],
        epoch: int,
        use_amp: bool,
        test_images: list[torch.Tensor] = None,
    ):
        running = {}
        for metric_name in self.metrics.keys():
            running[metric_name] = []
        running["Loss"] = []

        for idx, (LR, HR) in enumerate(tqdm(dataloader, position=0, leave=True, desc=f"Epoch {epoch + 1} {training_state} - {self.model.__class__.__name__}_{self.criterion.__class__.__name__}_{self.optimizer.__class__.__name__}")):
            # creating LR and HR tensors for the batch and moving them to the correct device.
            LR = LR.float().to(self.device)
            HR = HR.float().to(self.device)
            normalized_LR, normalized_HR = normalize_targets(targets=[LR, HR], mean=self.mean_pixel_value, std=self.std_pixel_value)

            if epoch == 0 and idx == 0:
                # Profiling for memory usage stats to avoid OOM errors.
                profile_layer_activations(self.model, normalized_LR, self.cuda, self.profile_layers_once, logger=self.logger)

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
                    y_pred = self.model(normalized_LR)
                    validate_batch_shapes(normalized_LR, normalized_HR, y_pred, self.max_pixels_per_image)
                with torch.autocast(device_type=self.device, enabled=False):
                    loss = self.criterion(y_pred.float(), normalized_HR.float())
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

            y_pred_denorm = denormalize_target(y_pred, mean=self.mean_pixel_value, std=self.std_pixel_value)

            running["Loss"].append(loss.item())
            for metric_name, metric_fn in self.metrics.items():
                input_parameters = signature(metric_fn).parameters.keys()
                if "data_range" in input_parameters:  
                    metric_value = metric_fn(y_pred_denorm.float(), HR, data_range=self.max_pixel_value - self.min_pixel_value)
                elif "max_value" in input_parameters:
                    metric_value = metric_fn(y_pred_denorm.float(), HR, max_value=self.max_pixel_value)
                else:
                    metric_value = metric_fn(y_pred_denorm.float(), HR)
                running[metric_name].append(metric_value)

            if training_state == "train":
                # Using set_to_none=True to reduce memory usage by freeing gradients immediately after backward pass.
                self.optimizer.zero_grad(set_to_none=True)

                # Scales the loss for mixed precision (autocast) training to prevent underflow in gradients,
                # and performs the backward pass.
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return [np.mean(running[key]) for key in ["Loss"] + list(self.metrics.keys())]

    

    def train(self, retrain=False, pth_path_name=None):
        """Returns: training and validation losses and difference in height coefficients per epoch for analysis and debugging.
        Args:

        """
        
        if pth_path_name is not None:
            pass
        elif self.optimizer.__class__.__name__ != "AdamW":
            pth_path_name = f"{self.model.__class__.__name__}_{self.criterion.__class__.__name__}_{self.optimizer.__class__.__name__}"
        else:
            pth_path_name = f"{self.model.__class__.__name__}_{self.criterion.__class__.__name__}"

        if retrain:
            print(f"Starting training for {pth_path_name}")
            # initializing metrics
            timers = {
                "train": 0.0,
                "val": 0.0
            }
            train_metrics = {}
            val_metrics = {}
            train_metrics["Loss"] = []
            val_metrics["Loss"] = []
            for metric_name in self.metrics.keys():
                train_metrics[metric_name] = []
                val_metrics[metric_name] = []

            if self.cuda:
                # Keeping track of peak memory usage per epoch for debugging OOM errors
                torch.cuda.reset_peak_memory_stats()

            # looping through epochs
            for epoch in tqdm(range(self.epochs)):
                time_start = time.time()
                self.model.train()
                curr_metrics = self._run_loop(
                    self.train_dataloader,
                    training_state="train",
                    epoch=epoch,
                    use_amp=self.cuda,
                )
                timers["train"] += time.time() - time_start

                self.logger.info("-" * 30)
                train_metrics["Loss"].append(curr_metrics[0])
                for i, metric_name in enumerate(self.metrics.keys()):
                    train_metrics[metric_name].append(curr_metrics[i + 1])
                    self.logger.info(
                        f"Epoch {epoch + 1} Train {metric_name}: {curr_metrics[i + 1]:.4f}"
                    )
                self.logger.info(
                    f"Epoch {epoch + 1} Train Loss: {curr_metrics[0]:.4f}"
                )
                # Validation loop, i.e. training loop but without backpropagation and with torch.no_grad() to save memory and computations.
                time_start = time.time()
                self.model.eval()
                self.profile_layers_once = (
                    False  # profile layers is only relevant for training
                )
                with torch.no_grad():
                    curr_val_metrics = self._run_loop(
                        self.val_dataloader,
                        training_state="val",
                        epoch=epoch,
                        use_amp=self.cuda,
                    )
                timers["val"] += time.time() - time_start

                self.logger.info("")

                val_metrics["Loss"].append(curr_val_metrics[0])
                for i, metric_name in enumerate(self.metrics.keys()):
                    val_metrics[metric_name].append(curr_val_metrics[i + 1])
                    self.logger.info(
                        f"Epoch {epoch + 1} Val {metric_name}: {curr_val_metrics[i + 1]:.4f}"
                    )
                val_loss = curr_val_metrics[0]
                self.logger.info(
                    f"Epoch {epoch + 1} Val Loss: {val_loss:.4f}"
                )

                self.logger.info("-" * 30)
                self.logger.info(f"Currently training {pth_path_name}")
                
                # Early stopping based on validation loss
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_count = 0
                    self.logger.info(f"Best validation loss: {val_loss:.4f}.")
                else:
                    self.patience_count += 1
                    self.logger.info(f"No improvement in validation loss. Patience: {self.patience_count}/{self.patience_limit}")
                    if self.patience_count >= self.patience_limit:
                        self.logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                        break

            # Saving the model for the current run and timestamping it for archival purposes
            path = "checkpoints"
            os.makedirs(path, exist_ok=True)
            torch.save(
                self.model.state_dict(),
                os.path.join(path, f"{pth_path_name}.pth"),
            )

            archives = os.path.join(path, "archives")
            os.makedirs(archives, exist_ok=True)

            torch.save(
                self.model.state_dict(),
                os.path.join(
                    archives,
                    f"{pth_path_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth",
                ),
            )

            self.plotter.plot_val_and_train_loss(train_metrics, val_metrics)
            if self.timer:
                self.train_time = timers["train"]
                self.val_time = timers["val"]
                self.logger.info(f"Total training time: {timers['train']:.2f} seconds. Average time per epoch: {timers['train']/len(train_metrics['Loss']):.2f} seconds.")
                self.logger.info(f"Total validation time: {timers['val']:.2f} seconds. Average time per epoch: {timers['val']/len(val_metrics['Loss']):.2f} seconds.")


        else:
            if not (
                Path(__file__).resolve().parent.parent
                / "checkpoints"
                / f"{pth_path_name}.pth"
            ).exists():
                self.logger.warning(
                    f"No existing model weights found for {self.model.__class__.__name__} with criterion {self.criterion.__class__.__name__} and optimizer {self.optimizer.__class__.__name__}. Cannot skip retraining."
                )
                self.train(retrain=True, pth_path_name=pth_path_name)
                return
            self.logger.info("Skipping retraining and using existing model weights.")
            if pth_path_name is not None:
                self.logger.info(f"Loading model weights from specified path: {pth_path_name}")
                model_pth = (
                    Path(__file__).resolve().parent.parent
                    / "checkpoints"
                    / f"{pth_path_name}.pth"
                )
            else:
                model_pth = (
                    Path(__file__).resolve().parent.parent
                    / "checkpoints"
                    / f"{pth_path_name}.pth"
                )
            self.model.load_state_dict(
                torch.load(
                    model_pth, map_location=torch.device(self.device), weights_only=True
                )
            )      
            
        

    def test(self):
        """Returns: test loss and difference coefficient for the test dataset."""
        
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            test_metrics = self._run_loop(
                self.test_dataloader, training_state="test", epoch=0, use_amp=False
            )

        self.logger.info(f"Test loss: {test_metrics[0]:.4f}")
        for i, metric_name in enumerate(self.metrics.keys()):
            self.logger.info(f"Test {metric_name}: {test_metrics[i + 1]:.4f}")
            
        if self.timer:
            test_time = time.time() - start_time
            self.logger.info(f"Total testing time: {test_time:.2f} seconds.")
            self.logger.info(f"Total running time: {(test_time + self.train_time + self.val_time):.2f} seconds.")
        return


def main(logger):
    current_dir = Path(__file__).resolve().parent

    # Initializing hyperparameters, metrics and configurations for the model pipeline
    metrics = {"MAE": MAE, "MSE": MSE, "RMSE": RMSE, "PSNR": PSNR, "SSIM": SSIM}
    model_config = {
        "LEARNING_RATE": 5e-5,
        "DYNAMIC_LR": True,
        "BATCH_SIZE": 3,
        "EPOCHS": 1000000,
        "PROFILE_LAYERS_ONCE": False,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "OPTIMIZER": optim.AdamW,
        "METRICS": metrics,
        "GLOBAL_NORMALIZATION": True,
        "TIMER": True
    }
    plotter_instance = plotter(
        save_dir=current_dir.parent / "checkpoints" / "plots",
        show_plots=False,
        save_plots=True,
    )
    

    # Initializing data
    data_root = current_dir.parent / "data"
    regions = ["jutland", "funen"]
    data = get_base_dataset(
        lr_data_dir_list=[data_root / "copernicus" / region for region in regions],
        hr_data_dir_list=[data_root / "dataforsyningen" / region for region in regions],
        batch_size=model_config["BATCH_SIZE"],
        cuda=model_config["DEVICE"] == "cuda",
        include_plot=False,
        logger=logger,
    )
    downsampled_data = dataset_to_downsampled_dataset(data, downsample_factor=3, logger=logger)

    model_config_SGD = copy.deepcopy(model_config)
    model_config_SGD["OPTIMIZER"] = optim.SGD
    model_config_RMS = copy.deepcopy(model_config)
    model_config_RMS["OPTIMIZER"] = optim.RMSprop

    # Creating models
    unet_model = UNet(in_channels=1, num_classes=1).to(model_config["DEVICE"])
    LoGSRN_model = LoGSRN(in_channels=1, num_classes=1).to(model_config["DEVICE"])
    models = [unet_model, LoGSRN_model]
    configs = [model_config, model_config_RMS]
    pipeline_dict = {}

    for i,datas in enumerate([data, downsampled_data]):
        
        datarange_for_loss=(data[4] - data[3])/data[6]  # (max - min) / std for global normalization, used for SSIM data_range parameter

        loss_functions = [MAESSIMLoss(alpha=0.5, data_range=datarange_for_loss), SmoothGradLoss(),GradLoss(), SmoothLoss(beta=0.5),MSESSIMLoss(alpha=0.5,data_range=datarange_for_loss), SSIMLoss(data_range=datarange_for_loss), MSSSIMLoss(data_range=datarange_for_loss)]


        for model in models:
            for criterion in loss_functions:
                for config in configs:
                    config["data"] = datas
                    pipeline = ModelPipeline(model=model, model_config=config, plotter=plotter_instance, logger=logger, criterion=criterion)
                    if config["OPTIMIZER"] == optim.AdamW:
                        pth_path_name = f"{model.__class__.__name__}_{criterion.__class__.__name__}"
                    else:
                        pth_path_name = model.__class__.__name__ + "_" + criterion.__class__.__name__ + "_" + config["OPTIMIZER"].__name__
                    if i == 1:  
                        pth_path_name += "_downsampled"
                    pipeline.train(retrain=False, pth_path_name=pth_path_name)
                    pipeline.test()
                    
                    pipeline_dict[f"{model.__class__.__name__}_{criterion.__class__.__name__}_{config['OPTIMIZER'].__name__}_{i}"] = pipeline


    # visualization 
    regions = ["jutland", "zealand", "bornholm"]
    visualization_data = get_base_dataset(
        lr_data_dir_list=[data_root / "selected" / "lr" / region for region in regions],
        hr_data_dir_list=[data_root / "selected" / "hr" / region for region in regions],
        batch_size=1,
        cuda=model_config["DEVICE"] == "cuda",
        division=DataDivision(train=0.0, val=0.0, test=1.0),
        randomize=False,
        category="visualization",
        logger=logger,
    )[2]  # only test data is needed for visualization

    regions = ["zealand", "bornholm"]
    evaluation_data = get_base_dataset(
        lr_data_dir_list=[data_root / "copernicus" / region for region in regions],
        hr_data_dir_list=[data_root / "dataforsyningen" / region for region in regions],
        batch_size=model_config["BATCH_SIZE"],
        cuda=model_config["DEVICE"] == "cuda",
        division=DataDivision(train=0.0, val=0.0, test=1.0),
        category="evaluation",
        randomize=False,
        logger=logger,
    )[2]
    
    visualiser(
        [pipeline_dict["UNet_SSIMLoss_AdamW_0"], pipeline_dict["UNet_SmoothLoss_AdamW_0"], pipeline_dict["UNet_MSESSIMLoss_AdamW_0"],pipeline_dict["UNet_MAESSIMLoss_AdamW_0"], pipeline_dict["UNet_MSESSIMLoss_AdamW_1"]],
        plotter_instance,
        visualization_data,
        list(data[:3]) + [evaluation_data, visualization_data],
        model_config["DEVICE"],
        metrics,
        min_val=data[3],
        max_val=data[4],
        mean=data[5],
        std=data[6],
        include_maps=True,
        include_constant_maps=True # only worth running once to get the map saved
    )

    print("Finished running main")
    logger.info("Finished running main")


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent

    logfile = current_dir.parent / "checkpoints" / "logs" / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(logfile),
                        format='%(asctime)s %(levelname)s: %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    try:
        main(logger)
    except Exception as e:
        logger.exception(f"An error occurred during execution: {str(e)}")
        raise e
