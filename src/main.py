import copy
import logging
import torch
from torch import optim
from pathlib import Path
from unet import UNet
from helpers import *  # noqa: F403
from plotter import plotter
from data_distributor import get_base_dataset, DataDivision, dataset_to_downsampled_dataset, unshuffle_dataloader
import time
from loss_functions import *  # noqa: F403
from visualiser import visualiser
from metrics import *  # noqa: F403
from logsrn import LoGSRN
from modelpipeline import ModelPipeline


def main(logger):
    current_dir = Path(__file__).resolve().parent

    # Initializing hyperparameters, metrics and configurations for the model pipeline
    metrics = {"MAE": MAE, "MSE": MSE, "RMSE": RMSE, "PSNR": PSNR, "SSIM": SSIM}
    model_config = {
        "LEARNING_RATE": 5e-5,
        "DYNAMIC_LR": True,
        "BATCH_SIZE": 16,
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

        loss_functions = [MAESSIMLoss(alpha=0.5, data_range=datarange_for_loss), SmoothGradLoss(), SmoothLoss(beta=0.5), MSESSIMLoss(alpha=0.5,data_range=datarange_for_loss), SSIMLoss(data_range=datarange_for_loss), MSSSIMLoss(data_range=datarange_for_loss)]


        for model in models:
            for criterion in loss_functions:
                for config in configs:
                    config["data"] = datas
                    pipeline = ModelPipeline(model=model, model_config=config, plotter=plotter_instance, logger=logger, criterion=criterion, downsampled_data=i==1)
                    pipeline.train(retrain=False)
                    pipeline.test()
                    
                    pipeline_dict[pipeline.pth_path_name] = pipeline


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
        [pipeline_dict["UNet_SSIMLoss_AdamW"], pipeline_dict["UNet_SmoothLoss_AdamW"], pipeline_dict["UNet_MSESSIMLoss_AdamW"],pipeline_dict["UNet_MAESSIMLoss_AdamW"], pipeline_dict["UNet_MSESSIMLoss_AdamW_downsampled"]],
        plotter_instance,
        visualization_data,
        [unshuffle_dataloader(loader) for loader in data[:3]] + [evaluation_data, visualization_data],
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
