import copy
import logging
import torch
from torch import optim
from pathlib import Path
from unet import UNet
from helpers import *  # noqa: F403
from plotter import plotter
from data_distributor import dataset_to_downsampled_dataset, unshuffle_dataloader
import time
from loss_functions import *  # noqa: F403
from visualiser import visualiser
from metrics import *  # noqa: F403
from logsrn import LoGSRN
from modelpipeline import ModelPipeline
from main_data_fetcher import get_denmark_data, get_ethiopia_data, move_to_selected
from visualiser_no_GT import visualiser_no_GT
from getting_datasets import getting_datasets
from inspect import signature

def datafetching_and_processing(logger: logging.Logger, VisualEvaluationData: list[str] = ["ethiopia"]):
    """
    Fetches and processes data for the specified regions.
    Args:
        logger (logging.Logger): Logger for logging information during data fetching and processing.
        VisualEvaluationData (list[str]): List of region names for which to fetch and process data for visual evaluation. The function will fetch and process data for these regions if it does not already exist in the specified output path. Default is ["ethiopia"].
    Note:
        The function first checks if the necessary data for the specified regions already exists in the output path. If not, it calls the appropriate functions to fetch and process the data for those regions. After fetching and processing the data, it moves select files to a 'selected' directory for easier access during visualization and evaluation.
        
        Currently, the function is set up to fetch and process data for Denmark and Ethiopia, but it can be easily extended to include additional regions by adding the necessary logic for fetching and processing data for those regions, as well as updating the list of files to move in the 'move_to_selected' function.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    lr_target_resolution = (128, 128)
    hr_target_resolution = 10
    

    if not (data_dir / "copernicus").exists() or not (data_dir / "dataforsyningen").exists():
        logger.info("Downloading and processing Denmark data...")
        get_denmark_data(output_path=data_dir, 
                        lr_target_resolution=lr_target_resolution, 
                        hr_target_resolution=hr_target_resolution,
                        include_merge=True)
        
    logger.info("Moving select files to 'selected' directory...")
    move_to_selected(output_path=data_dir)
    
    for region in VisualEvaluationData:
        if not (data_dir / region).exists():
            logger.info(f"Downloading and processing {region} data...")
            if region == "ethiopia":
                get_ethiopia_data(output_path=data_dir, 
                                target_resolution=lr_target_resolution,
                                include_merge=True)
    logger.info("Done fetching data!")


def pipelines_creator(datasets: list, loss_functions: list, models: list, configs: list, plotter_instance: plotter, logger: logging.Logger)-> dict[str, ModelPipeline]:
    pipeline_dict = {}
    for i, datas in enumerate(datasets):
        
        datarange_for_loss=(datas[4] - datas[3])/datas[6]  # (max - min) / std for global normalization, used for SSIM data_range parameter
        losses_initialized = []

        available_args = {
            "data_range": datarange_for_loss,
            "alpha": 0.5,
            "beta": 0.5,
        }

        for loss in loss_functions:
            params = signature(loss).parameters
            kwargs = {k: v for k, v in available_args.items() if k in params}
            losses_initialized.append(loss(**kwargs))
        
        for model in models:
            for criterion in losses_initialized:
                for config in configs:
                    config["data"] = datas
                    pipeline = ModelPipeline(model=model, model_config=config, plotter=plotter_instance, logger=logger, criterion=criterion, downsampled_data=i==1)
                    pipeline.train(retrain=False)
                    
                    pipeline_dict[pipeline.pth_path_name] = pipeline
    return pipeline_dict

def main(logger: logging.Logger):

    # Note that currently only Ethiopia is included in the visual evaluation,
    # but if more regions are wished for they can be added in main_data_fetcher.py
    datafetching_and_processing(logger, VisualEvaluationData=["ethiopia"])

    # Initializing hyperparameters, metrics and configurations for the model pipeline
    current_dir = Path(__file__).resolve().parent
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

    training_data, evaluation_data, visualization_data, visual_eval_data = getting_datasets(training_regions=["jutland", "funen"], evaluation_regions=["bornholm"], visualization_regions=["bornholm"], visual_eval_regions=["ethiopia"],model_config=model_config, logger=logger)

    downsampled_data = dataset_to_downsampled_dataset(training_data, downsample_factor=3, logger=logger)

    # model_config_SGD = copy.deepcopy(model_config)
    # model_config_SGD["OPTIMIZER"] = optim.SGD
    model_config_RMS = copy.deepcopy(model_config)
    model_config_RMS["OPTIMIZER"] = optim.RMSprop

    # Creating models
    unet_model = UNet(in_channels=1, num_classes=1).to(model_config["DEVICE"])
    LoGSRN_model = LoGSRN(in_channels=1, num_classes=1).to(model_config["DEVICE"])

    # all available losses are:
    # SmoothLoss, SmoothGradLoss, SSIMLoss, MSESSIMLoss, MAESSIMLoss, MSSSIMLoss
    pipeline_dict = pipelines_creator(
        datasets=[training_data, downsampled_data],
        loss_functions=[SmoothLoss, SmoothGradLoss, SSIMLoss, MSESSIMLoss, MAESSIMLoss, MSSSIMLoss],
        models=[unet_model, LoGSRN_model],
        configs=[model_config, model_config_RMS],
        plotter_instance=plotter_instance,
        logger=logger
    )

    # jutlandfunen_test = [unshuffle_dataloader(loader) for loader in training_data[2]]  # only test data is needed for evaluation
    visualiser(
        list(pipeline_dict.values()),
        plotter_instance,
        visualization_data[2],
        [evaluation_data[2], visualization_data[2]],
        model_config["DEVICE"],
        metrics,
        logger=logger,
        min_val=training_data[3],
        max_val=training_data[4],
        mean=training_data[5],
        std=training_data[6],
        boxplots=True,
        box_metric='SSIM',
        include_maps=True,
        include_constant_maps=False # only worth running once to get the map saved
    )

    visualiser_no_GT(
        list(pipeline_dict.values()),
        plotter_instance,
        visual_eval_data,  # only test data is needed for visualization
        model_config["DEVICE"],
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
