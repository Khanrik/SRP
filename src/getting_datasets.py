from data_distributor import get_base_dataset, DataDivision
from pathlib import Path
from itertools import chain
import torch
import logging

def getting_datasets(training_regions: list[str], evaluation_regions: list[str], visualization_regions: list[str], visual_eval_regions: list[str], model_config: dict, logger: logging.Logger) -> tuple[tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, float, float, float, float], list[tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, float, float, float, float]]]:
    """Fetches and prepares datasets for training, evaluation, and visualization based on the specified regions. The function checks if the necessary data for the specified regions already exists in the output path, and if not, it calls the appropriate functions to fetch and process the data for those regions. It then prepares the datasets for training, evaluation, and visualization by calling the 'get_base_dataset' function with the appropriate parameters for each category of data.
    Args:
        training_regions (list[str]): List of region names for which to prepare training data.
        evaluation_regions (list[str]): List of region names for which to prepare evaluation data.
        visualization_regions (list[str]): List of region names for which to prepare visualization data.
        visual_eval_regions (list[str]): List of region names for which to prepare visual evaluation data with no HR available.
        model_config (dict): Configuration dictionary for the model.
        logger (logging.Logger): Logger for logging information during dataset preparation.
    Returns:
        tuple[tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, float, float, float, float]]: A tuple containing the prepared datasets for training, evaluation, visualization, and visual evaluation. The datasets are returned as tuples of (train_data, val_data, test_data, min, max, mean, std) for each category.
    Note:
        visual_eval_data does not include the min max std or mean
    """
    data_root = Path(__file__).resolve().parent.parent / "data"

    training_data = get_base_dataset(
        lr_data_dir_list=[data_root / "copernicus" / region for region in training_regions],
        hr_data_dir_list=[data_root / "dataforsyningen" / region for region in training_regions],
        batch_size=model_config.get("BATCH_SIZE", 16),
        cuda=torch.cuda.is_available(),
        division=DataDivision(train=0.8, val=0.1, test=0.1),
        logger=logger
    )

    evaluation_data = get_base_dataset(
        lr_data_dir_list=[data_root / "copernicus" / region for region in evaluation_regions],
        hr_data_dir_list=[data_root / "dataforsyningen" / region for region in evaluation_regions],
        batch_size=model_config.get("BATCH_SIZE", 16),
        cuda=torch.cuda.is_available(),
        division=DataDivision(train=0.0, val=0.0, test=1.0),
        category="evaluation",
        logger=logger
    )

    # visualization always uses batch size of 1.

    visualization_data = get_base_dataset(
        lr_data_dir_list=[data_root / "selected" / "lr" / region for region in visualization_regions],
        hr_data_dir_list=[data_root / "selected" / "hr" / region for region in visualization_regions],
        batch_size=1,
        cuda=torch.cuda.is_available(),
        division=DataDivision(train=0.0, val=0.0, test=1.0),
        category="visualization",
        logger=logger
    )
    individual_visual_eval=[]
    for region in visual_eval_regions:
        region_test_data = get_base_dataset(
            lr_data_dir_list=[data_root / "selected" / "lr" / region],
            hr_data_dir_list=[data_root / "selected" / "hr" / region],
            batch_size=1,
            cuda=model_config["DEVICE"] == "cuda",
            division=DataDivision(train=0.0, val=0.0, test=1.0),
            randomize=False,
            category="visual_evaluation",
            logger=logger,
            same_as_lr_regions={region}
        )
        individual_visual_eval.append(region_test_data[2])  # only test data is needed for visual evaluation
    visual_eval_data = list(chain.from_iterable(individual_visual_eval))

    return training_data, evaluation_data, visualization_data, visual_eval_data
