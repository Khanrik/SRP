from helpers import results, normalize_targets, denormalize_target, metric_items
import torch
from tqdm import tqdm
from plotter import plotter
from torch.utils.data import DataLoader
from logging import Logger

def visualiser(ModelPipelineList: list, 
               plotter_instance: plotter, 
               selected_test_images: DataLoader, 
               denmark_data: list[DataLoader], 
               device: str, 
               metrics: dict,
               logger: Logger,
               mean: float, 
               std: float, 
               min_val: float, 
               max_val: float,
               include_table: bool = False,
               include_maps: bool = False, 
               include_constant_maps: bool = False, 
               boxplots: bool = True, 
               box_metric: str = 'SSIM'):
    """Visualizes the results of multiple model pipelines on a set of selected test images, and optionally includes metric maps and boxplots for comparison.
    Args:
        ModelPipelineList (list): A list of ModelPipeline instances to be tested.
        plotter_instance (plotter): An instance of the plotter class for visualization.
        selected_test_images (DataLoader): A DataLoader containing the test images to be used for testing and visualization.
        denmark_data (list[DataLoader]): All denmark data to be used for map visualization.
        device (str): The device to run the model on.
        metrics (dict): A dictionary of metric functions to be used for evaluation.
        mean (float): The mean value for normalization.
        std (float): The standard deviation for normalization.
        min_val (float): The minimum possible value of the images.
        max_val (float): The maximum possible value of the images.
        logger (Logger): A logger instance for logging the typst table of metrics.
        include_table (bool, optional): A boolean indicating whether to include a typst table of metrics in the visualization. Default is False.
        include_maps (bool, optional): A boolean indicating whether to include the data split map in the visualization. Default is False.
        include_constant_maps (bool, optional): A boolean indicating whether to include constant maps in the visualization. Default is False.
        boxplots (bool, optional): A boolean indicating whether to include boxplots in the visualization. Default is True.
        box_metric (str, optional): The name of the metric to be used for the boxplots. Default is 'SSIM'.
    Returns:
        None. The function generates visualizations based on the provided parameters and saves or shows the plots using the plotter instance.
    """
    if not (plotter_instance.save_plots or plotter_instance.show_plots):
        return
    
    if not ModelPipelineList:
        raise ValueError("visualiser requires at least one ModelPipeline instance.")


    test_result=[]
    for LR, HR, _ in tqdm(selected_test_images, position=0, leave=True):
        image_result = []
        # creating LR and HR tensors for the batch and moving them to the correct device.
        LR = LR.float().to(device)
        HR = HR.float().to(device)
        normalized_LR = normalize_targets(targets=LR, mean=mean, std=std)
        # create bilinear upsampled image for comparison in the horizontal results plot, and calculate metrics for it as well.
        bilinear = torch.nn.functional.interpolate(
            LR,
            size=HR.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        nearest = torch.nn.functional.interpolate(
            LR,
            size=HR.shape[-2:],
            mode="nearest",
        )
        # add original low resolution image
        image_result.append(
            results(
                image=LR[0],
                name="LR Input",
                metrics=metric_items(nearest, HR, metrics, min_val, max_val)
            )
        )
        image_result.append(
            results(
                image=bilinear[0],
                name="bilinear",
                metrics=metric_items(bilinear, HR, metrics, min_val, max_val)
            )
        )
        for pipeline in ModelPipelineList:
            pipeline.model.eval()
            with torch.no_grad():
                y_pred = pipeline.model(normalized_LR)
                y_pred_eval = denormalize_target(y_pred, mean=mean, std=std)
                            
            pred_results=results(
                image=y_pred_eval[0],
                name=f"{pipeline.model.__class__.__name__} with {pipeline.criterion.__class__.__name__} and {pipeline.optimizer.__class__.__name__}",
                metrics=metric_items(y_pred_eval, HR, metrics, min_val, max_val),
            )
            image_result.append(pred_results)
        
        # add ground truth image
        image_result.append(
            results(
                image=HR[0],
                name="GT",
                metrics=metric_items(HR, HR, metrics, min_val, max_val),
            )
        )
        
        test_result.extend([image_result])
    plotter_instance.plot_horizontal_results(test_result, interpolation="nearest")
    
    if boxplots or include_maps or include_table:
        plotter_instance.get_dataframe(denmark_data, ModelPipelineList, metrics, min_val=min_val, max_val=max_val, mean_val=mean, std_val=std)

    if boxplots:
        best_pipeline = plotter_instance.plot_boxplots(metric_name=box_metric) 

    if include_maps:
        plotter_instance.plot_metric_maps(best_pipeline, metrics)
        
        if include_constant_maps:
            plotter_instance.plot_datasplit_map()
            plotter_instance.plot_extrema_map()
    
    if include_table:
        plotter_instance.log_typst_table(logger, metrics)
    
    return