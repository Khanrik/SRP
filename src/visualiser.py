from helpers import results, normalize_targets, denormalize_target
import torch
from tqdm import tqdm
from plotter import plotter
from inspect import signature

def visualiser(ModelPipelineList, plotter_instance: plotter, selected_test_images, denmark_data: list, device, metrics, include_maps = False, include_constant_maps = False, min_val=0.0, max_val=1.0, mean=None, std=None):
    """Returns: None. Tests multiple model pipelines and prints their test losses and difference coefficients for comparison.
    Args:
        ModelPipelineList: A list of ModelPipeline instances to be tested.
        plotter_instance: An instance of the plotter class for visualization.
        selected_test_images: A list of lr and hr image pairs to be used for testing and visualization.
        denmark_data: All denmark data to be used for map visualization.
        device: The device to run the model on.
        metrics: A dictionary of metric functions to be used for evaluation.
        include_maps: A boolean indicating whether to include the data split map in the visualization.
        include_constant_maps: A boolean indicating whether to include constant maps in the visualization.
        min_val: The minimum possible value of the images.
        max_val: The maximum possible value of the images.
        mean: The mean value for normalization.
        std: The standard deviation for normalization.
    """
    if not (plotter_instance.save_plots or plotter_instance.show_plots):
        return
    
    if not ModelPipelineList:
        raise ValueError("visualiser requires at least one ModelPipeline instance.")




    def _metric_items(prediction, target):
        metric_items = []
        for metric_name, metric_func in metrics.items():
            input_parameters = signature(metric_func).parameters.keys()
            if "data_range" in input_parameters:
                metric_value = metric_func(prediction.float(), target, data_range=max_val - min_val)
            elif "max_value" in input_parameters:
                metric_value = metric_func(prediction.float(), target, max_value=max_val)
            else:
                metric_value = metric_func(prediction.float(), target)
            metric_items.append((metric_name, metric_value))
        return metric_items

    test_result=[]
    for LR, HR in tqdm(selected_test_images, position=0, leave=True):
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
                metrics=_metric_items(nearest, HR)
            )
        )
        image_result.append(
            results(
                image=bilinear[0],
                name="bilinear",
                metrics=_metric_items(bilinear, HR)
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
                metrics=_metric_items(y_pred_eval, HR),
            )
            image_result.append(pred_results)
        
        # add ground truth image
        image_result.append(
            results(
                image=HR[0],
                name="GT",
                metrics=_metric_items(HR, HR),
            )
        )
        
        test_result.extend([image_result])
    plotter_instance.plot_horizontal_results(test_result, interpolation="nearest")
    
    if include_maps:

        if include_constant_maps:
            plotter_instance.plot_datasplit_map(denmark_data)
            plotter_instance.plot_extrema_map(denmark_data)
        
        best_pipeline = plotter_instance.plot_boxplots(denmark_data, ModelPipelineList, metrics["SSIM"], mean_val=mean, std_val=std)
        plotter_instance.plot_metric_maps(denmark_data, best_pipeline, metrics, mean_val=mean, std_val=std)
    
    return