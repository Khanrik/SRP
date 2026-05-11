from helpers import results, normalize_targets, denormalize_target
import torch
from tqdm import tqdm
from inspect import signature

def visualiser(ModelPipelineList, plotter_instance, selected_test_images, device, metrics, min_val=0.0, max_val=1.0, mean=None, std=None):
    """Returns: None. Tests multiple model pipelines and prints their test losses and difference coefficients for comparison.
    Args:
        ModelPipelineList: A list of ModelPipeline instances to be tested.
        plotter_instance: An instance of the plotter class for visualization.
        selected_test_images: A list of lr and hr image pairs to be used for testing and visualization.
        metrics: A dictionary of metric functions to be used for evaluation.
        min_val: The minimum possible value of the images.
        max_val: The maximum possible value of the images.
        mean: The mean value for normalization.
        std: The standard deviation for normalization.
    """
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
                name=f"{pipeline.model.__class__.__name__} with {pipeline.criterion.__class__.__name__}",
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
    
    return