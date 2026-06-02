from helpers import results, normalize_targets, denormalize_target, metric_items
import torch
from tqdm import tqdm
from plotter import plotter

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
    
    if include_maps:
        plotter_instance.get_dataframe(denmark_data, ModelPipelineList, metrics, min_val=min_val, max_val=max_val)

        if include_constant_maps:
            plotter_instance.plot_datasplit_map()
            plotter_instance.plot_extrema_map()
        
        best_pipeline = plotter_instance.plot_boxplots()
        plotter_instance.plot_metric_maps(best_pipeline, metrics)
        plotter_instance.log_typst_table(metrics)
    
    return