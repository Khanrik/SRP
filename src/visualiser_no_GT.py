from typing import Tuple

from helpers import results, normalize_targets, denormalize_target, metric_items
import torch
from tqdm import tqdm
from plotter import plotter
import numpy as np

def visualiser_no_GT(ModelPipelineList: list, plotter_instance: plotter, selected_test_images: list, device: str):
    """Visualiser function for data with no ground truth. This is currently set up to visualize the Ethiopia data, which has no HR data available.
    Args:
        ModelPipelineList (list): A list of ModelPipeline instances to be visualized.
        plotter_instance (plotter): An instance of the plotter class to be used for plotting the results.
        selected_test_images (list): A list of tuples containing the LR and HR images for the selected test samples. The HR images may be dummy images if no GT is available, but they are still required for the function to run.
        device (str): The device on which the model is running (e.g., "cuda" or "cpu").
    Returns:
        None. The function generates visualizations based on the provided parameters and saves or shows the plots using the plotter instance.
    """
    if not (plotter_instance.save_plots or plotter_instance.show_plots):
        return
    
    if not ModelPipelineList:
        raise ValueError("visualiser_no_GT requires at least one ModelPipeline instance.")


    test_result=[]
    for LR,_ in tqdm(selected_test_images, position=0, leave=True):
        image_result = []
        # creating LR tensors for the batch and moving them to the correct device.
        LR = LR.float().to(device)
        normalized_LR = normalize_targets(targets=LR, mean=torch.mean(LR), std=torch.std(LR))
        # create bilinear upsampled image for comparison in the horizontal results plot, and calculate metrics for it as well.
        bilinear = torch.nn.functional.interpolate(
            LR,
            size=[size*3 for size in LR.shape[-2:]],
            mode="bilinear",
            align_corners=False,
        )
        
        # add original low resolution image
        image_result.append(
            results(
                image=LR[0],
                name="LR Input",
                metrics=[]
            )
        )
        image_result.append(
            results(
                image=bilinear[0],
                name="bilinear",
                metrics=[]
            )
        )
        for pipeline in ModelPipelineList:
            pipeline.model.eval()
            with torch.no_grad():
                y_pred = pipeline.model(normalized_LR)
                y_pred_eval = denormalize_target(y_pred, mean=torch.mean(LR), std=torch.std(LR))
                            
            pred_results=results(
                image=y_pred_eval[0],
                name=f"{pipeline.model.__class__.__name__} with {pipeline.criterion.__class__.__name__} and {pipeline.optimizer.__class__.__name__}",
                metrics=[]
            )
            image_result.append(pred_results)
        
        test_result.extend([image_result])
    plotter_instance.plot_horizontal_results(test_result, interpolation="nearest")
    
    return