from helpers import *
from skimage.metrics import structural_similarity as ssim

def visualiser(ModelPipelineList, plotter_instance, selected_test_images, device, max_pixel_value: float | None = None):
    """Returns: None. Tests multiple model pipelines and prints their test losses and difference coefficients for comparison.
    Args:
        ModelPipelineList: A list of ModelPipeline instances to be tested.
        plotter_instance: An instance of the plotter class for visualization.
        selected_test_images: A list of lr and hr image pairs to be used for testing and visualization.
    """
    if not ModelPipelineList:
        raise ValueError("visualiser requires at least one ModelPipeline instance.")

    if max_pixel_value is None:
        max_pixel_value = ModelPipelineList[0].max_pixel_value

    def _safe_ssim(pred_tensor, tgt_tensor, max_pixel_value):
        # Convert tensors to numpy images and handle channel ordering for skimage
        p = pred_tensor.detach().cpu().numpy()
        t = tgt_tensor.detach().cpu().numpy()
        # remove batch dim if present
        if p.ndim == 4:
            p = p[0]
        if t.ndim == 4:
            t = t[0]

        # If channels-first (C,H,W) -> transpose to (H,W,C)
        if p.ndim == 3 and p.shape[0] in (1, 3, 4):
            p = p.transpose(1, 2, 0)
            t = t.transpose(1, 2, 0)

        # Now p and t should be HxW or HxWxC
        h, w = p.shape[0], p.shape[1]
        min_side = min(h, w)
        # skimage default win_size is 7; ensure odd and <= min_side
        win_size = 7
        if min_side < win_size:
            win_size = min_side if (min_side % 2 == 1) else max(1, min_side - 1)
        if win_size < 3:
            return float('nan')

        # Try calling structural_similarity with channel_axis (newer skimage) or multichannel (older)
        try:
            if p.ndim == 3:
                return float(ssim(p, t, data_range=max_pixel_value, channel_axis=2, win_size=win_size))
            else:
                return float(ssim(p, t, data_range=max_pixel_value, win_size=win_size))
        except TypeError:
            # fallback for older skimage versions
            if p.ndim == 3:
                return float(ssim(p, t, data_range=max_pixel_value, multichannel=True, win_size=win_size))
            else:
                return float(ssim(p, t, data_range=max_pixel_value, win_size=win_size))


    def _metric_items(prediction, normalized_prediction, target, normalized_target, max_pixel_value, mse=None):
        if mse is None:
            mse = mean_squared_error(prediction.float(), target)
        mae = mean_absolute_error(prediction.float(), target)
        rmse = root_mean_squared_error(prediction.float(), target, mse)
        psnr = peak_signal_to_noise_ratio(normalized_prediction.float(), normalized_target)
        ssim_value = _safe_ssim(prediction, target, max_pixel_value)
        return [
            ("MAE", mae),
            ("RMSE", rmse),
            ("PSNR", psnr),
            ("SSIM", ssim_value),
        ]

    images = prepare_dataloader(
        selected_test_images,
        batch_size=1,
        pin_memory=device == "cuda",
        shuffle_bool=False
    )
    test_result=[]
    for LR, HR in tqdm(images, position=0, leave=True):
        image_result = []
        # creating LR and HR tensors for the batch and moving them to the correct device.
        LR = LR.float().to(device)
        HR = HR.float().to(device)
        normalized_LR, _, min_val, max_val = normalize_targets(LR)
        normalized_HR, _, _, _ = normalize_targets(HR)
        # create bilinear upsampled image for comparison in the horizontal results plot, and calculate metrics for it as well.
        bilinear = torch.nn.functional.interpolate(
            LR,
            size=HR.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        bilinear_mse = mean_squared_error(bilinear.float(), HR)
        # add original low resolution image
        image_result.append(
            results(
                image=LR[0],
                name="LR Input",
                metrics=[("MSE", 0), ("MAE", 0), ("RMSE", 0), ("PSNR", 0)],
            ),
        )
        image_result.append(
            results(
                image=bilinear[0],
                name="bilinear",
                metrics=_metric_items(bilinear, normalize_targets(bilinear)[0], HR, normalized_HR, max_pixel_value, bilinear_mse),
            )
        )
        for pipeline in ModelPipelineList:
            pipeline.model.eval()
            with torch.no_grad():
                y_pred = pipeline.model(normalized_LR)
                y_pred_eval = denormalize_target(y_pred, min_val, max_val)
            pred_mse = mean_squared_error(y_pred_eval.float(), HR)
                            
            pred_results=results(
                image=y_pred_eval[0],
                name=pipeline.model.__class__.__name__,
                metrics=_metric_items(y_pred_eval, y_pred, HR, normalized_HR, max_pixel_value, pred_mse),
            )
            image_result.append(pred_results)
        
        # add ground truth image
        hr_mse = mean_squared_error(HR.float(), HR)
        image_result.append(
            results(
                image=HR[0],
                name="GT",
                metrics=_metric_items(HR, normalized_HR, HR, normalized_HR, max_pixel_value, hr_mse),
            )
        )
        
        test_result.extend([image_result])
    plotter_instance.plot_horizontal_results(test_result, interpolation="nearest")
    
    return