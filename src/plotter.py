
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader
from shapely.geometry import box
import geopandas as gpd
from helpers import normalize_targets, denormalize_target
import torch
from tqdm import tqdm

# data and metrics
class plotter:
    def __init__(self, save_dir="plots",show_plots=False,save_plots=False):
        self.save_dir = Path(save_dir)
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _imshow(self, ax, tensor, *, interpolation="nearest"):
        ax.imshow(tensor, cmap="gray" if tensor.ndim == 2 else None, interpolation=interpolation)

    def plot_val_and_train_loss(self, train_metrics, val_metrics):
        """Returns: Self.
        Args:
            train_metrics: A dictionary containing lists of training metrics (e.g., loss, MAE, RMSE, PSNR) for each epoch.
            val_metrics: A dictionary containing lists of validation metrics (e.g., loss, MAE, RMSE, PSNR) for each epoch.
            display_plots: Whether to display the plots interactively.
        """
        if self.save_plots or self.show_plots:
            epochs = range(1, len(train_metrics['Loss']) + 1)
            # if there are over 4 metrics, split them into multiple figures to avoid overcrowding.
            num_figures = (len(train_metrics) + 3) // 4
            for fig_idx in range(num_figures):
                fig = plt.figure(figsize=(12, 8))
                for i, metric_name in enumerate(list(train_metrics.keys())[fig_idx * 4:(fig_idx + 1) * 4]):
                    train_values = train_metrics[metric_name]
                    val_values = val_metrics[metric_name]
                    plt.subplot(2, 2, i + 1)
                    plt.plot(epochs, train_values, label=f'Train {metric_name}')
                    plt.plot(epochs, val_values, label=f'Val {metric_name}')
                    plt.xlabel('Epoch')
                    plt.ylabel(metric_name)
                    plt.title(f'Training and Validation {metric_name}')
                    plt.legend()
                
                fig.tight_layout()

                if self.save_plots and self.save_dir:
                    fig.savefig(os.path.join(self.save_dir, f'training_validation_metrics_fig{fig_idx + 1}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png'), dpi=300, bbox_inches="tight")

            if self.show_plots:
                plt.show()

    def _to_plot_array(self, tensor):
        # Handling the batch dimension and channel dimension for plotting.
        tensor = tensor.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]  # Display first image in the batch.
        if tensor.ndim == 3:
            if tensor.shape[0] == 1:
                tensor = tensor[0]
            elif tensor.shape[0] in (3, 4):
                tensor = tensor.permute(1, 2, 0)
            else:
                tensor = tensor[0]
        if tensor.ndim != 2 and tensor.ndim != 3:
            raise ValueError(f"Unsupported tensor shape for plotting: {tuple(tensor.shape)}")
        return tensor.numpy()



    def plot_horizontal_results(self, results_list, interpolation="nearest", max_rows_per_figure=3):
        """Plot one sample or many samples in a single figure.

        Args:
            results_list: Either a flat list of results for one sample, or a nested
                list where each inner list contains the results for one sample.
            interpolation: Matplotlib interpolation mode used for all images.
            max_rows_per_figure: Maximum number of sample rows to render per figure.
        """
        if not (self.save_plots or self.show_plots):
            return

        if not results_list:
            raise ValueError("results_list cannot be empty.")

        is_nested = isinstance(results_list[0], list)
        sample_groups = results_list if is_nested else [results_list]

        for group in sample_groups:
            if len(group) != len(sample_groups[0]):
                raise ValueError("All result groups must contain the same number of entries.")

        num_cols = len(sample_groups[0])
        column_titles = [result.name for result in sample_groups[0]]

        pages = [sample_groups[start:start + max_rows_per_figure] for start in range(0, len(sample_groups), max_rows_per_figure)]

        for page_idx, page_groups in enumerate(pages, start=1):
            num_rows = len(page_groups)
            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=(4 * num_cols, 3.6 * num_rows),
                squeeze=False,
            )

            for row_idx, group in enumerate(page_groups):
                for col_idx, result in enumerate(group):
                    ax = axes[row_idx][col_idx]

                    img = result.image.detach().cpu()
                    if img.ndim == 3:
                        if img.shape[0] in [1, 3]:
                            img = img.permute(1, 2, 0)

                    img = img.numpy()
                    if img.ndim == 3 and img.shape[-1] == 1:
                        img = img.squeeze(-1)

                    self._imshow(ax, img, interpolation=interpolation)
                    if row_idx == 0:
                        ax.set_title(column_titles[col_idx], fontsize=12, pad=8)
                    if col_idx == 0:
                        sample_number = (page_idx - 1) * max_rows_per_figure + row_idx + 1
                        ax.set_ylabel(f"Sample {sample_number}", fontsize=11, rotation=0, labelpad=32, va="center")
                    ax.axis("off")

                    metrics_text = "\n".join(
                        f"{metric_name} {metric_value:.4f}"
                        for metric_name, metric_value in result.metrics
                    )
                    ax.text(
                        0.5,
                        -0.12,
                        metrics_text,
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=8,
                    )

            fig.suptitle(f"Horizontal Results Page {page_idx}/{len(pages)}", fontsize=13, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.97))

            if self.save_plots and self.save_dir:
                fig.savefig(
                    os.path.join(
                        self.save_dir,
                        f"horizontal_results_page{page_idx}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )

        if self.show_plots:
            plt.show()


    def plot_datasplit_map(self, loaders: list[DataLoader], crs="EPSG:25832"):
        if not (self.save_plots or self.show_plots):
            return
        records = []

        for loader in loaders:
            dataset = loader.dataset
            for idx in range(len(dataset)):
                bbox = dataset.get_bbox(idx)
                records.append({
                    "category": dataset.category,
                    "geometry": box(
                        bbox.left,
                        bbox.bottom,
                        bbox.right,
                        bbox.top
                    )
                })
        
        gdf = gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs=crs
        )

        fig, ax = plt.subplots(figsize=(12, 8))

        gdf.plot(
            column="category",
            categorical=True,
            cmap="Set1",
            legend=True,
            edgecolor="black",
            linewidth=0.2,
            ax=ax
        )

        ax.set_title("Data Split Map", fontsize=14, pad=12)
        ax.set_axis_off()
        fig.tight_layout()

        if self.save_plots and self.save_dir:
            fig.savefig(
                os.path.join(
                    self.save_dir,
                    f"datasplit_map_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
        if self.show_plots:
            plt.show()


    def plot_metric_maps(self, loaders: list[DataLoader], pipeline, metrics: dict, crs="EPSG:25832"):
        if not (self.save_plots or self.show_plots):
            return

        num_cols = len(metrics)

        fig, axes = plt.subplots(
            1,
            num_cols,
            figsize=(4 * num_cols, 6),
            squeeze=False,
        )

        for col_idx, (metric_name, metric_func) in enumerate(metrics.items()):
            ax = axes[0][col_idx]
            records = []

            for loader in loaders:
                dataset = loader.dataset
                # Pair each loaded batch with its exact dataset indices (works with shuffle and any batch size).
                for batch_indices, (LR_batch, HR_batch) in tqdm(
                    zip(loader.batch_sampler, loader),
                    desc=f"Processing {metric_name} map",
                    total=len(loader),
                ):
                    LR_batch = LR_batch.float().to(pipeline.device)
                    HR_batch = HR_batch.float().to(pipeline.device)

                    if isinstance(batch_indices, torch.Tensor):
                        batch_indices = batch_indices.tolist()

                    pipeline.model.eval()
                    for sample_offset, dataset_idx in enumerate(batch_indices):
                        LR = LR_batch[sample_offset : sample_offset + 1]
                        HR = HR_batch[sample_offset : sample_offset + 1]
                        normalized_LR, normalized_HR, min_val, max_val = normalize_targets(LR, HR)
                        with torch.no_grad():
                            y_pred = pipeline.model(normalized_LR)
                            y_pred_eval = denormalize_target(y_pred, min_val, max_val)

                        metric_value = metric_func(y_pred_eval, HR)
                        if isinstance(metric_value, torch.Tensor):
                            metric_value = metric_value.detach().cpu().item()

                        bbox = dataset.get_bbox(int(dataset_idx))
                        records.append({
                            "metric_value": metric_value,
                            "geometry": box(
                                bbox.left,
                                bbox.bottom,
                                bbox.right,
                                bbox.top
                            )
                        })

            gdf = gpd.GeoDataFrame(
                records,
                geometry="geometry",
                crs=crs
            )
            gdf.plot(
                column="metric_value",
                cmap="viridis",
                legend=True,
                edgecolor="black",
                linewidth=0.2,
                ax=ax
            )

            ax.set_title(f"{metric_name} Map", fontsize=12, pad=8)
            ax.set_axis_off()

        fig.tight_layout()

        if self.save_plots and self.save_dir:
            fig.savefig(
                os.path.join(
                    self.save_dir,
                    f"metric_maps_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )

        if self.show_plots:
            plt.show()
