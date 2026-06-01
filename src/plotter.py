
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from shapely.geometry import box
import geopandas as gpd
from helpers import normalize_targets, denormalize_target
import torch
from tqdm import tqdm

# data and metrics
class plotter:
    def __init__(self, save_dir="plots", show_plots=False, save_plots=False):
        self.save_dir = Path(save_dir)
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._sample_cache: dict[tuple[int, ...], list[dict[str, object]]] = {}

    def _imshow(self, ax, tensor, *, interpolation="nearest"):
        ax.imshow(tensor, cmap="gray" if tensor.ndim == 2 else None, interpolation=interpolation)

    def _save_figure(self, fig, filename_prefix):
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if self.save_plots and self.save_dir:
            fig.savefig(
                os.path.join(
                    self.save_dir,
                    f"{filename_prefix}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )

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
                
                self._save_figure(fig, f"train_val_metrics_{fig_idx + 1}")

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
                        ax.set_title(column_titles[col_idx], fontsize=11, pad=10)
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
            # ensure layout is tightened to avoid title overlap when showing
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            self._save_figure(fig, f"horizontal_results_page_{page_idx}")

        if self.show_plots:
            plt.show()


    def _get_sample_records(self, loaders: list[DataLoader]):
        cache_key = tuple(id(loader) for loader in loaders)
        cached_records = self._sample_cache.get(cache_key)
        if cached_records is not None:
            return cached_records

        sample_records: list[dict[str, object]] = []

        for loader in loaders:
            dataset = loader.dataset
            for batch_indices, (LR_batch, HR_batch) in tqdm(
                zip(loader.batch_sampler, loader),
                desc=f"Caching samples for {dataset.category} dataset",
                total=len(loader),
            ):
                if isinstance(batch_indices, torch.Tensor):
                    batch_indices = batch_indices.tolist()

                LR_batch = LR_batch.detach().cpu()
                HR_batch = HR_batch.detach().cpu()

                for sample_offset, dataset_idx in enumerate(batch_indices):
                    bbox = dataset.get_bbox(int(dataset_idx))
                    sample_records.append({
                        "category": dataset.category,
                        "dataset_idx": int(dataset_idx),
                        "geometry": box(
                            bbox.left,
                            bbox.bottom,
                            bbox.right,
                            bbox.top,
                        ),
                        "LR": LR_batch[sample_offset : sample_offset + 1],
                        "HR": HR_batch[sample_offset : sample_offset + 1],
                    })

        self._sample_cache[cache_key] = sample_records
        return sample_records


    def plot_datasplit_map(self, loaders: list[DataLoader], crs="EPSG:25832"):
        if not (self.save_plots or self.show_plots):
            return
        sample_records = self._get_sample_records(loaders)
        records = [
            {
                "category": record["category"],
                "geometry": record["geometry"],
            }
            for record in sample_records
        ]
        
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
        
        self._save_figure(fig, "datasplit_map")
        
        if self.show_plots:
            plt.show()


    def plot_extrema_map(self, loaders: list[DataLoader], crs="EPSG:25832"):
        if not (self.save_plots or self.show_plots):
            return

        records = {
            "lr_min": [],
            "lr_max": [],
            "hr_min": [],
            "hr_max": [],
        }

        for record in self._get_sample_records(loaders):
            lr_sample = record["LR"]
            hr_sample = record["HR"]
            geometry = record["geometry"]
            lr_min = torch.min(lr_sample).detach().cpu().item()
            lr_max = torch.max(lr_sample).detach().cpu().item()
            hr_min = torch.min(hr_sample).detach().cpu().item()
            hr_max = torch.max(hr_sample).detach().cpu().item()

            records["lr_min"].append({"chunk_min": lr_min, "geometry": geometry})
            records["lr_max"].append({"chunk_max": lr_max, "geometry": geometry})
            records["hr_min"].append({"chunk_min": hr_min, "geometry": geometry})
            records["hr_max"].append({"chunk_max": hr_max, "geometry": geometry})

        if not any(records.values()):
            return

        gdfs = {
            name: gpd.GeoDataFrame(items, geometry="geometry", crs=crs)
            for name, items in records.items()
            if items
        }

        fig, axes = plt.subplots(2, 2, figsize=(16, 14), squeeze=False)

        plot_specs = [
            ("lr_min", "chunk_min", "LR Chunk Minimum Map", "viridis", axes[0][0]),
            ("lr_max", "chunk_max", "LR Chunk Maximum Map", "plasma", axes[0][1]),
            ("hr_min", "chunk_min", "HR Chunk Minimum Map", "viridis", axes[1][0]),
            ("hr_max", "chunk_max", "HR Chunk Maximum Map", "plasma", axes[1][1]),
        ]

        for key, column, title, cmap, ax in plot_specs:
            gdfs[key].plot(
                column=column,
                cmap=cmap,
                legend=True,
                edgecolor="black",
                linewidth=0.2,
                ax=ax,
            )
            ax.set_title(title, fontsize=12, pad=8)
            ax.set_axis_off()

        self._save_figure(fig, "extrema_maps")

        if self.show_plots:
            plt.show()


    def plot_boxplots(self, loaders: list[DataLoader], model_pipeline_list, metric_func, mean_val: float, std_val: float):
        pipeline_labels = [f"{pipeline.model.__class__.__name__} with {pipeline.criterion.__class__.__name__} and {pipeline.optimizer.__class__.__name__}" for pipeline in model_pipeline_list]
        pipeline_scores: list[list[float]] = [[] for _ in model_pipeline_list]

        sample_records = self._get_sample_records(loaders)

        for p_idx, pipeline in enumerate(model_pipeline_list):
            pipeline.model.eval()

            for record in tqdm(sample_records, desc=f"Evaluating {pipeline.name}", leave=False):
                LR = normalize_targets(record["LR"], mean=mean_val, std=std_val).to(pipeline.device)
                HR = record["HR"].to(pipeline.device)

                with torch.no_grad():
                    y_pred = pipeline.model(LR)
                    y_pred_eval = denormalize_target(y_pred, mean=mean_val, std=std_val)

                metric_val = float(metric_func(y_pred_eval.float(), HR))
                pipeline_scores[p_idx].append(metric_val)

        def _average_score(values: list[float]) -> float:
            finite_values = [value for value in values if np.isfinite(value)]
            return np.mean(finite_values)

        average_scores = [_average_score(scores) for scores in pipeline_scores]
        best_pipeline_idx = int(np.argmax(average_scores))
        best_pipeline = model_pipeline_list[best_pipeline_idx]

        fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(model_pipeline_list)), 6))
        bp = ax.boxplot(pipeline_scores, labels=pipeline_labels, vert=True, patch_artist=True)
        ax.set_title("SSIM Distribution by Pipeline", fontsize=14, pad=12)
        ax.set_ylabel("SSIM")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

        # Annotate quartiles Q0..Q4 (0=min, 1=Q1, 2=median, 3=Q3, 4=max) for each pipeline
        ylim = ax.get_ylim()
        y_span = ylim[1] - ylim[0] if ylim[1] > ylim[0] else 1.0
        label_offset = y_span * 0.02
        quartile_percents = [0, 25, 50, 75, 100]
        quartile_labels = ["Q0", "Q1", "Q2", "Q3", "Q4"]
        for idx, scores in enumerate(pipeline_scores, start=1):
            finite_scores = [v for v in scores if np.isfinite(v)]
            if not finite_scores:
                continue
            qvals = np.percentile(finite_scores, quartile_percents)
            for q_idx, qv in enumerate(qvals):
                # stagger label placement to avoid overlap: top quartiles placed above, bottom below
                if q_idx >= 3:
                    y = float(qv) + label_offset * (q_idx - 2)
                    va = "bottom"
                elif q_idx == 2:
                    y = float(qv)
                    va = "center"
                else:
                    y = float(qv) - label_offset * (1 - q_idx)
                    va = "top"
                ax.text(idx, y, f"{quartile_labels[q_idx]} {float(qv):.4f}", ha="center", va=va, fontsize=7, color="black")
            
            lower_whisk = np.min(bp['whiskers'][2 * (idx - 1)].get_ydata())
            upper_whisk = np.max(bp['whiskers'][2 * (idx - 1) + 1].get_ydata())
            ax.text(idx, lower_whisk, f"WL {float(lower_whisk):.4f}", ha="center", va="top", fontsize=7, color="blue")
            ax.text(idx, upper_whisk, f"WH {float(upper_whisk):.4f}", ha="center", va="bottom", fontsize=7, color="blue")

        self._save_figure(fig, "ssim_boxplots")

        if self.show_plots:
            plt.show()

        return best_pipeline


    def plot_metric_maps(self, loaders: list[DataLoader], model_pipeline, metrics: dict, mean_val: float, std_val: float, crs="EPSG:25832"):
        if not (self.save_plots or self.show_plots):
            return
        num_cols = len(metrics)
        fig, axes = plt.subplots(
            2,
            num_cols,
            figsize=(4 * num_cols, 9),
            gridspec_kw={"height_ratios": [3, 1]},
            squeeze=False,
        )

        records_by_metric = {metric_name: [] for metric_name in metrics.keys()}
        values_by_metric = {metric_name: [] for metric_name in metrics.keys()}

        sample_records = self._get_sample_records(loaders)

        model_pipeline.model.eval()
        for record in sample_records:
            LR = record["LR"].float().to(model_pipeline.device)
            HR = record["HR"].float().to(model_pipeline.device)
            normalized_LR = normalize_targets(LR, mean=mean_val, std=std_val)
            with torch.no_grad():
                y_pred = model_pipeline.model(normalized_LR)
                y_pred_eval = denormalize_target(y_pred, mean=mean_val, std=std_val)

            geometry = record["geometry"]

            # Calculate all metrics for this sample.
            for metric_name, metric_func in metrics.items():
                metric_value = metric_func(y_pred_eval, HR)
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.detach().cpu().item()

                values_by_metric[metric_name].append(metric_value)

                records_by_metric[metric_name].append({
                    "metric_value": metric_value,
                    "geometry": geometry,
                })

        for col_idx, (metric_name, metric_func) in enumerate(metrics.items()):
            map_ax = axes[0][col_idx]
            gdf = gpd.GeoDataFrame(
                records_by_metric[metric_name],
                geometry="geometry",
                crs=crs
            )
            gdf.plot(
                column="metric_value",
                cmap="viridis",
                legend=True,
                edgecolor="black",
                linewidth=0.2,
                ax=map_ax
            )
            map_ax.set_title(f"{metric_name}", fontsize=12, pad=8)
            map_ax.set_axis_off()

            boxplot_ax = axes[1][col_idx]
            finite_values = [value for value in values_by_metric[metric_name] if np.isfinite(value)]
            if finite_values:
                boxplot_ax.boxplot(finite_values, vert=True, patch_artist=True)
            else:
                boxplot_ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            boxplot_ax.set_xticks([])
            boxplot_ax.grid(axis="y", alpha=0.3)
            
        self._save_figure(fig, "metric_maps")

        if self.show_plots:
            plt.show()
