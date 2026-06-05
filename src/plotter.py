
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
from shapely.geometry import box
import geopandas as gpd
from helpers import normalize_targets, denormalize_target, metric_items
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
            
            plt.close('all')

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

    def plot_horizontal_results(self, results_list: list, interpolation: str="nearest", max_rows_per_figure: int=3):
        """Plot one sample or many samples in a single figure for visual display.

        Args:
            results_list (list): Either a flat list of results for one sample, or a nested list where each inner list contains the results for one sample.
            interpolation (str): Matplotlib interpolation mode used for all images.
            max_rows_per_figure (int): Maximum number of sample rows to render per figure.
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

        pages = [sample_groups[start:start + max_rows_per_figure] for start in range(0, len(sample_groups), max_rows_per_figure)]

        for page_idx, page_groups in enumerate(pages, start=1):
            # Compute columns for this page only
            num_cols_page = max(len(group) for group in page_groups)
            # Build titles for this page
            column_titles_page = []
            for col_idx in range(num_cols_page):
                title = None
                for group in page_groups:
                    if col_idx < len(group):
                        title = group[col_idx].name
                        break
                column_titles_page.append(title or "")

            num_rows = len(page_groups)
            fig, axes = plt.subplots(
                num_rows,
                num_cols_page,
                figsize=(4 * num_cols_page, 3.6 * num_rows),
                squeeze=False,
            )

            for row_idx, group in enumerate(page_groups):
                for col_idx in range(num_cols_page):
                    ax = axes[row_idx][col_idx]
                    # If this group doesn't have this column (e.g. GT missing), leave axis blank.
                    if col_idx >= len(group):
                        ax.axis("off")
                        if row_idx == 0:
                            ax.set_title(column_titles_page[col_idx], fontsize=11, pad=10)
                        continue

                    result = group[col_idx]
                    img = result.image.detach().cpu()
                    if img.ndim == 3:
                        if img.shape[0] in [1, 3]:
                            img = img.permute(1, 2, 0)

                    img = img.numpy()
                    if img.ndim == 3 and img.shape[-1] == 1:
                        img = img.squeeze(-1)

                    self._imshow(ax, img, interpolation=interpolation)
                    if row_idx == 0:
                        ax.set_title(column_titles_page[col_idx], fontsize=11, pad=10)
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
            fig.tight_layout()
            self._save_figure(fig, f"horizontal_results_page_{page_idx}")

        if self.show_plots:
            plt.show()
        plt.close('all')

    def get_dataframe(self, loaders: list[DataLoader], model_pipeline_list: list, metrics: dict, min_val: float, max_val: float, mean_val: float, std_val: float, crs: str="EPSG:25832", include_bilinear: bool=True):
        """Generates a GeoDataFrame containing the evaluation metrics for each sample in the dataset, along with their corresponding geometries.
        Args:
            loaders (list[DataLoader]): A list of DataLoaders containing the datasets to be evaluated.
            model_pipeline_list (list): A list of model pipelines to be evaluated.
            metrics (dict): A dictionary of metrics to be calculated.
            min_val (float): The minimum value for normalization.
            max_val (float): The maximum value for normalization.
            mean_val (float): The mean pixel value for normalization.
            std_val (float): The standard deviation of pixel values for normalization.
            crs (str): The coordinate reference system for the GeoDataFrame.
            include_bilinear (bool): Whether to include a bilinear interpolation baseline in the resulting metrics table.
        """
        rows = []
        for loader in loaders:
            dataset = loader.dataset
            bbox_cache = {}
            for pipeline in model_pipeline_list:
                pipeline.model.eval()

            for LR, HR, dataset_indices in tqdm(loader, desc="Evaluating pipelines and baselines", leave=False, total=len(loader)):
                LR_cpu = LR.float()
                HR_cpu = HR.float()
                batch_size = HR_cpu.shape[0]

                shared_rows = []
                for batch, dataset_idx in enumerate(dataset_indices):
                    if dataset_idx not in bbox_cache:
                        bbox_cache[dataset_idx] = dataset.get_bbox(dataset_idx)
                    bbox = bbox_cache[dataset_idx]
                    shared_rows.append(
                        {
                            "category": dataset.category,
                            "dataset_idx": dataset_idx,
                            "geometry": box(
                                bbox.left,
                                bbox.bottom,
                                bbox.right,
                                bbox.top,
                            ),
                            "lr_min": LR_cpu[batch].min().item(),
                            "lr_max": LR_cpu[batch].max().item(),
                            "hr_min": HR_cpu[batch].min().item(),
                            "hr_max": HR_cpu[batch].max().item(),
                        }
                    )

                if include_bilinear:
                    bilinear = torch.nn.functional.interpolate(
                        LR_cpu,
                        size=HR_cpu.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for batch in range(batch_size):
                        row = shared_rows[batch].copy()
                        row.update(
                            {
                                "name": "bilinear",
                                "model": "baseline",
                                "criterion": "interpolation",
                                "optimizer": "N/A",
                                "downsampled HR input": False,
                            }
                        )

                        bilinear_sample = bilinear[batch:batch+1]
                        hr_sample = HR_cpu[batch:batch+1]
                        results = metric_items(bilinear_sample, hr_sample, metrics, min_val, max_val)
                        for name, value in results:
                            row[name] = float(value)

                        rows.append(row)

                for pipeline in model_pipeline_list:
                    with torch.no_grad():
                        LR_device = LR_cpu.to(pipeline.device)
                        HR_device = HR_cpu.to(pipeline.device)
                        normalized_LR = normalize_targets(targets=LR_device, mean=mean_val, std=std_val)
                        pred = pipeline.model(normalized_LR)
                        pred = denormalize_target(pred, mean=mean_val, std=std_val)

                        for batch in range(batch_size):
                            row = shared_rows[batch].copy()
                            row.update(
                                {
                                    "name": pipeline.pth_path_name,
                                    "model": pipeline.model.__class__.__name__,
                                    "criterion": pipeline.criterion.__class__.__name__,
                                    "optimizer": pipeline.optimizer.__class__.__name__,
                                    "downsampled HR input": pipeline.downsampled_data,
                                }
                            )

                            pred_sample = pred[batch:batch+1]
                            hr_sample = HR_device[batch:batch+1]
                            results = metric_items(pred_sample, hr_sample, metrics, min_val, max_val)
                            for name, value in results:
                                row[name] = float(value)

                            rows.append(row)
        self.gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    def plot_datasplit_map(self):
        """Plots a map showing the distribution of samples across different categories and datasets.
        Each sample is represented as a polygon based on its bounding box, colored according to its category (e.g., train, val, test).
        The map provides a visual overview of how the data is split and distributed geographically.
        """
        if not (self.save_plots or self.show_plots):
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_frame = self.gdf.drop_duplicates(subset=["category", "dataset_idx"]).copy()
        plot_frame.plot(
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
        plt.close('all')

    def plot_extrema_map(self):
        """
        Plots maps showing the distribution of minimum and maximum values for each sample across different categories and datasets.
        Each sample is represented as a polygon based on its bounding box, colored according to its category (e.g., train, val, test).
        The maps provide a visual overview of how the data is split and distributed geographically.
        """
        if not (self.save_plots or self.show_plots):
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), squeeze=False)
        plot_frame = self.gdf.drop_duplicates(subset=["category", "dataset_idx"]).copy() # ensure one record per sample (not one per pipeline)
        plot_specs = [
            ("lr_min", "LR Chunk Minimum Map", "viridis", axes[0][0]),
            ("lr_max", "LR Chunk Maximum Map", "plasma", axes[0][1]),
            ("hr_min", "HR Chunk Minimum Map", "viridis", axes[1][0]),
            ("hr_max", "HR Chunk Maximum Map", "plasma", axes[1][1]),
        ]
        for key, title, cmap, ax in plot_specs:
            plot_frame.plot(
                column=key,
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
        plt.close('all')


    def plot_boxplots(self, metric_name: str = "SSIM"):
        """Plots boxplots comparing the distribution of a specified metric across different model pipelines.
        Args:
            metric_name (str, optional): The name of the metric to be compared across pipelines. Defaults to "SSIM".
        """
        if not (self.save_plots or self.show_plots):
            return
        
        pipeline_labels = self.gdf["name"].unique().tolist()
        pipeline_scores = []
        for pipeline_name in pipeline_labels:
            # Find best model excluding bilinear baseline
            if pipeline_name != "bilinear":
                pipeline_scores.append(self.gdf[self.gdf["name"] == pipeline_name][metric_name].tolist())

        def _average_score(values: list[float]) -> float:
            finite_values = [value for value in values if np.isfinite(value)]
            return np.mean(finite_values)
        
        average_scores = [_average_score(scores) for scores in pipeline_scores]
        best_pipeline = pipeline_labels[int(np.argmax(average_scores))]

        # re-introduce bilinear for boxplots
        pipeline_scores.append(self.gdf[self.gdf["name"] == "bilinear"][metric_name].tolist())

        fig, ax = plt.subplots(figsize=(max(10, 3.6 * len(pipeline_labels)), 6))
        bp = ax.boxplot(pipeline_scores, labels=pipeline_labels, vert=True, patch_artist=True)
        ax.set_title(f"{metric_name} Distribution by Pipeline. \n Evaluated best: {best_pipeline}", fontsize=14, pad=12)
        ax.set_ylabel(metric_name)
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

        self._save_figure(fig, f"{metric_name.lower()}_boxplots")
        if self.show_plots:
            plt.show()
        plt.close('all')
        return best_pipeline

    def plot_metric_maps(self, pipeline_name: str, metrics: dict):
        """Plots maps showing the distribution of specified metrics for a given model pipeline, along with boxplots comparing the metric distributions across pipelines.
        Args:
            pipeline_name (str): The name of the model pipeline for which to plot the metric maps.
            metrics (dict): A dictionary of metric names and their corresponding values to be plotted on the maps.
        """
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
        fig.suptitle(f"Metric Maps and Boxplots for Pipeline: {pipeline_name}", fontsize=14, y=0.995)

        pipeline_gdf = self.gdf[self.gdf["name"] == pipeline_name]

        for col_idx, metric_name in enumerate(metrics.keys()):
            map_ax = axes[0][col_idx]
            gdf = gpd.GeoDataFrame(
                pipeline_gdf[["geometry", metric_name]].copy(),
                geometry="geometry",
                crs=self.gdf.crs
            )
            gdf = gdf[np.isfinite(gdf[metric_name])]
            gdf.plot(
                column=metric_name,
                cmap="viridis",
                legend=True,
                edgecolor="black",
                linewidth=0.2,
                ax=map_ax
            )
            map_ax.set_title(f"{metric_name}", fontsize=12, pad=8)
            map_ax.set_axis_off()

            boxplot_ax = axes[1][col_idx]
            boxplot_ax.boxplot(gdf[metric_name].tolist(), vert=True, patch_artist=True)
            boxplot_ax.set_xticks([])
            boxplot_ax.grid(axis="y", alpha=0.3)
            
        self._save_figure(fig, "metric_maps")
        if self.show_plots:
            plt.show()
        plt.close('all')

    def _create_typst_table(self, summary, headers, group_display_names, caption):
        typst_lines = ["#onecol("]
        typst_lines.append("  figure(")
        typst_lines.append("    table(")
        typst_lines.append(f"      columns: {len(headers)},")
        typst_lines.append("")
        typst_lines.append("      " + ", ".join(f"[{group_display_names.get(header, header)}]" for header in headers) + ",")
        typst_lines.append("")

        summary_records = summary.to_dict(orient="records")
        row_count = len(summary_records)
        for row_idx, row in enumerate(summary_records):
            cells = []
            for col in headers:
                value = row[col]
                if isinstance(value, float):
                    cells.append(f"[{value:.4f}]")
                    continue

                if row_idx > 0 and summary_records[row_idx - 1][col] == value:
                    continue

                span = 1
                next_idx = row_idx + 1
                while next_idx < row_count and summary_records[next_idx][col] == value:
                    span += 1
                    next_idx += 1

                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                cells.append(f"table.cell(rowspan: {span})[{value}]")
            typst_lines.append("      " + ", ".join(cells) + ",")

        typst_lines.append("    ),")
        typst_lines.append(f"    caption: [{caption}]")
        typst_lines.append("  )")
        typst_lines.append(")")
        return typst_lines

    def log_typst_table(self, logger, metrics: dict):
        """Logs a typst-formatted table containing the average metric values for each model pipeline configuration, grouped by model architecture, loss function, and optimizer.
        Args:
            logger (logging.Logger): A logging.Logger instance for logging messages.
            metrics (dict): A dictionary of metric names and their corresponding values to be included in the table.
        """
        group_cols = ["model", "criterion", "optimizer"]
        group_display_names = {
            "model": "Architecture",
            "criterion": "@LF",
            "optimizer": "Optimizer",
        }
        metric_cols = [name for name in metrics.keys()]
        headers = group_cols + metric_cols

        typst_lines = []
        for downsampled_value, caption in [(False, "All model configurations trained on Copernicus @LR input and evaluated on the entirety of Denmark"),
                                           (True, "All model configurations trained on downsampled @HR input and evaluated on the entirety of Denmark")]:
            summary = (
                self.gdf[self.gdf["downsampled HR input"] == downsampled_value][group_cols + metric_cols]
                .groupby(group_cols, dropna=False)
                .mean()
                .reset_index()
            )
            typst_lines.extend(
                self._create_typst_table(summary, headers, group_display_names, caption)
            )

        logger.info("\n\n" + "\n".join(typst_lines) + "\n\n")