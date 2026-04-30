
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path


# data and metrics
class plotter:
    def __init__(self, save_dir="plots",show_plots=False,save_plots=False):
        self.save_dir = Path(save_dir)
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _imshow(self, ax, tensor, *, interpolation="nearest"):
        ax.imshow(tensor, cmap="gray" if tensor.ndim == 2 else None, interpolation=interpolation)

    def plot_val_and_train_loss(self, train_losses, train_maes, train_rmses, train_psnrs, val_losses, val_maes, val_rmses, val_psnrs):
        """Returns: Self.
        Args:
            train_losses: List of training losses per epoch.
            train_maes: List of training MAEs per epoch.
            train_rmses: List of training RMSEs per epoch.
            train_psnrs: List of training PSNRs per epoch.
            val_losses: List of validation losses per epoch.
            val_maes: List of validation MAEs per epoch.
            val_rmses: List of validation RMSEs per epoch.
            val_psnrs: List of validation PSNRs per epoch.
            display_plots: Whether to display the plots interactively.
        """
        if self.save_plots or self.show_plots:
            epochs = range(1, len(train_losses) + 1)

            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(epochs, train_losses, label='Train Loss')
            plt.plot(epochs, val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.plot(epochs, train_maes, label='Train MAE')
            plt.plot(epochs, val_maes, label='Val MAE')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Absolute Error')
            plt.title('Training and Validation MAE')
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(epochs, train_rmses, label='Train RMSE')
            plt.plot(epochs, val_rmses, label='Val RMSE')
            plt.xlabel('Epoch')
            plt.ylabel('Root Mean Squared Error')
            plt.title('Training and Validation RMSE')
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(epochs, train_psnrs, label='Train PSNR')
            plt.plot(epochs, val_psnrs, label='Val PSNR')
            plt.xlabel('Epoch')
            plt.ylabel('Peak Signal-to-Noise Ratio')
            plt.title('Training and Validation PSNR')
            plt.legend()

            plt.tight_layout()

            if self.show_plots:
                plt.show()

            if self.save_plots and self.save_dir:
                plt.savefig(os.path.join(self.save_dir, f'training_validation_metrics_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png'), dpi=300, bbox_inches="tight")

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


    def plot_training_images(self, LR, HR, prediction, train_loss, train_mae, train_rmse, train_psnr, display_images=False):
        """Returns: Self.
        Args:
            LR: Low-resolution input image tensor.
            HR: High-resolution target image tensor.
            prediction: Model's predicted high-resolution image tensor.
            train_loss: Current training loss.
            train_mae: Current training mean absolute error.
            train_rmse: Current training root mean squared error.
            train_psnr: Current training peak signal-to-noise ratio.
        """
        
        if self.save_plots or self.show_plots:
            LR_img = self._to_plot_array(LR)
            HR_img = self._to_plot_array(HR)
            pred_img = self._to_plot_array(prediction)

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title('Low-Resolution Input')
            plt.imshow(LR_img, cmap='gray' if LR_img.ndim == 2 else None)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title('High-Resolution Target')
            plt.imshow(HR_img, cmap='gray' if HR_img.ndim == 2 else None)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title('Model Prediction')
            plt.imshow(pred_img, cmap='gray' if pred_img.ndim == 2 else None)
            plt.axis('off')

            plt.suptitle(f"Train Loss: {sum(train_loss):.4f}, Train MAE: {sum(train_mae):.4f}, Train RMSE: {sum(train_rmse):.4f}, Train PSNR: {sum(train_psnr):.4f}")
            plt.tight_layout()
            if self.save_plots: 
                plt.savefig(os.path.join(self.save_dir, f'training_images_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png'), dpi=300, bbox_inches="tight")
            if self.show_plots:
                plt.show()


    def plot_horizontal_results(self, results_list, interpolation="nearest"):
        """Plot one sample or many samples in a single figure.

        Args:
            results_list: Either a flat list of results for one sample, or a nested
                list where each inner list contains the results for one sample.
            interpolation: Matplotlib interpolation mode used for all images.
        """
        if not (self.save_plots or self.show_plots):
            return

        if not results_list:
            raise ValueError("results_list cannot be empty.")

        is_nested = isinstance(results_list[0], list)
        sample_groups = results_list if is_nested else [results_list]

        num_rows = len(sample_groups)
        num_cols = len(sample_groups[0])

        for group in sample_groups:
            if len(group) != num_cols:
                raise ValueError("All result groups must contain the same number of entries.")

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(4 * num_cols, 4 * num_rows),
            squeeze=False,
        )

        column_titles = [result.name for result in sample_groups[0]]

        for row_idx, group in enumerate(sample_groups):
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
                    ax.set_ylabel(f"Sample {row_idx + 1}", fontsize=11, rotation=0, labelpad=32, va="center")
                ax.axis("off")

                metrics_text = (
                    f"MSE {result.MSE:.4f}\n"
                    f"MAE {result.MAE:.4f}\n"
                    f"RMSE {result.RMSE:.4f}\n"
                    f"PSNR {result.PSNR:.4f}"
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

        fig.tight_layout()

        if self.save_plots and self.save_dir:
            fig.savefig(
                os.path.join(
                    self.save_dir,
                    f"horizontal_results_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )

        if self.show_plots:
            plt.show()


