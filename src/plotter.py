
import time
from helpers import results
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass


# data and metrics
class plotter:
    def plot_val_and_train_loss(self, train_losses, train_maes, train_rmses, train_psnrs, val_losses, val_maes, val_rmses, val_psnrs, save_path=None, display_plots=False):
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
        """
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
        plt.savefig(f'train_val_metrics_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png')
        if display_plots:
            plt.show()

    def _to_plot_array(self,tensor):
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
        difference_tensor = torch.abs(prediction - HR)
        

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
        plt.savefig(f'training_images_{time.strftime("%Y-%m-%d_%H-%M-%S")}.png')
        if display_images:
            plt.show()


    def plot_horizontal_results(self, results_list, save_path=None):
        """Returns: Self.
        Args:
            results: A list of results object containing lists of training and validation metrics per epoch.
        """

        n = len(results_list)

        fig = plt.figure(figsize=(4 * n, 6))
        gs = GridSpec(
            2, n,
            height_ratios=[4, 1.8],
            hspace=0.03,
            wspace=0.05
        )

        # --------------------------------------------------------
        # TOP ROW: IMAGES
        # --------------------------------------------------------
        for i, r in enumerate(results_list):
            ax = fig.add_subplot(gs[0, i])

            img = r.image.detach().cpu()

            # Handle CHW -> HWC
            if img.ndim == 3:
                if img.shape[0] in [1, 3]:
                    img = img.permute(1, 2, 0)

            img = img.numpy()

            # grayscale squeeze
            if img.shape[-1] == 1:
                img = img.squeeze(-1)

            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(r.name, fontsize=12, pad=8)
            ax.axis("off")

        # --------------------------------------------------------
        # TABLE DATA
        # --------------------------------------------------------
        metric_names = ["MSE", "MAE", "RMSE", "PSNR"]

        cell_text = []

        for metric in metric_names:
            row = []
            for r in results_list:
                val = getattr(r, metric)
                row.append(f"{val:.4f}")
            cell_text.append(row)

        col_labels = [r.name for r in results_list]

        # --------------------------------------------------------
        # TABLE
        # --------------------------------------------------------
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.axis("off")

        table = ax_table.table(
            cellText=cell_text,
            rowLabels=metric_names,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            rowLoc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)

        # cleaner borders
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.6)

        # bold column headers
        for col in range(n):
            table[(0, col)].set_text_props(weight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


