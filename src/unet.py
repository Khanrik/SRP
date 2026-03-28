import torch
import torch.nn as nn
from contextlib import nullcontext
from PIL import Image
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from unet_helper import UNet



class SegmentationFolderDataset(Dataset):
    def __init__(
        self,
        split_dir,
        LR_subdir="LR",
        HR_subdir="HR",
        LR_transform=None,
        HR_transform=None,
        lr_resize_to=None,
        hr_resize_to=None,
    ):
        """Returns: Self. Initializes the dataset for paired LR and HR images for super-resolution training.
        Args:
            split_dir: Directory containing the LR and HR subdirectories with the image files.
            LR_subdir: Subdirectory name for low-resolution images.
            HR_subdir: Subdirectory name for high-resolution images.
            LR_transform: (Optional) torchvision transforms to apply to LR images.
            HR_transform: (Optional) torchvision transforms to apply to HR images.
            lr_resize_to: (Optional) (height, width) to resize LR images to. If None, no resizing is done.
            hr_resize_to: (Optional) (height, width) to resize HR images to. If None, no resizing is done.
        """
        self.LR_dir = Path(split_dir) / LR_subdir #Finding data files
        self.HR_dir = Path(split_dir) / HR_subdir
        self.LR_files = sorted(self.LR_dir.glob("*"))  # PNG/TIF/JPG etc.

        #Defining transformations for the data, if not provided
        if LR_transform is None: 
            lr_ops = []
            if lr_resize_to is not None:
                lr_ops.append(transforms.Resize(lr_resize_to, antialias=True)) #Resize (optional)
            lr_ops.append(transforms.ToTensor())
            LR_transform = transforms.Compose(lr_ops)
        if HR_transform is None:
            hr_ops = []
            if hr_resize_to is not None:
                hr_ops.append(transforms.Resize(hr_resize_to, antialias=True))
            hr_ops.append(transforms.ToTensor())
            HR_transform = transforms.Compose(hr_ops)

        self.LR_transform = LR_transform
        self.HR_transform = HR_transform

        if not self.LR_dir.exists():
            raise FileNotFoundError(f"LR directory does not exist: {self.LR_dir}")
        if not self.HR_dir.exists():
            raise FileNotFoundError(f"HR directory does not exist: {self.HR_dir}")

    def __len__(self):
        # Len override. size of LR dataset
        return len(self.LR_files)

    def __getitem__(self, idx):
        # Getitem override. for a given LR image index, find the corresponding HR image, apply transformations,
        # and return both as tensors
        LR_path = self.LR_files[idx]
        HR_name = LR_path.name.replace("copernicus_", "dataforsyningen_", 1)
        HR_path = self.HR_dir / HR_name

        if not HR_path.exists():
            raise FileNotFoundError(
                f"Missing HR file for LR '{LR_path.name}'. Expected: '{HR_name}' in '{self.HR_dir}'."
            )

        LR = Image.open(LR_path)
        HR = Image.open(HR_path)
        LR = self.LR_transform(LR).float()
        HR = self.HR_transform(HR).float()
        return LR, HR


class Trainer:
    def __init__(self, model, optimizer, criterion, device, learning_rate=3e-4, max_pixels_per_image=1024*1024, profile_layers_once=True):
        """ Returns: Self. Initializes the Trainer with the model, optimizer, loss function, device, and learning rate.
        Args:
            model: The neural network model.
            optimizer: The chosen optimizer for training the model.
            criterion: The chosen loss function.
            device: The device to run the model on, eg. "cuda" or "cpu".
            learning_rate: The learning rate for the optimizer. (Default 3e-4).
            max_pixels_per_image: Max num of pixels allowed in input/target images to avoid OOM errors. (Default 1024x1024).
            profile_layers_once: Whether to profile layers in start of training. (Default True).
        """
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion
        self.device = device
        self.pin_memory = device == "cuda"
        self.num_workers = 0
        self.use_amp = device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_pixels_per_image = max_pixels_per_image
        self.profile_layers_once = profile_layers_once

    def _diff_in_height_coefficient(self, prediction, target):
        # For height prediction, we can use mean absolute error as a simple "difference in height coefficient".
        difference = torch.mean(torch.abs(prediction - target))
        return difference

    def _tensor_mb(self, tensor):
        # Calculate the approximate memory usage of a tensor in megabytes. 
        # This is for avoiding OOM errors and does not account for additional memory used (autograd, optimizer states, etc).
        return tensor.nelement() * tensor.element_size() / (1024 ** 2)

    def _validate_batch_shapes(self, LR, target, prediction):
        # Ensuring that the input image, target, and prediction have reasonable shapes and sizes 
        # to avoid OOM errors and shape mismatches during training.
        LR_pixels = LR.shape[-2] * LR.shape[-1]
        target_pixels = target.shape[-2] * target.shape[-1]
        if LR_pixels > self.max_pixels_per_image:
            raise ValueError(
                f"Input image too large: {tuple(LR.shape)} ({LR_pixels} pixels/image). "
                f"Lower resolution or increase max_pixels_per_image."
            )
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: pred={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}."
            )
        if target_pixels > self.max_pixels_per_image * 9:
            raise ValueError(
                f"Target image is very large: {tuple(target.shape)} ({target_pixels} pixels/image). "
                f"Use smaller lr_resize_to/hr_resize_to values."
            )

    def _log_shape_and_memory(self, stage, epoch, batch_idx, LR, target, prediction):
        # Log the shapes, data types, value ranges of the input, target, and prediction tensors,
        # as well as the approximate memory usage of the batch tensors and CUDA memory stats if using GPU. 
        # This is for debugging and analysis of training dynamics and OOM issues.
        batch_memory_mb = (
            self._tensor_mb(LR) + self._tensor_mb(target) + self._tensor_mb(prediction)
        )
        print(
            f"[{stage}] epoch={epoch + 1} batch={batch_idx + 1} "
            f"LR={tuple(LR.shape)} target={tuple(target.shape)} pred={tuple(prediction.shape)} "
            f"dtype={LR.dtype} LR_minmax=({LR.min().item():.4f},{LR.max().item():.4f}) "
            f"target_minmax=({target.min().item():.4f},{target.max().item():.4f}) "
            f"pred_minmax=({prediction.min().item():.4f},{prediction.max().item():.4f})"
        )
        if self.device == "cuda":
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"[{stage}] memory: batch_tensors={batch_memory_mb:.2f}MB "
                f"cuda_allocated={allocated_mb:.2f}MB cuda_reserved={reserved_mb:.2f}MB "
                f"cuda_peak_allocated={peak_allocated_mb:.2f}MB"
            )
        else:
            print(f"[{stage}] memory: batch_tensors={batch_memory_mb:.2f}MB (cpu)")

    def _profile_layer_activations(self, sample_batch):
        # This function registers forward hooks on convolutional and pooling layers
        # to log the shape and approximate memory usage of their outputs during a forward pass with a sample batch.

        if not self.profile_layers_once:
            return

        hooks = [] # Note to self: this is a simple way to profile layer activations for debugging and analysis.

        def hook_fn(name):
            def _hook(_module, _inp, out):
                if isinstance(out, torch.Tensor): # Only log if output is a tensor (some layers may output tuples, dicts, etc.)
                    out_mb = self._tensor_mb(out)
                    print(f"[layer] {name}: shape={tuple(out.shape)} approx={out_mb:.2f}MB")
            return _hook 
        
        # Going through all layers and registering hooks on conv and pooling layers 
        # to log their output shapes and memory usage during the first forward pass
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d)): # We can add more layer types if needed
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Run a forward pass with the sample batch to trigger the hooks and log layer outputs
        self.model.eval() 
        with torch.no_grad(): 
            # Enables autocasting for the forward pass if using mixed precision (use_amp=true), 
            # which can reduce memory usage and is important to profile accurately.
            autocast_ctx = ( 
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.use_amp
                else nullcontext()
            )
            with autocast_ctx:
                _ = self.model(sample_batch)

        for hook in hooks:
            hook.remove()
        self.profile_layers_once = False

    def _prepare_dataloaders(self, train_dataset, val_dataset, test_dataset, batch_size):
        # This function initializes the iterable over the datasets for training, validation, and testing 
        # with the specified batch size and the Trainer's num_workers and pin_memory settings.
        train_dataloader = DataLoader(
            dataset=train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            persistent_workers=False,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            persistent_workers=False,
            shuffle=True,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            persistent_workers=False,
            shuffle=True,
        )

        self.test_dataloader = test_dataloader # Storing test dataloader for later use in testing after training is complete
        return train_dataloader, val_dataloader, test_dataloader

    def train(self, train_dataset, val_dataset, test_dataset, num_epochs, batch_size):
        """Returns: training and validation losses and difference in height coefficients per epoch for analysis and debugging.
        Args:
            train_dataset: Dataset for training.
            val_dataset: Dataset for validation.
            test_dataset: Dataset for testing.
            num_epochs: Number of epochs to train.
            batch_size: Batch size for training and validation.
        
        """
        train_dataloader, val_dataloader, test_dataloader = \
            self._prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)
        
        if train_dataloader is None or val_dataloader is None or test_dataloader is None:
            raise ValueError(
                f"Dataloaders not properly initialized. Train samples: {len(train_dataset)}, "
                f"Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
            )

        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
            raise ValueError(
                f"Empty dataset split detected. Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
            )

        num_train_batches = len(train_dataloader)
        num_val_batches = len(val_dataloader)
        if num_train_batches == 0 or num_val_batches == 0:
            raise ValueError(
                f"Empty dataloader detected. Train batches: {num_train_batches}, "
                f"Val batches: {num_val_batches}"
            )


        train_losses = []
        train_dcs = []
        val_losses = []
        val_dcs = []

        for epoch in tqdm(range(num_epochs)):
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats() #Keeping track of peak memory usage per epoch for debugging OOM errors

            self.model.train()
            train_running_loss = 0
            train_running_dc = 0
            
            for idx, LR_HR in enumerate(tqdm(train_dataloader, position=0, leave=True)):
                # creating LR and HR tensors for the batch and moving them to the correct device.
                LR = LR_HR[0].float().to(self.device) 
                HR = LR_HR[1].float().to(self.device)

                if epoch == 0 and idx == 0:
                    # Profiling for memory usage stats to avoid OOM errors.
                    self._profile_layer_activations(LR) 

                # Using autocasting for the forward pass and loss calculation to reduce memory usage.
                # If use_amp is false, this will just be a nullcontext and have no effect.
                autocast_ctx = (
                    torch.autocast(device_type=self.device, dtype=torch.float16)
                    if self.use_amp
                    else nullcontext()
                )

                try:
                    with autocast_ctx:
                        # Forward pass through the model to get predictions, and calculating the loss with the criterion.
                        y_pred = self.model(LR)
                        self._validate_batch_shapes(LR, HR, y_pred)
                        loss = self.criterion(y_pred, HR)
                except RuntimeError as err:
                    if "out of memory" in str(err).lower():
                        # catching OOM errors during the forward pass or loss calculation.
                        if self.device == "cuda":
                            torch.cuda.empty_cache() 
                        raise RuntimeError(
                            "OOM during forward/loss. Reduce BATCH_SIZE, use smaller resize_to, or simplify model."
                        ) from err
                    raise
                
                # Using set_to_none=True to reduce memory usage by freeing gradients immediately after backward pass.
                self.optimizer.zero_grad(set_to_none=True) 
                dc = self._diff_in_height_coefficient(y_pred.float(), HR)

                train_running_loss += loss.item()
                train_running_dc += dc.item()

                # Scales the loss for mixed precision (autocast) training to prevent underflow in gradients, 
                # and performs the backward pass.
                self.scaler.scale(loss).backward() 
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if idx == 0:
                    self._log_shape_and_memory("train", epoch, idx, LR, HR, y_pred)

            train_loss = train_running_loss / (idx + 1)
            train_dc = train_running_dc / (idx + 1)
            
            train_losses.append(train_loss)
            train_dcs.append(train_dc)

            # Validation loop, i.e. training loop but without backpropagation and with torch.no_grad() to save memory and computations.
            self.model.eval()
            val_running_loss = 0
            val_running_dc = 0
            
            with torch.no_grad():
                for idx, LR_HR in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                    LR = LR_HR[0].float().to(self.device)
                    HR = LR_HR[1].float().to(self.device)

                    autocast_ctx = (
                        torch.autocast(device_type=self.device, dtype=torch.float16)
                        if self.use_amp
                        else nullcontext()
                    )
                    with autocast_ctx:
                        y_pred = self.model(LR)
                        self._validate_batch_shapes(LR, HR, y_pred)
                        loss = self.criterion(y_pred, HR)
                    dc = self._diff_in_height_coefficient(y_pred, HR)
                    
                    val_running_loss += loss.item()
                    val_running_dc += dc.item()

                    if idx == 0:
                        self._log_shape_and_memory("val", epoch, idx, LR, HR, y_pred)

                val_loss = val_running_loss / (idx + 1)
                val_dc = val_running_dc / (idx + 1)
            
            val_losses.append(val_loss)
            val_dcs.append(val_dc)

            print("-" * 30)
            print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
            print(f"Training Difference EPOCH {epoch + 1}: {train_dc:.4f}")
            print("\n")
            print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
            print(f"Validation Difference EPOCH {epoch + 1}: {val_dc:.4f}")
            print("-" * 30)

        # Saving the model
        torch.save(self.model.state_dict(), 'my_checkpoint.pth')
        return train_losses, train_dcs, val_losses, val_dcs


def main():
    current_dir = Path(__file__).parent
    data_root = current_dir.parent / "data"  # Contains train/, val/, test/
    basesize = 128*2
    #LR_RESIZE_TO = (basesize, basesize)
    #HR_RESIZE_TO = (basesize*3, basesize*3)
    LR_RESIZE_TO = None
    HR_RESIZE_TO = None
    train_dataset = SegmentationFolderDataset(
        data_root / "train", lr_resize_to=LR_RESIZE_TO, hr_resize_to=HR_RESIZE_TO
    )
    val_dataset   = SegmentationFolderDataset(
        data_root / "val", lr_resize_to=LR_RESIZE_TO, hr_resize_to=HR_RESIZE_TO
    )
    test_dataset  = SegmentationFolderDataset(
        data_root / "test", lr_resize_to=LR_RESIZE_TO, hr_resize_to=HR_RESIZE_TO
    )

    LEARNING_RATE = 3e-4
    BATCH_SIZE = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.AdamW
    criterion = nn.SmoothL1Loss()

    trainer = Trainer(model, optimizer, criterion, device, learning_rate=LEARNING_RATE)
    train_losses, train_dcs, val_losses, val_dcs = \
        trainer.train(train_dataset, val_dataset, test_dataset, num_epochs=10, batch_size=BATCH_SIZE)
    print(f"Training complete. \n Final training loss and dc: {train_losses[-1]:.4f}, {train_dcs[-1]:.4f}")
    print(f"Validation loss and dc: {val_losses[-1]:.4f}, {val_dcs[-1]:.4f}")

if __name__ == "__main__":
    main()

