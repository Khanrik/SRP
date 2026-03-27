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
        self.LR_dir = Path(split_dir) / LR_subdir
        self.HR_dir = Path(split_dir) / HR_subdir
        self.LR_files = sorted(self.LR_dir.glob("*"))  # png/tif/jpg etc.
        if LR_transform is None:
            lr_ops = []
            if lr_resize_to is not None:
                lr_ops.append(transforms.Resize(lr_resize_to, antialias=True))
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
        return len(self.LR_files)

    def __getitem__(self, idx):
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
    def __init__(self, model, optimizer, criterion, device, learning_rate=3e-4):
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion
        self.device = device
        self.pin_memory = device == "cuda"
        self.num_workers = 0
        self.use_amp = device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.max_pixels_per_image = 1024 * 1024
        self.profile_layers_once = True
    
    def _diff_in_height_coefficient(self, prediction, target):
        difference = torch.mean(torch.abs(prediction - target))
        return difference

    def _tensor_mb(self, tensor):
        return tensor.nelement() * tensor.element_size() / (1024 ** 2)

    def _validate_batch_shapes(self, img, target, prediction):
        img_pixels = img.shape[-2] * img.shape[-1]
        target_pixels = target.shape[-2] * target.shape[-1]
        if img_pixels > self.max_pixels_per_image:
            raise ValueError(
                f"Input image too large: {tuple(img.shape)} ({img_pixels} pixels/image). "
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

    def _log_shape_and_memory(self, stage, epoch, batch_idx, img, target, prediction):
        batch_memory_mb = (
            self._tensor_mb(img) + self._tensor_mb(target) + self._tensor_mb(prediction)
        )
        print(
            f"[{stage}] epoch={epoch + 1} batch={batch_idx + 1} "
            f"img={tuple(img.shape)} target={tuple(target.shape)} pred={tuple(prediction.shape)} "
            f"dtype={img.dtype} img_minmax=({img.min().item():.4f},{img.max().item():.4f}) "
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
        if not self.profile_layers_once:
            return

        hooks = []

        def hook_fn(name):
            def _hook(_module, _inp, out):
                if isinstance(out, torch.Tensor):
                    out_mb = self._tensor_mb(out)
                    print(f"[layer] {name}: shape={tuple(out.shape)} approx={out_mb:.2f}MB")
            return _hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d)):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        self.model.eval()
        with torch.no_grad():
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

        self.test_dataloader = test_dataloader
        return train_dataloader, val_dataloader, test_dataloader

    def train(self, train_dataset, val_dataset, test_dataset, num_epochs, batch_size):
        train_dataloader, val_dataloader, test_dataloader = self._prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size)
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
                torch.cuda.reset_peak_memory_stats()
            self.model.train()
            train_running_loss = 0
            train_running_dc = 0
            
            for idx, img_HR in enumerate(tqdm(train_dataloader, position=0, leave=True)):
                img = img_HR[0].float().to(self.device)
                HR = img_HR[1].float().to(self.device)

                if epoch == 0 and idx == 0:
                    self._profile_layer_activations(img)

                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.use_amp
                    else nullcontext()
                )
                try:
                    with autocast_ctx:
                        y_pred = self.model(img)
                        self._validate_batch_shapes(img, HR, y_pred)
                        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                            hr_min = HR.min().item()
                            hr_max = HR.max().item()
                            if hr_min < 0.0 or hr_max > 1.0:
                                raise ValueError(
                                    "BCEWithLogitsLoss requires target in [0, 1]. "
                                    f"Observed target range: [{hr_min:.4f}, {hr_max:.4f}]. "
                                    "Use a regression loss (e.g., SmoothL1Loss) for height prediction."
                                )
                        loss = self.criterion(y_pred, HR)
                except RuntimeError as err:
                    if "out of memory" in str(err).lower():
                        if self.device == "cuda":
                            torch.cuda.empty_cache()
                        raise RuntimeError(
                            "OOM during forward/loss. Reduce BATCH_SIZE, use smaller resize_to, or simplify model."
                        ) from err
                    raise

                self.optimizer.zero_grad(set_to_none=True)
                dc = self._diff_in_height_coefficient(y_pred.float(), HR)

                train_running_loss += loss.item()
                train_running_dc += dc.item()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if idx == 0:
                    self._log_shape_and_memory("train", epoch, idx, img, HR, y_pred)

            train_loss = train_running_loss / (idx + 1)
            train_dc = train_running_dc / (idx + 1)
            
            train_losses.append(train_loss)
            train_dcs.append(train_dc)

            self.model.eval()
            val_running_loss = 0
            val_running_dc = 0
            
            with torch.no_grad():
                for idx, img_HR in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                    img = img_HR[0].float().to(self.device)
                    HR = img_HR[1].float().to(self.device)

                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=torch.float16)
                        if self.use_amp
                        else nullcontext()
                    )
                    with autocast_ctx:
                        y_pred = self.model(img)
                        self._validate_batch_shapes(img, HR, y_pred)
                        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                            hr_min = HR.min().item()
                            hr_max = HR.max().item()
                            if hr_min < 0.0 or hr_max > 1.0:
                                raise ValueError(
                                    "BCEWithLogitsLoss requires target in [0, 1]. "
                                    f"Observed target range: [{hr_min:.4f}, {hr_max:.4f}]. "
                                    "Use a regression loss (e.g., SmoothL1Loss) for height prediction."
                                )
                        loss = self.criterion(y_pred, HR)
                    dc = self._diff_in_height_coefficient(y_pred, HR)
                    
                    val_running_loss += loss.item()
                    val_running_dc += dc.item()

                    if idx == 0:
                        self._log_shape_and_memory("val", epoch, idx, img, HR, y_pred)

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
    data_root = current_dir.parent / "data"  # contains train/, val/, test/
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
    train_losses, train_dcs, val_losses, val_dcs = trainer.train(train_dataset, val_dataset, test_dataset, num_epochs=10, batch_size=BATCH_SIZE)
    print(f"Training complete. \n Final training loss and dc: {train_losses[-1]:.4f}, {train_dcs[-1]:.4f}")
    print(f"Validation loss and dc: {val_losses[-1]:.4f}, {val_dcs[-1]:.4f}")

if __name__ == "__main__":
    main()

