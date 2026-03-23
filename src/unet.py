import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from unet_helper import UNet



class SegmentationFolderDataset(Dataset):
    def __init__(self, split_dir, LR_subdir="LR", HR_subdir="HR", LR_transform=None, HR_transform=None):
        self.LR_dir = Path(split_dir) / LR_subdir
        self.HR_dir = Path(split_dir) / HR_subdir
        self.LR_files = sorted(self.LR_dir.glob("*"))  # png/tif/jpg etc.
        self.LR_transform = LR_transform or transforms.ToTensor()
        self.HR_transform = HR_transform or transforms.ToTensor()

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
        self.num_workers = torch.cuda.device_count() * 4 if device == "cuda" else 0
    
    def _diff_in_height_coefficient(self, prediction, target):
        prediction_copy = prediction.clone()
        difference = torch.sum(torch.abs(prediction_copy - target))
        return difference

    def _prepare_dataloaders(self, train_dataset, val_dataset, test_dataset, batch_size):
        train_dataloader = DataLoader(
            dataset=train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            shuffle=True,
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
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
            self.model.train()
            train_running_loss = 0
            train_running_dc = 0
            
            for idx, img_HR in enumerate(tqdm(train_dataloader, position=0, leave=True)):
                img = img_HR[0].float().to(self.device)
                HR = img_HR[1].float().to(self.device)

                y_pred = self.model(img)
                self.optimizer.zero_grad()

                dc = self._diff_in_height_coefficient(y_pred, HR)
                loss = self.criterion(y_pred, HR)

                train_running_loss += loss.item()
                train_running_dc += dc.item()

                loss.backward()
                self.optimizer.step()

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

                    y_pred = self.model(img)
                    loss = self.criterion(y_pred, HR)
                    dc = self._diff_in_height_coefficient(y_pred, HR)
                    
                    val_running_loss += loss.item()
                    val_running_dc += dc.item()

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
    train_dataset = SegmentationFolderDataset(data_root / "train")
    val_dataset   = SegmentationFolderDataset(data_root / "val")
    test_dataset  = SegmentationFolderDataset(data_root / "test")

    LEARNING_RATE = 3e-4
    BATCH_SIZE = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_channels=1, num_classes=1).to(device)
    optimizer = optim.AdamW
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(model, optimizer, criterion, device, learning_rate=LEARNING_RATE)
    train_losses, train_dcs, val_losses, val_dcs = trainer.train(train_dataset, val_dataset, test_dataset, num_epochs=10, batch_size=BATCH_SIZE)
    print(f"Training complete. \n Final training loss and dc: {train_losses[-1]:.4f}, {train_dcs[-1]:.4f}")
    print(f"Validation loss and dc: {val_losses[-1]:.4f}, {val_dcs[-1]:.4f}")

if __name__ == "__main__":
    main()

