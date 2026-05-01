
import torch
from generator import Generator
from discriminator import Discriminator
from utils import calculate_error, TV_loss
import torch.nn.functional as F
from tqdm import tqdm

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_distributor import get_base_dataset
from helpers import prepare_dataloader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(1, 128)
generator = generator.to(device)
discriminator = Discriminator(1, 128)
discriminator = discriminator.to(device)

optim_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

num_epochs = 250

current_dir = Path(__file__).resolve().parent
data_root = current_dir.parent.parent / "data"  # Contains train/, val/, test/
checkpoint_dir = current_dir.parent.parent / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
generator_checkpoint_path = checkpoint_dir / "D-SRGAN.pth"
generator_checkpoint_timestamped_path = checkpoint_dir / "archives" / f"D-SRGAN_{time.strftime('%Y-%m-%d_%H-%M-%S')}.pth"

regions = ["jutland", "funen"]
data = get_base_dataset(
    lr_data_dir_list=[data_root / "copernicus" /
                      region for region in regions],
    hr_data_dir_list=[data_root / "dataforsyningen" /
                      region for region in regions],
)
batch_size = 3
pin_memory = True if torch.cuda.is_available() else False

train_loader = prepare_dataloader(data.train, batch_size=batch_size, pin_memory=pin_memory, shuffle_bool=True)
val_loader = prepare_dataloader(data.val, batch_size=batch_size, pin_memory=pin_memory, shuffle_bool=False)

num_train_batches = float(len(train_loader))
num_val_batches = float(len(val_loader))
best_val_psnr = float("-inf")

for epoch in range(num_epochs):
    print(f"Epoch {epoch}: ", end ="")
    
    G_adv_loss = 0
    G_rec_loss = 0
    G_tot_loss = 0
    D_adv_loss = 0
    
    generator.train()
    for batch, (lr, hr) in enumerate(tqdm(train_loader, desc=f"Training for epoch {epoch}")):

      for p in discriminator.parameters():
        p.requires_grad = False
        #training generator
      optim_G.zero_grad()
 
      lr_images = lr.to(device)
      hr_images = hr.to(device)
      lr_images = lr_images.float()
      predicted_hr_images = generator(lr_images)
      predicted_hr_labels = discriminator(predicted_hr_images)
      gf_loss = F.binary_cross_entropy_with_logits(predicted_hr_labels, torch.ones_like(predicted_hr_labels)) #adverserial loss

      # reconstruction loss

      # gr_loss = 100*F.l1_loss(predicted_hr_images, hr_images) # L1 loss
      tv_loss = TV_loss(predicted_hr_images,0.0000005)
      gr_loss = 100*F.mse_loss(predicted_hr_images, hr_images) + tv_loss # L2 loss

      g_loss = gf_loss + gr_loss 

      G_adv_loss += gf_loss.item()
      G_rec_loss += gr_loss.item()
      G_tot_loss += g_loss.item()
      
      g_loss.backward()
      optim_G.step()
      
      # training discriminator
      for p in discriminator.parameters():
        p.requires_grad = True
      optim_D.zero_grad()
      predicted_hr_images = generator(lr_images).detach() # avoid back propogation to generator
      hr_images = hr_images.float()
      adv_hr_real = discriminator(hr_images)
      adv_hr_fake = discriminator(predicted_hr_images)
      df_loss = F.binary_cross_entropy_with_logits(adv_hr_real, torch.ones_like(adv_hr_real)) + F.binary_cross_entropy_with_logits(adv_hr_fake, torch.zeros_like(adv_hr_fake))
      D_adv_loss += df_loss.item()
      df_loss.backward()
      optim_D.step()


    #After each epoch, we perform validation
    with torch.inference_mode():
      val_psnr = 0
      val_ssim = 0
      for batch_idx, (lr, hr) in enumerate(tqdm(val_loader, desc=f"Validating for epoch {epoch}")):
        lr = lr.to(device)
        hr = hr.to(device)
        lr = lr.float()
        predicted_hr = generator(lr)

        psnr, ssim = calculate_error(hr, predicted_hr)
        val_psnr += psnr
        val_ssim += ssim

    val_psnr /= num_val_batches
    val_ssim /= num_val_batches

    if val_psnr > best_val_psnr:
      best_val_psnr = val_psnr
      torch.save(generator.state_dict(), generator_checkpoint_path)
      torch.save(generator.state_dict(), generator_checkpoint_timestamped_path)

    print(f"PSNR: {val_psnr:.3f} SSIM: {val_ssim:.3f}\n")

print(f"Best generator weights saved to {generator_checkpoint_path}")
