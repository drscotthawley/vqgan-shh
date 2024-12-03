#!/bin/env python3 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
torch.set_float32_matmul_precision('high')
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import vgg16
from torchvision.utils import make_grid
import torch.nn.functional as F
import wandb
from tqdm.auto import tqdm
import random
import argparse
import matplotlib.pyplot as plt
import io
import tempfile
import numpy as np

from vqgan_shh.lucid_vqvae import ImprovedLucidVQVAE
from vqgan_shh.adversarial import AdversarialLoss

class FlowerPairDataset(Dataset):
    "li'l thing that grabs two flowers at a time"
    def __init__(self, base_dataset):
        self.dataset, self.indices = base_dataset, list(range(len(base_dataset)))
        
    def __len__(self): 
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # Get source image and class
        source_img, source_class = self.dataset[idx]
        target_idx = idx # random.choice(self.indices)
        target_img, target_class = self.dataset[target_idx]
        
        return source_img, source_class, target_img, target_class


def create_loaders(batch_size=32, image_size=128, shuffle_val=True):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_base = datasets.Flowers102( root='./data', split='train', transform=train_transforms, download=True)
    val_base = datasets.Flowers102(root='./data', split='val', transform=val_transforms, download=True)
    train_dataset = FlowerPairDataset(train_base)
    val_dataset = FlowerPairDataset(val_base)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=8)
    
    return train_loader, val_loader

def perceptual_loss(vgg, img1, img2):
    # Normalize images to VGG range
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(img1.device)
    img1 = (img1 - mean) / std
    img2 = (img2 - mean) / std
    
    # Get features from multiple layers
    features1 = vgg(img1)
    features2 = vgg(img2)
    
    # Compute loss at each layer
    loss = 0
    for f1, f2 in zip(features1, features2):
        loss += F.mse_loss(f1, f2)
        
    return loss


def pwrspec(y, eps=1e-8): 
    return torch.log(eps + torch.abs(torch.fft.fft2(y)))

def spectral_loss(x, x_recon):
    return F.mse_loss(pwrspec(x), pwrspec(x_recon))


def fw_spectral_loss(x, x_recon, beta=2.0, eps=1e-8):
    "spectral loss but we weight by frequency"
    freqs = torch.sqrt(torch.fft.fftfreq(x.shape[-2], device=x.device)[:, None]**2 + 
                      torch.fft.fftfreq(x.shape[-1], device=x.device)[None, :]**2)
    weights = (freqs ** beta).unsqueeze(0).unsqueeze(0)
    return torch.mean(weights * (torch.log(eps + torch.abs(torch.fft.fft2(x_recon))) - 
                                torch.log(eps + torch.abs(torch.fft.fft2(x))))**2)


def compute_losses(recon, target_imgs, vq_loss, vgg, adv_loss=None, epoch=None, config=None):
    """Compute all losses in a single place. Returns dict of loss tensors."""

    losses = {
        'mse': F.mse_loss(recon, target_imgs),
        'vq': vq_loss,
        'perceptual': perceptual_loss(vgg, recon, target_imgs), 
        'spectral': spectral_loss(recon, target_imgs)
    }
    
    # Only add adversarial losses after warmup
    if adv_loss is not None and epoch > config.warmup_epochs:
        d_loss, real_features = adv_loss.discriminator_loss(target_imgs, recon)
        g_loss = adv_loss.generator_loss(recon, real_features)
        losses['d_loss'] = d_loss
        losses['g_loss'] = config.lambda_adv * g_loss
        
    return losses

def get_total_loss(losses, config=None):
    """Compute weighted sum of losses."""
    total = config.lambda_mse*losses['mse'] + config.lambda_vq*losses['vq'] + \
        config.lambda_perc * losses['perceptual'] + config.lambda_spec * losses['spectral']

    if 'g_loss' in losses:
        total = total + losses['g_loss']
    return total


# two diagnostics for the discriminator
def get_discriminator_stats(adv_loss, real_images, fake_images):
    with torch.no_grad():
        d_real = adv_loss.discriminator(real_images).mean()
        d_fake = adv_loss.discriminator(fake_images).mean()
        return {
            'd_real_mean': d_real.item(),
            'd_fake_mean': d_fake.item(),
            'd_conf_gap': (d_real - d_fake).item()
        }
def get_gradient_stats(discriminator):
    total_norm = 0.0
    for p in discriminator.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return {'d_grad_norm': math.sqrt(total_norm)}



def get_discriminator_stats(adv_loss, real_images, fake_images):
    with torch.no_grad():
        d_real = adv_loss.discriminator(real_images)[0].mean()  # Add [0] to get first element of tuple
        d_fake = adv_loss.discriminator(fake_images)[0].mean()  # Add [0] to get first element of tuple
        return {
            'd_real_mean': d_real.item(),
            'd_fake_mean': d_fake.item(),
            'd_conf_gap': (d_real - d_fake).item()
        }

def get_gradient_stats(discriminator):
    total_norm = 0.0
    for p in discriminator.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return {'d_grad_norm': math.sqrt(total_norm)}



def process_batch(model, vgg, batch, device, is_train=True, optimizer=None, d_optimizer=None, 
                 adv_loss=None, epoch=None, config=None, batch_idx=0):
    """Process a single batch and return losses."""
    source_imgs = batch[0].to(device)
    target_imgs = source_imgs

    # Pre-warmup behavior - sectioned this off b/c one time I broke everything when I added the GAN part. 
    if epoch <= config.warmup_epochs:
        recon, vq_loss = model(source_imgs)
        losses = compute_losses(recon, target_imgs, vq_loss, vgg, adv_loss=None, epoch=None, config=config)
        losses['total'] = get_total_loss(losses, config)
        
        if is_train:
            optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        return {k: v.item() for k, v in losses.items()}, recon

    # Post-warmup behavior
    else:
        # Train discriminator multiple times
        d_losses = []
        d_stats_list = []
        grad_stats_list = []
        
        recon, vq_loss = model(source_imgs)  # Get fresh recons
        recon = recon.detach()  # Detach since we only need for D training
        if is_train and adv_loss is not None:
            for _ in range(3):  # Train D multiple times per batch
                # Reuse the same reconstruction for D training
                d_loss, real_features = adv_loss.discriminator_loss(target_imgs, recon)
                d_stats = get_discriminator_stats(adv_loss, target_imgs, recon)
                d_stats_list.append(d_stats)  

                d_optimizer.zero_grad()
                d_loss.backward()
                grad_stats = get_gradient_stats(adv_loss.discriminator)
                grad_stats_list.append(grad_stats)  
                d_losses.append(d_loss.item())  
                d_optimizer.step()

        # Train generator (less frequently)
        g_losses = {}
        if not is_train or batch_idx % 2 == 0:  # Update G every 3rd batch
            recon, vq_loss = model(source_imgs)
            losses = compute_losses(recon, target_imgs, vq_loss, vgg, adv_loss, epoch, config=config)
            losses['total'] = get_total_loss(losses, config)
            
            if is_train:
                optimizer.zero_grad()
                losses['total'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            g_losses = {k: v.item() for k, v in losses.items()}

        # Combine all losses and stats for logging
        combined_losses = g_losses
        if d_losses:
            combined_losses['d_loss'] = sum(d_losses) / len(d_losses)
            # Add averaged discriminator stats
            avg_d_stats = {k: sum(d[k] for d in d_stats_list) / len(d_stats_list) 
                          for k in d_stats_list[0]}
            avg_grad_stats = {k: sum(d[k] for d in grad_stats_list) / len(grad_stats_list) 
                            for k in grad_stats_list[0]}
            combined_losses.update(avg_d_stats)
            combined_losses.update(avg_grad_stats)

        return combined_losses, recon



def train_epoch(model, vgg, loader, optimizer, d_optimizer, device, epoch, adv_loss=None, config=None):
    model.train()
    # Start with basic losses
    epoch_losses = {k: 0 for k in ['mse', 'vq', 'perceptual', 'spectral', 'total']}
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        batch_losses, recon = process_batch(model, vgg, batch, device, True, optimizer, d_optimizer=d_optimizer, adv_loss=adv_loss, epoch=epoch, config=config, batch_idx=batch_idx )
        
        # Only update losses that exist in batch_losses
        epoch_losses.update({k: epoch_losses[k] + batch_losses[k] for k in batch_losses.keys() & epoch_losses.keys()})
        
        # Add new losses from batch if they appear (after warmup)
        for k in batch_losses:
            if k not in epoch_losses:
                epoch_losses[k] = batch_losses[k]
        
        pbar.set_postfix({k: f'{v:.4g}' for k, v in batch_losses.items()})
        
        if batch_idx % 100 == 0 and not config.no_wandb:
            wandb.log({f'batch/{k}_loss': v for k, v in batch_losses.items()} | {'epoch':epoch})


    return {k: v / len(loader) for k, v in epoch_losses.items()}


def glamp(x, min_val=0, max_val=1, g=1.0): # g >=1 means normal clamp
    # soft clamp, see https://sigmoid.social/@drscotthawley/110545005227110916
    # ONLY using this for viz of recon images, nowhere else
    if g >= 1.0: 
        return torch.clamp(x, min_val, max_val)
    glamped = (1 - g) * torch.tanh(x) + g * torch.clamp(x, -1, 1)
    rescaled = (glamped + 1) / 2 * (max_val - min_val) + min_val  # Rescale to [min_val, max_val]
    return rescaled



def viz_codebook(model, config, epoch):
    if config.no_wandb: return
    # Extract VQ codebook vectors
    codebook_vectors = model.vq.codebook.detach().cpu().numpy()
    
    # Reshape the codebook vectors to the desired shape
    codebook_image = codebook_vectors.reshape(config.vq_num_embeddings, config.vq_embedding_dim)
    
    # Create an image of the codebook vectors using matplotlib
    plt.figure(figsize=(16, 4))
    plt.imshow(codebook_image.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('VQ Codebook Vectors')
    plt.ylabel('Embedding Dimension')
    plt.xlabel('Codebook Index')
    
    # Adjust layout to remove extra margins and whitespace
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
        tmpfile.flush()
        
        # Log the image to wandb
        wandb.log({
            'codebook/image': wandb.Image(tmpfile.name, caption=f'Epoch {epoch} - VQ Codebook Vectors')
        })
    
    plt.close()
    
    # Compute the magnitudes of the codebook vectors
    magnitudes = np.linalg.norm(codebook_vectors, axis=1)

    # Create a figure with one row and two columns for the histograms
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))

    # Plot the histogram of magnitudes
    axs[0].hist(magnitudes, bins=50, color='blue', edgecolor='black')
    axs[0].set_title('Histogram of Codebook Vector Magnitudes')
    axs[0].set_xlabel('Magnitude')
    axs[0].set_ylabel('Frequency')

    # Plot the histogram of elements
    axs[1].hist(codebook_vectors.flatten(), bins=200, color='blue', edgecolor='black')
    axs[1].set_title('Histogram of Codebook Vector Elements')
    axs[1].set_xlabel('Element Value')
    axs[1].set_ylabel('Frequency')

    # Adjust layout to remove extra margins and whitespace
    plt.tight_layout()

    # Save the histogram image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
        tmpfile.flush()

        # Log the histogram image to wandb
        wandb.log({
            'codebook/histograms': wandb.Image(tmpfile.name, caption=f'Epoch {epoch} - Histograms of Codebook Vectors')
        })

    plt.close()


@torch.no_grad()
def validate(model, vgg, loader, device, epoch, adv_loss=None, config=None):
    was_training = model.training
    model.train()  # For VQ loss
    # Start with basic losses, just like in train_epoch
    val_losses = {k: 0 for k in ['mse', 'vq', 'perceptual', 'spectral', 'total']}
    
    for batch_idx, batch in enumerate(loader):
        batch_losses, recon = process_batch(model, vgg, batch, device, False, adv_loss=adv_loss, epoch=epoch, config=config)
        
        # Only update losses that exist in batch_losses
        val_losses.update({k: val_losses[k] + batch_losses[k] for k in batch_losses.keys() & val_losses.keys()})
        
        # Add new losses from batch if they appear (after warmup)
        for k in batch_losses:
            if k not in val_losses:
                val_losses[k] = batch_losses[k]
        
        if batch_idx == 0 and not config.no_wandb:  # Log first batch visualizations
            orig = batch[0][:8].to(device)
            recon = glamp(recon[:8], orig.min(), orig.max())  # Note: assumes glamp exists
            viz_images = torch.cat([orig, recon])
            wandb.log({'epoch':epoch,
                'demo/examples': wandb.Image(make_grid(viz_images, nrow=8, normalize=True), 
                                          caption=f'Epoch {epoch} - Top: Source, Bottom: Recon'),
                **{f'validation/batch_{k}_loss': v for k, v in val_losses.items()}
            })
            
    if epoch % 1 == 0:# and epoch>0:
        viz_codebook(model, config, epoch)
    
    model.train(was_training)
    return {k: v / len(loader) for k, v in val_losses.items()}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=84) # for 16GB VRAM, 64x64 images, w/o grad checkpointing. For 128x128, set to 48 and turn on grad ckpt
    parser.add_argument('--epochs', type=int, default=1000000, help='number of epochs. (just let it keep training for hours/days/weeks/etc.)')
    parser.add_argument('--base-lr', type=float, default=1e-4, help='base learning rate for batch size of 32')
    parser.add_argument('--image-size', type=int, default=64, help='will rescale images to squares of (image-size, image-size)')
    parser.add_argument('--warmup-epochs', type=int, default=15, help='number of epochs before enabling adversarial loss')   
    parser.add_argument('--lambda-mse', type=float, default=0.5, help="regularization param for MSE loss")
    parser.add_argument('--lambda-vq', type=float, default=0.25, help="reg factor mult'd by VQ commitment loss")
    parser.add_argument('--lambda-perc', type=float, default=2e-4, help="regularization param for perceptual loss")
    parser.add_argument('--lambda-spec', type=float, default=1e-4,  help="regularization param for spectral loss (1e-4='almost off')")
    parser.add_argument('--lambda-adv', type=float, default=0.1,  help="regularization param for adversarial loss")
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint to resume training from')
    parser.add_argument('--hidden-channels', type=int, default=256)
    parser.add_argument('--vq-num-embeddings', type=int, default=1024, help='aka codebook length')
    parser.add_argument('--vq-embedding-dim', type=int, default=16*16, help='dims of codebook vectors')
    parser.add_argument('--num-downsamples', type=int, default=2, help='total downsampling is 2**[this]')
    parser.add_argument('--project-name', type=str, default="vqgan-shh", help='WandB project name')
    parser.add_argument('--no-grad-ckpt', action='store_true', help='disable gradient checkpointing (disabled uses more memory but faster)') 

    args = parser.parse_args()
    args.learning_rate = args.base_lr * math.sqrt(args.batch_size / 32)
    print("args = ",args)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    train_loader, val_loader = create_loaders(args.batch_size, args.image_size)
    
    # Initialize VGG for perceptual loss
    vgg = vgg16(weights='IMAGENET1K_V1').features[:16].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    
    model = ImprovedLucidVQVAE(
        in_channels=3, 
        hidden_channels=args.hidden_channels,
        num_downsamples=args.num_downsamples,
        vq_num_embeddings=args.vq_num_embeddings,
        vq_embedding_dim=args.vq_embedding_dim,
        use_checkpoint=not args.no_grad_ckpt,  # checkpointing uses less VRAM but is a bit slower
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    min_lr = 1e-5
    lambda1 = lambda epoch: max(min_lr / args.learning_rate, 0.97 ** epoch)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    # descriminator  for adv loss
    adv_loss = AdversarialLoss(device).to(device)  # device twice may be overkill 
    d_optimizer = optim.Adam(adv_loss.discriminator.parameters(),  weight_decay=1e-5,
                            lr=args.learning_rate * 0.1)  # 10x smaller than G
                            
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    
    if not args.no_wandb:
        wandb.init(project=args.project_name)
        wandb.config.update(vars(args))
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if epoch == args.warmup_epochs: print("*** WARMUP PERIOD FINISHED. Engaging adversarial training. ***")

        train_losses = train_epoch(model, vgg, train_loader, optimizer, d_optimizer, device, epoch, adv_loss, config=args)
        val_losses = validate(model, vgg, val_loader, device, epoch, adv_loss, config=args)

        if not args.no_wandb: 
            wandb.log({ 
            **{f'epoch/train_{k}_loss': v for k, v in train_losses.items()},
            **{f'epoch/val_{k}_loss': v for k, v in val_losses.items()},
            'epoch': epoch,
            'learning_rate': optimizer.param_groups[0]['lr'] })
                
        # Save checkpoint
        if (epoch) % 20 == 0 and epoch>0:
            ckpt_path = f'checkpoints/model_epoch{epoch}.pt'
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
        scheduler.step()


if __name__ == '__main__':
    main()
