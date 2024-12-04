#!/bin/env python3 
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # re. NATTEN's non-stop FutureWarnings
import torch
torch.set_float32_matmul_precision('high')
import math
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torchvision.utils import make_grid
import wandb
from tqdm.auto import tqdm
import argparse
import matplotlib.pyplot as plt


from vqgan_shh.data import create_loaders
from vqgan_shh.models import VQVAE
from vqgan_shh.losses import *
from vqgan_shh.viz import viz_codebook


def process_batch(model, vgg, batch, device, is_train=True, optimizer=None, d_optimizer=None, 
                 adv_loss=None, epoch=None, config=None, batch_idx=0):
    """Process a single batch and return losses.  Called by training and/or validation"""
    source_imgs = batch[0].to(device)
    target_imgs = source_imgs
    count = 1 # we train the generator on 1 batch unless otherwise noted

    # Pre-warmup behavior - sectioned this off b/c one time I broke everything when I added the GAN part. 
    if epoch < config.warmup_epochs:
        recon, vq_loss = model(source_imgs)
        losses = compute_losses(recon, target_imgs, vq_loss, vgg, adv_loss=None, epoch=None, config=config)
        losses['total'] = get_total_loss(losses, config)
        
        if is_train:
            optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
        return {k: v.item() for k, v in losses.items()}, recon, count

    # Post-warmup behavior
    else:
        # Train discriminator multiple times
        d_losses, d_stats_list, grad_stats_list = [], [], []
        
        if is_train and adv_loss is not None:
            recon, vq_loss = model(source_imgs)  
            for _ in range(1):  # Train D this many times per batch        
                recon = recon.detach()  # Detach since we only need for D training
                d_loss, real_features = adv_loss.discriminator_loss(target_imgs, recon)
                d_stats = get_discriminator_stats(adv_loss, target_imgs, recon)
                d_stats_list.append(d_stats)  

                d_optimizer.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(adv_loss.discriminator.parameters(), max_norm=1.0)  
                grad_stats = get_gradient_stats(adv_loss.discriminator)
                grad_stats_list.append(grad_stats)  
                d_losses.append(d_loss.item())  
                d_optimizer.step()

        # Train generator (less frequently)
        g_losses = {}
        if not is_train or batch_idx % 1 == 0:  # Update G every so many batches
            recon, vq_loss = model(source_imgs)
            losses = compute_losses(recon, target_imgs, vq_loss, vgg, adv_loss, epoch, config=config)
            losses['total'] = get_total_loss(losses, config)
            
            if is_train:
                optimizer.zero_grad()
                losses['total'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            g_losses = {k: v.item() for k, v in losses.items()}
        else:
            count = 0

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

        return combined_losses, recon, count



def train_epoch(model, vgg, loader, optimizer, d_optimizer, device, epoch, adv_loss=None, config=None):
    model.train()
    # Start with basic losses
    epoch_losses = {k: 0 for k in ['mse', 'vq', 'perceptual', 'spectral', 'total']}
    
    total_batches = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        batch_losses, recon, count = process_batch(model, vgg, batch, device, True, optimizer, d_optimizer=d_optimizer, adv_loss=adv_loss, epoch=epoch, config=config, batch_idx=batch_idx )
        total_batches += count
        # Only update losses that exist in batch_losses
        epoch_losses.update({k: epoch_losses[k] + batch_losses[k] for k in batch_losses.keys() & epoch_losses.keys()})
        
        # Add new losses from batch if they appear (after warmup)
        for k in batch_losses:
            if k not in epoch_losses:
                epoch_losses[k] = batch_losses[k]
        
        pbar.set_postfix({k: f'{v:.4g}' for k, v in batch_losses.items()})
        
        if batch_idx % 100 == 0 and not config.no_wandb:
            wandb.log({f'batch/{k}_loss': v for k, v in batch_losses.items()} | {'epoch':epoch})

    return {k: v / total_batches for k, v in epoch_losses.items()}


@torch.no_grad()
def validate(model, vgg, loader, device, epoch, adv_loss=None, config=None):
    was_training = model.training
    model.train()  # For VQ loss
    # Start with basic losses, just like in train_epoch
    val_losses = {k: 0 for k in ['mse', 'vq', 'perceptual', 'spectral', 'total']}
    
    total_batches = 0
    for batch_idx, batch in enumerate(loader):
        batch_losses, recon, count = process_batch(model, vgg, batch, device, False, adv_loss=adv_loss, epoch=epoch, config=config)
        total_batches += count 
        # Only update losses that exist in batch_losses
        val_losses.update({k: val_losses[k] + batch_losses[k] for k in batch_losses.keys() & val_losses.keys()})
        
        # Add new losses from batch if they appear (after warmup)
        for k in batch_losses:
            if k not in val_losses:
                val_losses[k] = batch_losses[k]
        
        if batch_idx == 0 and not config.no_wandb:  # Log first batch visualizations
            orig = batch[0][:8].to(device)
            recon = torch.clamp(recon[:8], orig.min(), orig.max()) 
            viz_images = torch.cat([orig, recon])
            wandb.log({'epoch':epoch,
                'demo/examples': wandb.Image(make_grid(viz_images, nrow=8, normalize=True), 
                                          caption=f'Epoch {epoch} - Top: Source, Bottom: Recon'),
                **{f'validation/batch_{k}_loss': v for k, v in val_losses.items()}
            })
            
    if epoch % 1 == 0:# and epoch>0:
        viz_codebook(model, config, epoch)
    
    model.train(was_training)
    return {k: v / total_batches for k, v in val_losses.items()}



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Training parameters
    # my tests:
    #  --image-size=64 --batch-size=84 --warmup-epochs=15
    #  --image-size=128 --batch-size=56 --warmup-epochs=15
    parser.add_argument('--batch-size', type=int, default=56) # for 16GB VRAM, 64x64 images, w/o grad checkpointing. For 128x128, set to 56 and turn on grad ckpt
    parser.add_argument('--data', type=str, default=None, help='path to top-level-directory containing custom image data. If not specified, uses Flowers102')
    parser.add_argument('--epochs', type=int, default=1000000, help='number of epochs. (just let it keep training for hours/days/weeks/etc.)')
    # for wandb sweeps
    #parser.add_argument('--epochs', type=int, default=300, help='number of epochs. (just let it keep training for hours/days/weeks/etc.)')
    parser.add_argument('--base-lr', type=float, default=1e-4, help='base learning rate for batch size of 32')
    parser.add_argument('--image-size', type=int, default=128, help='will rescale images to squares of (image-size, image-size)')
    parser.add_argument('--warmup-epochs', type=int, default=15, help='number of epochs before enabling adversarial loss')   
    parser.add_argument('--lambda-mse', type=float, default=0.5, help="regularization param for MSE loss")
    parser.add_argument('--lambda-vq', type=float, default=0.25, help="reg factor mult'd by VQ commitment loss")
    parser.add_argument('--lambda-perc', type=float, default=2e-4, help="regularization param for perceptual loss")
    parser.add_argument('--lambda-spec', type=float, default=1e-4,  help="regularization param for spectral loss (1e-4='almost off')") # with lambda_spec=0, spec_loss serves as an independent metric
    parser.add_argument('--lambda-adv', type=float, default=0.03,  help="regularization param for G part of adversarial loss")
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
    
    model = VQVAE(
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

    # Descriminator for adversarial loss
    adv_loss = AdversarialLoss(device, use_checkpoint=not args.no_grad_ckpt).to(device)  # device twice may be overkill 
    d_optimizer = optim.Adam(adv_loss.discriminator.parameters(),  weight_decay=1e-5,
                            lr=args.learning_rate * 0.1)  # D LR. maybe different LR from G. 
                          
    start_epoch = 0
    if args.checkpoint is not None: # Start from a checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    
    if not args.no_wandb:
        wandb.init(project=args.project_name)
        wandb.config.update(vars(args))

    os.makedirs('checkpoints', exist_ok=True) 
    
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
