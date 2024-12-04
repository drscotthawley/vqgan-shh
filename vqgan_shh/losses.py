import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True") # annoying warnings when grad checkpointing. it's fine, really

from .models import PatchDiscriminator


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
    if adv_loss is not None and epoch >= config.warmup_epochs:
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


#------ not used as losses but are kind of loss-like:
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
#---------


def hinge_d_loss(real_pred, fake_pred):
    return torch.mean(F.relu(1.0 - real_pred)) + torch.mean(F.relu(1.0 + fake_pred))



class AdversarialLoss(nn.Module):
    def __init__(self, device, use_checkpoint=False):
        super().__init__()
        self.device = device
        self.discriminator = PatchDiscriminator(use_checkpoint=use_checkpoint).to(device)
        self.criterion = hinge_d_loss

        self.register_buffer('real_label', torch.ones(1))
        self.register_buffer('fake_label', torch.zeros(1))
        self.to(device)


    def get_target_tensor(self, prediction, target_is_real):
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def feature_matching_loss(self, real_features, fake_features):
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss / len(real_features)

    def discriminator_loss(self, real_images, fake_images):
        real_pred, real_features = self.discriminator(real_images)
        fake_pred, _ = self.discriminator(fake_images.detach())
        return hinge_d_loss(real_pred, fake_pred), real_features

    def generator_loss(self, fake_images, real_features=None):
        fake_pred, fake_features = self.discriminator(fake_images)
        g_loss = -torch.mean(fake_pred)
        if real_features is not None:
            fm_loss = self.feature_matching_loss(real_features, fake_features)
            g_loss = g_loss + fm_loss
        return g_loss
    
