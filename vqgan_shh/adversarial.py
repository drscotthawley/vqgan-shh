import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True") # annoying warnings when grad checkpointing. it's fine, really


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.skip = None if stride == 1 and in_channels == out_channels else \
                   spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0))
        self.norm1 = nn.GroupNorm(min(32, out_channels//4), out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels//4), out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x):
        identity = self.skip(x) if self.skip else x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        return self.act(out)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)



class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, n_layers=3, use_checkpoint=False):
        """
        PatchGAN discriminator with ResidualBlocks.
        Args:
            in_channels: Number of input channels (3 for RGB)
            hidden_channels: Base channel count
            n_layers: Number of downsampling layers
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        
        # Initial conv layer
        layers = [
            spectral_norm(nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers with ResidualBlocks
        current_channels = hidden_channels
        for i in range(n_layers):
            next_channels = min(hidden_channels * (2 ** (i+1)), 512)
            layers.append(ResidualBlock(current_channels, next_channels, 
                                      stride=2 if i < n_layers-1 else 1,
                                      use_checkpoint=use_checkpoint))
            current_channels = next_channels
            
        # Final layer for patch-wise predictions
        layers.append(
            spectral_norm(nn.Conv2d(current_channels, 1, kernel_size=4, 
                                  stride=1, padding=1))
        )
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        features = []
        for layer in self.main:
            x = layer(x)
            if isinstance(layer, (nn.LeakyReLU, ResidualBlock)):
                features.append(x)
        return x, features



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

    # def discriminator_loss(self, real_images, fake_images):
    #     # Real images
    #     real_pred, real_features = self.discriminator(real_images)
    #     d_real_loss = self.criterion(real_pred, self.get_target_tensor(real_pred, True))
        
    #     # Fake images
    #     fake_pred, fake_features = self.discriminator(fake_images.detach())
    #     d_fake_loss = self.criterion(fake_pred, self.get_target_tensor(fake_pred, False))
        
    #     return (d_real_loss + d_fake_loss) * 0.5, real_features

    # def generator_loss(self, fake_images, real_features=None):
    #     fake_pred, fake_features = self.discriminator(fake_images)
    #     g_loss = self.criterion(fake_pred, self.get_target_tensor(fake_pred, True))
        
    #     # Add feature matching loss if real_features provided
    #     if real_features is not None:
    #         fm_loss = self.feature_matching_loss(real_features, fake_features)
    #         g_loss = g_loss + fm_loss
            
    #     return g_loss
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
    
