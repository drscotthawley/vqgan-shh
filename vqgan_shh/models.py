import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from vector_quantize_pytorch import VectorQuantize
import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True") # annoying warnings when grad checkpointing. it's fine, really


class EncDecResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_channels)
            )
        self.use_checkpoint = use_checkpoint
    
    def _forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.silu(out)
        return out

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, num_downsamples=3, 
                 vq_num_embeddings=512, vq_embedding_dim=128, use_checkpoint=False):
        super().__init__()
        
        self.num_downsamples = num_downsamples
        self.use_checkpoint = use_checkpoint
        
        # Encoder with checkpointing
        encoder_layers = []
        in_channels_current = in_channels
        for i in range(num_downsamples):
            out_channels = hidden_channels * (2 ** i)
            encoder_layers.append(EncDecResidualBlock(in_channels_current, out_channels, 
                                stride=2, use_checkpoint=use_checkpoint))
            encoder_layers.append(EncDecResidualBlock(out_channels, out_channels, 
                                stride=1, use_checkpoint=use_checkpoint))
            in_channels_current = out_channels
                
        encoder_layers.append(EncDecResidualBlock(in_channels_current, vq_embedding_dim, 
                            stride=1, use_checkpoint=use_checkpoint))
        encoder_layers.append(nn.Conv2d(vq_embedding_dim, vq_embedding_dim, 1)) # final conv2d undoes swish at end of EncDecResidualBlock
        self.encoder = nn.Sequential(*encoder_layers)

        # Vector Quantizer
        self.vq = VectorQuantize(
            dim=vq_embedding_dim,
            codebook_size=vq_num_embeddings,
            decay=0.95,
            commitment_weight=1.0,
            kmeans_init=True,
            threshold_ema_dead_code=2
        )
        
        # Decoder with checkpointing
        decoder_layers = []
        
        # Initial projection
        current_channels = hidden_channels * (2 ** (num_downsamples - 1))
        initial_layers = [
            nn.Conv2d(vq_embedding_dim, current_channels, 1),
            EncDecResidualBlock(current_channels, current_channels, use_checkpoint=use_checkpoint)
        ]
        decoder_layers.extend(initial_layers)

        # Upsampling blocks
        for i in range(num_downsamples - 1, -1, -1):
            out_channels = hidden_channels * (2 ** max(0, i - 1))
            if i == 0:
                out_channels = hidden_channels
                
            block = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                EncDecResidualBlock(current_channels, out_channels, use_checkpoint=use_checkpoint),
                EncDecResidualBlock(out_channels, out_channels, use_checkpoint=use_checkpoint)
            ]
            decoder_layers.extend(block)
            current_channels = out_channels

        # Final layers
        final_layers = [
            nn.Conv2d(current_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        ]
        decoder_layers.extend(final_layers)

        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self.encoder, x, use_reentrant=False)
        return self.encoder(x)
        
    def decode(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self.decoder, x, use_reentrant=False)
        return self.decoder(x)
        
    def forward(self, x):
            
        z = self.encode(x)
        z = z.permute(0, 2, 3, 1)
        z = z.reshape(-1, z.shape[-1])
        
        z_q, indices, commit_loss = self.vq(z)
        
        z_q = z_q.view(x.shape[0], x.shape[2] // (2 ** self.num_downsamples), 
                      x.shape[3] // (2 ** self.num_downsamples), -1)
        z_q = z_q.permute(0, 3, 1, 2)
        
        x_recon = self.decode(z_q)
        return x_recon, commit_loss
    


## ------------------adversarial Discriminator ----------------------------

class DiscrResBlock(nn.Module):
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
        PatchGAN discriminator with DiscrResBlocks.
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
        
        # Intermediate layers with DiscrResBlocks
        current_channels = hidden_channels
        for i in range(n_layers):
            next_channels = min(hidden_channels * (2 ** (i+1)), 512)
            layers.append(DiscrResBlock(current_channels, next_channels, 
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
            if isinstance(layer, (nn.LeakyReLU, DiscrResBlock)):
                features.append(x)
        return x, features

