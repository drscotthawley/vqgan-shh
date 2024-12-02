import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import random
from vector_quantize_pytorch import VectorQuantize


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
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
    
    def forward(self, x):
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

class LucidVQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, num_downsamples=3, 
                 vq_num_embeddings=512, vq_embedding_dim=128):
        super().__init__()
        
        self.num_downsamples = num_downsamples  # Store num_downsamples as an instance variable
        
        # Encoder
        encoder_layers = []
        in_channels_current = in_channels
        for i in range(num_downsamples):
            out_channels = hidden_channels * (2 ** i)
            encoder_layers.append(ResidualBlock(in_channels_current, out_channels, stride=2))
            in_channels_current = out_channels
        encoder_layers.append(nn.Conv2d(in_channels_current, vq_embedding_dim, 1))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer
        self.vq = VectorQuantize(
            dim=vq_embedding_dim,
            codebook_size=vq_num_embeddings,
            decay=0.8,
            commitment_weight=0.25
        )
        
        # Decoder
        decoder_layers = []

        # Initial projection from embedding dim
        decoder_layers.append(nn.Conv2d(vq_embedding_dim, hidden_channels * (2 ** (num_downsamples - 1)), 1))
        decoder_layers.append(nn.GroupNorm(8, hidden_channels * (2 ** (num_downsamples - 1))))
        decoder_layers.append(nn.SiLU())

        # Upsampling steps
        for i in range(num_downsamples - 1, 0, -1):
            in_channels_current = hidden_channels * (2 ** i)
            out_channels = hidden_channels * (2 ** (i - 1))
            decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            decoder_layers.append(nn.Conv2d(in_channels_current, out_channels, 3, stride=1, padding=1))
            decoder_layers.append(nn.GroupNorm(8, out_channels))
            decoder_layers.append(nn.SiLU())

        # Final upsampling to original input channels
        decoder_layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        decoder_layers.append(nn.Conv2d(hidden_channels, in_channels, 3, stride=1, padding=1))

        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        z = self.encode(x)  # [B, embed_dim, H, W]
        z = z.permute(0, 2, 3, 1)  # [B, H, W, embed_dim]
        z = z.reshape(-1, z.shape[-1])  # [B*H*W, embed_dim]
        
        z_q, indices, commit_loss = self.vq(z)
        
        z_q = z_q.view(x.shape[0], x.shape[2] // (2 ** self.num_downsamples), x.shape[3] // (2 ** self.num_downsamples), -1)
        z_q = z_q.permute(0, 3, 1, 2)
        
        x_recon = self.decode(z_q)
        return x_recon, commit_loss


class ImprovedLucidVQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, num_downsamples=3, 
                 vq_num_embeddings=512, vq_embedding_dim=128):
        super().__init__()
        
        self.num_downsamples = num_downsamples
        
        # Encoder - Now with more residual blocks
        encoder_layers = []
        in_channels_current = in_channels
        for i in range(num_downsamples):
            out_channels = hidden_channels * (2 ** i)
            # Two residual blocks per downsample for better feature extraction
            encoder_layers.append(ResidualBlock(in_channels_current, out_channels, stride=2))
            encoder_layers.append(ResidualBlock(out_channels, out_channels, stride=1))
            in_channels_current = out_channels
            
        encoder_layers.append(ResidualBlock(in_channels_current, vq_embedding_dim, stride=1))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Improved Vector Quantizer with better initialization
        self.vq = VectorQuantize(
            dim=vq_embedding_dim,
            codebook_size=vq_num_embeddings,
            decay=0.95,  # Slower decay for more stable codebook
            commitment_weight=1.0,  # Stronger commitment
            kmeans_init=True,  # Initialize with k-means
            threshold_ema_dead_code=2
        )
        
        # Enhanced Decoder with residual connections
        decoder_layers = []
        
        # Initial projection
        current_channels = hidden_channels * (2 ** (num_downsamples - 1))
        decoder_layers.extend([
            nn.Conv2d(vq_embedding_dim, current_channels, 1),
            ResidualBlock(current_channels, current_channels)
        ])

        # Upsampling blocks
        for i in range(num_downsamples - 1, -1, -1):
            out_channels = hidden_channels * (2 ** max(0, i - 1))
            if i == 0:
                out_channels = hidden_channels
                
            decoder_layers.extend([
                # Bilinear upsampling for smoother results
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ResidualBlock(current_channels, out_channels),
                ResidualBlock(out_channels, out_channels)
            ])
            current_channels = out_channels

        # Final layers with fewer normalizations for better detail
        decoder_layers.extend([
            nn.Conv2d(current_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        ])

        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, x):
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