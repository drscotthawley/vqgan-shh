import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from vector_quantize_pytorch import VectorQuantize
import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True") # annoying warnings when grad checkpointing. it's fine, really
#from flash_attn import flash_attn_func

    
try:
    import natten
    print("Using NATTEN version ",natten.__version__)
except ImportError:
    warnings.warn("Warning: NATTEN not found. Running without.")
    natten = None




class Normalize(nn.Module):
    def __init__(self, num_channels, num_groups=32, eps=1e-6, affine=True):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, 
                                num_channels=num_channels, 
                                eps=eps, 
                                affine=affine)

    def forward(self, x):
        return self.norm(x)
    
class AttnBlock(nn.Module):# Use the AttnBlock from the official VQGAN code
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class NATTENBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, num_heads=8, init_scale=0.02):
        super().__init__()
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.head_dim = dim // num_heads
        self.scaling = (self.head_dim ** -0.5) * 0.5
        
        # Replace LayerNorm with GroupNorm
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Initialize with smaller weights
        nn.init.normal_(self.qkv.weight, std=init_scale)
        nn.init.normal_(self.proj.weight, std=init_scale)
        
        if natten is None:
            raise ImportError("Please install NATTEN: pip install natten")
            
    def _forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # Apply GroupNorm (works in channel-first format)
        x = self.norm(x)
        
        # Only permute once for the linear layers
        x = x.permute(0, 2, 3, 1)  # B H W C
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, H, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # 3 B heads H W dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = natten.functional.na2d_qk(q, k, self.kernel_size)
        attn = attn.softmax(dim=-1)
        x = natten.functional.na2d_av(attn, v, self.kernel_size)
            
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)  # Back to B C H W
        return identity + (x * self.gamma)  # Scaled attention plus identity
        
    def forward(self, x):
        if x.requires_grad:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)



class EncDecResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False, attention='natten'):
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
        if attention == 'natten' and natten:
            self.attn = NATTENBlock(out_channels)
        elif attention == 'full':  
            self.attn = AttnBlock(out_channels)
        else:
            self.attn = None


    def _forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.silu(out)

        if self.attn:
            out = self.attn(out) 

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
            if i >= num_downsamples-2: 
                attention = 'natten'
            else: 
                attention = None
            encoder_layers.append(EncDecResidualBlock(in_channels_current, out_channels, 
                                stride=2, use_checkpoint=use_checkpoint, attention=attention))
            encoder_layers.append(EncDecResidualBlock(out_channels, out_channels, 
                                stride=1, use_checkpoint=use_checkpoint, attention=attention))
            in_channels_current = out_channels
                
        encoder_layers.append(EncDecResidualBlock(in_channels_current, vq_embedding_dim, 
                            stride=1, use_checkpoint=use_checkpoint, attention=attention))
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
            EncDecResidualBlock(current_channels, current_channels, use_checkpoint=use_checkpoint, attention='full') # attn is cheap at this resolution
        ]
        decoder_layers.extend(initial_layers)

        # Upsampling blocks
        for i in range(num_downsamples - 1, -1, -1):
            out_channels = hidden_channels * (2 ** max(0, i - 1))
            if i == 0:
                out_channels = hidden_channels
            if i > num_downsamples-2:
                attention, mode = 'natten','bicubic'
            else:
                attention, mode = None, 'bilinear'
            block = [
                nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
                EncDecResidualBlock(current_channels, out_channels, use_checkpoint=use_checkpoint, attention=attention),
                EncDecResidualBlock(out_channels, out_channels, use_checkpoint=use_checkpoint, attention=None)
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

