from torch import nn
import torch
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor

class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = nn.functional.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x

class Attention(nn.Module):
    def __init__(self, dim: int, 
                 num_heads: int, 
                 dropout: float=0.):
        super().__init__()
        self.heads = num_heads
        self.scale = dim ** -0.5
        self.se_layer = SELayer(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        # print("b:",b)
        # print("n:",n)
        # print("_:",_)
        # print("h:",h)
        x = self.se_layer(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Mlp(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 mlp_dim: int,
                 out_dim: int):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act_fn = torch.nn.GELU()
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, 
                 dim: int,
                 num_heads: int,
                 mlp_dim: int,):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim, mlp_dim, dim)
        self.attn = Attention(dim, num_heads)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, 
                 dim: int,
                 mlp_dim: int,
                 num_layers: int,
                 num_heads: int,):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(dim, eps=1e-6)
        for _ in range(num_layers):
            self.layer.append(
                Block(dim, num_heads, mlp_dim)
            )

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class VisionTransformer(nn.Module):
    def __init__(self, dim: int, 
                 num_layers: int, 
                 num_heads: int, 
                 mlp_dim: int, 
                 dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(
                    dim,
                    num_heads,
                    dropout
                ))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim)
            # nn.Linear(dim, 512)
        )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        x = self.to_latent(x)
        x = self.mlp_head(x)
        return x


class MAETransformer(nn.Module):
    def __init__(self, 
                 encoder_dim: int=768,
                 mlp_dim: int=3072,
                 enc_num_heads: int=12,
                 enc_num_layers: int=12,
                 decoder_dim: int=512,
                 dec_num_heads: int=6,
                 dec_num_layers: int=8,
                 img_size: tuple[int, int, int]=(64, 128, 128),
                 masking_ratio: float=0.75,
                 patch_size: int=16,
                 dropout: float=0.1):
        super().__init__()
        
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.patch_dim_in = patch_size * patch_size * patch_size
        self.patch_dim_out = patch_size * patch_size * patch_size
        # Gets the patch position embeddings
        self.patch_position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, encoder_dim))

        self.to_patches = Rearrange('b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)',
                                    p1=patch_size, p2=patch_size, p3=patch_size)
        self.patch_to_encoder = nn.Linear(self.patch_dim_in, encoder_dim)
        self.encoder = Encoder(
            dim=encoder_dim,
            mlp_dim=mlp_dim,
            num_layers=enc_num_layers,
            num_heads=enc_num_heads,
        )

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()

        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim))
        self.decoder_pos_embeddings = nn.Embedding(self.num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, self.patch_dim_out)

        self.decoder = VisionTransformer(decoder_dim,
                                         dec_num_layers,
                                         dec_num_heads,
                                         mlp_dim=decoder_dim * 4)

        self.dropout = nn.Dropout(dropout)
        

    def random_masking(self, n_patches, device):
        len_keep = int(n_patches * (1 - self.masking_ratio))

        noise = torch.rand(n_patches, device=device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise)  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep = ids_shuffle[:len_keep]
        ids_remove = ids_shuffle[len_keep:]

        return ids_keep, ids_remove

    def forward(self, x: torch.Tensor, masked_loss: bool=True) -> tuple[torch.Tensor, torch.Tensor|None]:
        device = x.device

        # Step 1: Convert images to patches
        patches = self.to_patches(x) # (B, N, patch_dim)

        batch, n_patches, _ = patches.shape

        tokens = self.patch_to_encoder(patches) # (B, N, encoder_dim)

        tokens += self.patch_position_embeddings[:, :n_patches, :]
        if self.masking_ratio > 0:
            unmasked_indices, masked_indices = self.random_masking(n_patches, device)

            x_masked = tokens[:, masked_indices, :]
            x_unmasked = tokens[:, unmasked_indices, :]
        else:
            x_unmasked = tokens
            unmasked_indices = torch.arange(n_patches, device=device)
            masked_indices = torch.tensor([], device=device, dtype=torch.long)
        # Step 2: Encode the unmasked patches
        encoded_tokens, _ = self.encoder(x_unmasked)

        # Step 3: Prepare decoder input
        decoder_tokens = self.enc_to_dec(encoded_tokens)
        if self.masking_ratio > 0:
            len_keep = decoder_tokens.shape[1]
            len_mask = n_patches - len_keep
            mask_tokens = self.mask_token.repeat(batch, len_mask, 1)
            decoder_tokens_ = torch.zeros(batch, n_patches, decoder_tokens.size(-1), device=device)
            decoder_tokens_[:, unmasked_indices, :] = decoder_tokens
            decoder_tokens_[:, masked_indices, :] = mask_tokens
            decoder_tokens_[:, masked_indices, :] += self.decoder_pos_embeddings(masked_indices)
            decoder_tokens_[:, unmasked_indices, :] += self.decoder_pos_embeddings(unmasked_indices)
        else:
            decoder_tokens_ = decoder_tokens + self.decoder_pos_embeddings(
                torch.arange(n_patches, device=device)
            )
        decoded_tokens = self.decoder(decoder_tokens_)
        B, N, C = decoded_tokens.shape
        x_patch_recon = decoded_tokens.permute(0, 2, 1).reshape(B, C, 
                                                                int(x.size(2) / self.patch_size),
            int(x.size(3) / self.patch_size),
            int(x.size(4) / self.patch_size),
        )  # (B, C, D', H', W')
        # Step 4: Reconstruct the pixels
        reconstructed_patches = self.to_pixels(decoded_tokens)  # (B, N, patch_dim)
        reconstructed_patches = rearrange(reconstructed_patches, 'b (x y z) (p1 p2 p3 c) -> b c (x p1) (y p2) (z p3)',
                                          p1=self.patch_size,
                                          p2=self.patch_size,
                                          p3=self.patch_size,
                                          x=int(x.size(2) / self.patch_size),
                                          y=int(x.size(3) / self.patch_size),
                                          z=int(x.size(4) / self.patch_size),
                                          c=1
                                          )
        # Step 5: Reassemble the patches into images
        targets = patches
        pred_masked = reconstructed_patches[:, masked_indices, :]
        targets_masked = targets[:, masked_indices, :]
        
        if masked_loss and self.masking_ratio > 0:
            loss = F.mse_loss(pred_masked, targets_masked)
            return reconstructed_patches, loss
        
        return reconstructed_patches, None