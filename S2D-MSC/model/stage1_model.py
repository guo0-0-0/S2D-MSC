from functools import partial
from SGSA import *
import numpy as np

"""
Reference:
https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
Create 2D grid representing the positional embeddings of the frames.
embed_dim is the dimention of the embedding for encoder / decoder,
grid_size is the squareroot of the number of patches.
"""


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            bias=True
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Siamese_encoder(nn.Module):
    def __init__(
            self,
            img_size=32,
            patch_size=1,
            in_chans=2,
            embed_dim=96,
            depth=6,
            num_heads=4,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False)

        self.blocks = nn.ModuleList([
            BaseBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5),
            cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio, pos):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        ############ fixed sampling #######################
        noise = torch.rand(N, L, device=x.device) + 1
        noise_pos = torch.rand(N, len_keep, device=x.device)
        noise.scatter_(dim=1, index=pos, src=noise_pos)

        # ############# random sampling #######################
        # noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio, pos):

        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio > 0 :
            x, mask, ids_restore = self.random_masking(x, mask_ratio, pos=pos)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if mask_ratio > 0:
            return x, mask, ids_restore
        else:
            return x


class Siamese_decoder(nn.Module):
    def __init__(
            self,
            patch_size=1,
            in_chans=2,
            num_patches=1024,
            embed_dim=96,
            decoder_embed_dim=384,
            decoder_depth=6,
            decoder_num_heads=4,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm
    ):

        super().__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim,
                 decoder_num_heads,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=norm_layer
                 )
            for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.patch_size ** 2 * in_chans,
            bias=True
        )

        self.initialize_weights()

    def initialize_weights(self):

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** .5),
            cls_token=True
        )

        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, f1, f2, ids_restore_2, SGSA_mask=None):

        x_1 = self.decoder_embed(f1)
        x_1 = x_1 + self.decoder_pos_embed

        x_2 = self.decoder_embed(f2)

        mask_tokens = self.mask_token.repeat(
            f2.shape[0],
            ids_restore_2.shape[1] + 1 - x_2.shape[1],
            1
        )

        x_2_ = torch.cat([x_2[:, 1:, :], mask_tokens], dim=1)

        x_2_ = torch.gather(
            x_2_,
            dim=1,
            index=ids_restore_2.unsqueeze(-1).repeat(1, 1, x_2.shape[2])
        )

        x_2 = torch.cat([x_2[:, :1, :], x_2_], dim=1)

        x_2 = x_2 + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x_2 = blk(x_1, x_2, SGSA_mask=SGSA_mask)

        x = self.decoder_norm(x_2)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        return x


def build_SGSA_mask_fast(pos, H, W, device, dtype=torch.float32):
    """

    """
    B, num_sample = pos.shape
    N_patch = H * W
    N = N_patch + 1  # +1 for cls token

    mask = torch.full((B, N, N), float('-inf'), device=device, dtype=dtype)
    pos = pos + 1

    batch = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_sample)

    mask[batch, :, pos] = 0.0

    mask[:, 0, :] = 0.0   
    mask[:, :, 0] = 0.0  

    return mask


class SiameseAutoencoderViT(nn.Module):
    def __init__(
            self,
            window_size=4,
            img_size=32,
            patch_size=1,
            in_chans=2,
            embed_dim=192,
            depth=6,
            num_heads=4,
            decoder_embed_dim=384,
            decoder_depth=6,
            decoder_num_heads=4,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm

    ):
        super().__init__()

        self.encoder = Siamese_encoder(img_size=img_size,
                                        patch_size=patch_size,
                                        in_chans=in_chans,
                                        embed_dim=embed_dim,
                                        depth=depth,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        norm_layer=norm_layer
                                        )


        self.decoder = Siamese_decoder(patch_size=patch_size,
                                        in_chans=in_chans,
                                        num_patches=1024,
                                        embed_dim=embed_dim,
                                        decoder_embed_dim=decoder_embed_dim,
                                        decoder_depth=decoder_depth,
                                        decoder_num_heads=decoder_num_heads,
                                        mlp_ratio=mlp_ratio,
                                        norm_layer=norm_layer
                                       )

        self.img_size = img_size
        self.patch_size = patch_size

    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 2))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 2, h * p, h * p))
        return imgs

    def forward(self, imgs, mask_ratio, pos):

        device = imgs.device

        SGSA_mask = build_SGSA_mask_fast(
            pos,
            self.img_size,
            self.img_size,
            device
        )

        latent_1 = self.encoder(imgs[:, 0].float(), mask_ratio=0, pos=pos)  
        latent_2, mask_2, ids_restore_2 = self.encoder(imgs[:, 1].float(), mask_ratio=mask_ratio, pos=pos)  

        x1 = self.decoder(latent_1, latent_2, ids_restore_2, SGSA_mask=SGSA_mask)  # shape [b (h w) c]

        x = self.unpatchify(x1)
        return x





