from model.stage1_model import SiameseAutoencoderViT  # 
from functools import partial


def build_SGSA_mask_fast(pos, H, W, device, dtype=torch.float32):

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


class Stage2_ViT(nn.Module):
    def __init__(self, pretrained_path=None,
                    img_size=32,
                    patch_size=1,
                    in_chans=2,
                    embed_dim=192,
                    depth=6,
                    num_heads=4,
                    decoder_embed_dim=384,
                    decoder_depth=6,
                    decoder_num_heads=4,
                    mlp_ratio=4.
                 ):
        super().__init__()

    
        pretrained_model = SiameseAutoencoderViT(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim,  depth=depth, num_heads=num_heads,decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))


        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            pretrained_model.load_state_dict(state_dict, strict=False)

            print(f"Loaded encoder weights from {pretrained_path}")

        self.encoder = pretrained_model.encoder
        self.decoder = pretrained_model.decoder

        self.patch_size = patch_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_size ** 2 * in_chans, bias=True)
        self.img_size = img_size
        

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






