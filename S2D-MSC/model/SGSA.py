import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp
import torch.nn as nn


class BaseAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, SGSA_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  

        # attn shape: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if SGSA_mask is not None:
            attn = attn + SGSA_mask.unsqueeze(1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x
    

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        qkv_1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv_1.unbind(0)

        qkv_2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q2, k2, v2 = qkv_2.unbind(0)

        x = F.scaled_dot_product_attention(q2, k1, v1)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class BaseBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = BaseAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )


    def forward(self, x, SGSA_mask=None):
        res = x
        x_ = self.norm1(x)
        x = self.attn(x_)
        x = res + x
        x = x + self.mlp(self.norm2(x))

        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            drop=0.
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )

        self.attn = BaseAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )

        self.cross_attention = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )


        self.SGSA_attn = BaseAttention(
            dim,
            num_heads=4,
            qkv_bias=qkv_bias
        )

    def forward(self, x1, x2, SGSA_mask):

        x = x2 + self.cross_attention(self.norm1(x1), self.norm1(x2))

        res = x
        x_ = self.norm2(x)

        x = self.attn(x_)

        x = x + self.SGSA_attn(x_, SGSA_mask=SGSA_mask)

        x = res + x
        x = x + self.mlp(self.norm3(x))

        return x
