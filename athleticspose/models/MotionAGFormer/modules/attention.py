"""Attention module from TaatiTeam/MotionAGFormer repository (Apache-2.0 License)."""

from torch import nn


class Attention(nn.Module):
    """Attention module."""

    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        mode="spatial",
    ):
        """Initialize Attention module."""
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """Forward pass of Attention module."""
        b, t, j, c = x.shape

        qkv = (
            self.qkv(x).reshape(b, t, j, 3, self.num_heads, c // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        )  # (3, B, H, T, J, C)
        if self.mode == "temporal":
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v)
        elif self.mode == "spatial":
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_spatial(self, q, k, v):
        """Forward pass of spatial attention."""
        b, h, t, j, c = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(b, t, j, c * self.num_heads)
        return x  # (B, T, J, C)

    def forward_temporal(self, q, k, v):
        """Forward pass of temporal attention."""
        b, h, t, j, c = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(b, t, j, c * self.num_heads)
        return x  # (B, T, J, C)
