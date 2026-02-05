from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _to_sequence(feats: torch.Tensor) -> torch.Tensor:
    if feats.ndim == 4:
        # (B, C, H, W) -> (B, HW, C)
        b, c, h, w = feats.shape
        return feats.reshape(b, c, h * w).transpose(1, 2)
    if feats.ndim == 3:
        # (B, N, C)
        return feats
    if feats.ndim == 2:
        # (B, C) -> (B, 1, C)
        return feats.unsqueeze(1)
    raise ValueError(f"Unsupported feature shape: {tuple(feats.shape)}")


def _pool_features(feats: torch.Tensor) -> torch.Tensor:
    if feats.ndim == 4:
        return feats.mean(dim=(2, 3))
    if feats.ndim == 3:
        return feats.mean(dim=1)
    if feats.ndim == 2:
        return feats
    raise ValueError(f"Unsupported feature shape: {tuple(feats.shape)}")


class BackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.num_features = getattr(backbone, "num_features", None)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            out = self.backbone.forward_features(x)
        else:
            out = self.backbone(x)

        if isinstance(out, dict):
            for key in ("x_norm_patchtokens", "x_norm_clstoken", "x", "last_hidden_state"):
                if key in out:
                    return out[key]
            # Fall back to first tensor-like value
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise RuntimeError("Backbone returned dict without tensor features.")

        return out


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = _pool_features(feats)
        x = self.dropout(x)
        return self.fc(x)


class TotalScoreHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = _pool_features(feats)
        x = self.dropout(x)
        return self.fc(x)


class MLDecoderHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        d_model: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model) if d_model is not None else int(in_dim)
        self.num_classes = int(num_classes)
        if self.d_model % int(num_heads) != 0:
            raise ValueError(f"ml_decoder_dim ({self.d_model}) must be divisible by num_heads ({num_heads}).")

        self.kv_proj = nn.Identity() if in_dim == self.d_model else nn.Linear(in_dim, self.d_model)
        self.query_embed = nn.Parameter(torch.randn(self.num_classes, self.d_model) * 0.02)
        self.attn = nn.MultiheadAttention(self.d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        seq = _to_sequence(feats)
        kv = self.kv_proj(seq)
        b = kv.shape[0]
        q = self.query_embed.unsqueeze(0).expand(b, -1, -1)

        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        attn_out = self.norm(attn_out)
        logits = self.classifier(attn_out).squeeze(-1)
        return logits


class MultiTaskModel(nn.Module):
    def __init__(self, backbone: BackboneWrapper, head: nn.Module, aux_head: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.aux_head = aux_head

    def forward(self, x: torch.Tensor):
        feats = self.backbone.forward_features(x)
        logits = self.head(feats)
        if self.aux_head is None:
            return logits
        aux = self.aux_head(feats)
        return {"logits": logits, "aux": aux}

    def get_classifier(self):
        return self.head
