"""
Model architecture: Swin Transformer backbone + Cross-Attention Fusion
+ Evidential Deep Learning (EDL) head.

This file defines STRUCTURE only — no training logic lives here.
Training notebooks (see training/) import these same classes so the
architecture used to produce a checkpoint always matches the
architecture used to load it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_b


def build_swin_backbone(unfreeze_last_stage: bool = True) -> nn.Module:
    """Builds a Swin-V2-B backbone with its classification head removed
    (we only want the 1024-dim feature vector, not a 1000-class
    ImageNet prediction). By default only the last transformer stage
    is left trainable — this is the partial fine-tuning setup used
    throughout training, balancing accuracy against compute cost."""
    backbone = swin_v2_b(weights=None)
    backbone.head = nn.Identity()

    for _name, param in backbone.named_parameters():
        param.requires_grad = False

    if unfreeze_last_stage:
        for name, param in backbone.named_parameters():
            if "features.7" in name or "features.6" in name or "norm" in name:
                param.requires_grad = True

    return backbone


class CrossAttentionFusion(nn.Module):
    """Lets clinical metadata features attend over image features,
    rather than simply concatenating the two modalities. Metadata is
    the query; image features are the key/value — the model learns
    which visual regions are relevant given the patient context."""

    def __init__(self, image_dim: int, metadata_dim: int, fusion_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.meta_proj = nn.Linear(metadata_dim, fusion_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

    def forward(self, image_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        img = self.image_proj(image_features).unsqueeze(1)
        meta = self.meta_proj(metadata_features).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=meta, key=img, value=img)
        fused = self.norm1(meta + attn_out)
        fused = self.norm2(fused + self.ffn(fused))
        return fused.squeeze(1)


class EDLHead(nn.Module):
    """Outputs non-negative 'evidence' per class via Softplus, instead
    of a softmax probability distribution. evidence + 1 gives Dirichlet
    parameters (alpha), from which both a class probability AND an
    uncertainty estimate can be derived in a single forward pass."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc(x))


class MultimodalSkinModelEDL(nn.Module):
    """Full pipeline: image -> Swin features -> Cross-Attention Fusion
    with metadata -> EDL head -> evidence vector."""

    def __init__(self, swin_backbone: nn.Module, image_feature_dim: int,
                 metadata_dim: int, fusion_dim: int = 256, num_classes: int = 7):
        super().__init__()
        self.swin_backbone = swin_backbone
        self.fusion = CrossAttentionFusion(image_feature_dim, metadata_dim, fusion_dim)
        self.pre_edl = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.edl_head = EDLHead(fusion_dim // 2, num_classes)

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_features = self.swin_backbone(image)
        fused = self.fusion(image_features, metadata)
        x = self.pre_edl(fused)
        return self.edl_head(x)


def compute_uncertainty(evidence: torch.Tensor, num_classes: int):
    """Converts raw evidence into (class_probabilities, uncertainty).
    uncertainty = K / S, where K is the number of classes and S is the
    Dirichlet strength (total evidence + K). Higher S -> lower
    uncertainty -> the model has more 'evidence' backing its answer."""
    alpha = evidence + 1
    S = alpha.sum(dim=1, keepdim=True)
    prob = alpha / S
    uncertainty = num_classes / S
    return prob, uncertainty


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus (mole)",
    "vasc": "Vascular Lesion",
}
