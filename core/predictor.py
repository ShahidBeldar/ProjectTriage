"""
SkinLesionPredictor: the one object the API layer talks to.

It owns model loading, preprocessing, inference, and wiring the result
through quality checks and risk triage. The API routes never touch
torch, the model, or the metadata encoder directly — only this class
does, which is what makes swapping in a better model later a one-line
config change instead of a rewrite.
"""

import io
import pickle
from datetime import datetime, timezone

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from core.env import resolve_path, get_device
from core.architecture import (
    MultimodalSkinModelEDL,
    build_swin_backbone,
    compute_uncertainty,
    CLASS_NAMES,
    CLASS_FULL_NAMES,
)
from core.metadata_schema import MetadataEncoder
from core.quality_check import check_image_quality
from core.risk_triage import triage

IMAGE_FEATURE_DIM = 1024
FUSION_DIM = 256
NUM_CLASSES = 7

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SkinLesionPredictor:
    def __init__(self, model_filename: str = "multimodal_edl_model_weighted.pth",
                 encoder_filename: str = "metadata_encoder.pkl"):
        self.device = get_device()
        self.model_filename = model_filename
        self.encoder_filename = encoder_filename
        self.model = None
        self.metadata_encoder: MetadataEncoder = None
        self.model_metadata = {}
        self.load()

    def load(self):
        encoder_path = resolve_path(self.encoder_filename)
        with open(encoder_path, "rb") as f:
            self.metadata_encoder = pickle.load(f)

        backbone = build_swin_backbone(unfreeze_last_stage=True)
        model = MultimodalSkinModelEDL(
            swin_backbone=backbone,
            image_feature_dim=IMAGE_FEATURE_DIM,
            metadata_dim=self.metadata_encoder.output_dim,
            fusion_dim=FUSION_DIM,
            num_classes=NUM_CLASSES,
        ).to(self.device)

        model_path = resolve_path(self.model_filename)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Consistency check: the encoder must produce the same features, in the same
        # order, as the ones the checkpoint was trained on. Warn-only so the API still starts.
        ck_cols = checkpoint.get("metadata_feature_cols")
        if ck_cols is not None:
            norm = lambda n: n.replace("age_scaled", "age").replace("loc_", "localization_")
            if [norm(c) for c in ck_cols] != list(self.metadata_encoder.feature_names):
                import warnings
                warnings.warn("Metadata encoder features differ from the checkpoint's "
                              "metadata_feature_cols; predictions may be unreliable.")
        model.eval()

        self.model = model
        self.model_metadata = {
            "model_file": self.model_filename,
            "val_accuracy": checkpoint.get("best_val_acc"),
            "mel_recall": checkpoint.get("mel_recall"),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }

    def reload(self, model_filename: str = None):
        """Swap to a different checkpoint without restarting the process."""
        if model_filename:
            self.model_filename = model_filename
        self.load()

    def predict(self, image_bytes: bytes, age: float, sex: str, localization: str) -> dict:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        quality = check_image_quality(pil_image)

        image_tensor = VAL_TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        metadata_vec = self.metadata_encoder.transform(
            {"age": age, "sex": sex, "localization": localization}
        )
        metadata_tensor = torch.tensor(metadata_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            evidence = self.model(image_tensor, metadata_tensor)
            prob, uncertainty = compute_uncertainty(evidence, NUM_CLASSES)

        prob_np = prob.cpu().numpy()[0]
        uncertainty_val = uncertainty.item()
        pred_idx = int(prob_np.argmax())
        pred_class = CLASS_NAMES[pred_idx]

        result = triage(pred_class, uncertainty_val, quality_passed=quality.passed)

        return {
            "predicted_class": pred_class,
            "predicted_class_full_name": CLASS_FULL_NAMES[pred_class],
            "confidence": round(float(prob_np[pred_idx]), 4),
            "uncertainty": round(uncertainty_val, 4),
            "risk_level": result.level,
            "risk_label": result.label,
            "risk_reason": result.reason,
            "class_probabilities": {
                CLASS_NAMES[i]: round(float(prob_np[i]), 4) for i in range(NUM_CLASSES)
            },
            "image_quality": {
                "passed": quality.passed,
                "issues": quality.issues,
            },
        }
