"""
Schema-driven clinical metadata encoding.

METADATA_SCHEMA is the single source of truth for every clinical field
the model can use. To add a field in the future:
  1. Add an entry below.
  2. Provide a column of that name with real values when fitting.
  3. Set active=True once real training data exists for it, and
     re-fit + re-fine-tune the fusion/EDL layers (not the Swin backbone).

Fields with active=False are documented extension points, not working
features — they exist so the architecture doesn't need to change when
real data for them becomes available. Being honest about this in the
report is better than claiming an untrained field works.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class FieldSpec:
    name: str
    kind: str  # "numeric" | "categorical"
    active: bool
    categories: Optional[List[str]] = None  # required for categorical
    note: str = ""


METADATA_SCHEMA: List[FieldSpec] = [
    FieldSpec("age", "numeric", active=True),
    FieldSpec(
        "sex", "categorical", active=True,
        categories=["male", "female", "unknown"],
    ),
    FieldSpec(
        "localization", "categorical", active=True,
        categories=[
            "back", "lower extremity", "trunk", "upper extremity", "abdomen",
            "face", "chest", "foot", "unknown", "neck", "scalp", "hand",
            "ear", "genital", "acral",
        ],
    ),
    # --- Extension points below: schema-ready, not yet trained ---
    FieldSpec(
        "skin_color", "categorical", active=False,
        categories=["type_I", "type_II", "type_III", "type_IV", "type_V", "type_VI"],
        note="Fitzpatrick scale. Estimable from the image itself; not yet "
             "trained into the fusion layer.",
    ),
    FieldSpec(
        "evolution", "categorical", active=False,
        categories=["stable", "changed_recently", "unknown"],
        note="No labeled training data available in HAM10000.",
    ),
    FieldSpec(
        "symptoms", "categorical", active=False,
        categories=["none", "itching", "bleeding", "pain", "other"],
        note="No labeled training data available in HAM10000.",
    ),
    FieldSpec(
        "previous_disease_history", "categorical", active=False,
        categories=["none", "prior_skin_cancer", "family_history", "other"],
        note="No labeled training data available in HAM10000.",
    ),
]


def active_fields() -> List[FieldSpec]:
    return [f for f in METADATA_SCHEMA if f.active]


class MetadataEncoder:
    """Fits scalers/encoders for every active schema field and turns
    a dict of raw values into the feature vector the fusion model
    expects. Field order is always the schema's active-field order,
    so this must be the same encoder (or a re-fit one with the same
    schema) used at both training and inference time."""

    def __init__(self, schema: List[FieldSpec] = None):
        self.schema = schema or METADATA_SCHEMA
        self._scalers = {}
        self._encoders = {}
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> "MetadataEncoder":
        self.feature_names = []
        for f in active_fields():
            if f.kind == "numeric":
                scaler = StandardScaler().fit(df[[f.name]])
                self._scalers[f.name] = scaler
                self.feature_names.append(f.name)
            else:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                enc.fit(df[[f.name]])
                self._encoders[f.name] = enc
                self.feature_names.extend(
                    f"{f.name}_{c}" for c in enc.categories_[0]
                )
        return self

    @property
    def output_dim(self) -> int:
        return len(self.feature_names)

    def transform(self, values: dict) -> np.ndarray:
        """values: dict with at least every active field's raw value,
        e.g. {"age": 45, "sex": "male", "localization": "back"}.
        Extra keys (inactive fields someone passes anyway) are ignored."""
        parts = []
        for f in active_fields():
            if f.kind == "numeric":
                scaler = self._scalers[f.name]
                parts.append(scaler.transform([[values[f.name]]])[0])
            else:
                enc = self._encoders[f.name]
                parts.append(enc.transform([[values[f.name]]])[0])
        return np.concatenate(parts).astype(np.float32)

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorized version for fitting/training pipelines over a
        whole dataframe at once, instead of row-by-row transform()."""
        parts = []
        for f in active_fields():
            if f.kind == "numeric":
                parts.append(self._scalers[f.name].transform(df[[f.name]]))
            else:
                parts.append(self._encoders[f.name].transform(df[[f.name]]))
        return np.concatenate(parts, axis=1).astype(np.float32)
