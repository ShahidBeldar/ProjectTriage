"""
Image quality pre-screening, run before inference.

Catches obviously unusable uploads (too small, too blurry, too dark
or blown out) so the system can flag them rather than quietly
returning a prediction that was never reliable to begin with.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np
from PIL import Image

MIN_RESOLUTION = 100          # pixels, shorter side
BLUR_VARIANCE_THRESHOLD = 50.0   # below this, image is likely too blurry
DARK_MEAN_THRESHOLD = 25.0       # mean pixel value below this = underexposed
BRIGHT_MEAN_THRESHOLD = 230.0    # mean pixel value above this = overexposed


@dataclass
class QualityResult:
    passed: bool
    issues: List[str] = field(default_factory=list)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian is a standard, dependency-light blur
    metric: sharp images have high-frequency edges (high variance),
    blurry images don't."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    lap = np.zeros_like(gray, dtype=np.float32)
    # Simple valid-region convolution — no external CV library required.
    for i in range(1, h - 1):
        row = gray[i - 1:i + 2, :]
        lap[i, 1:w - 1] = (
            row[0, 1:w - 1] + row[2, 1:w - 1]
            + gray[i, 0:w - 2] + gray[i, 2:w]
            - 4 * gray[i, 1:w - 1]
        )
    return float(lap.var())


def check_image_quality(image: Image.Image) -> QualityResult:
    issues = []

    width, height = image.size
    if min(width, height) < MIN_RESOLUTION:
        issues.append(f"Resolution too low ({width}x{height}); minimum is {MIN_RESOLUTION}px.")

    gray = np.array(image.convert("L"), dtype=np.float32)

    mean_brightness = float(gray.mean())
    if mean_brightness < DARK_MEAN_THRESHOLD:
        issues.append("Image appears underexposed (too dark).")
    elif mean_brightness > BRIGHT_MEAN_THRESHOLD:
        issues.append("Image appears overexposed (too bright).")

    # Blur check on a capped-size version for speed, not full resolution
    small = image.convert("L")
    small.thumbnail((256, 256))
    blur_var = _laplacian_variance(np.array(small, dtype=np.float32))
    if blur_var < BLUR_VARIANCE_THRESHOLD:
        issues.append(f"Image appears blurry (sharpness score {blur_var:.1f}).")

    return QualityResult(passed=len(issues) == 0, issues=issues)
