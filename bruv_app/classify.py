"""Crop embeddings and zero-shot species guesses.

The same image embeddings feed the continuity merge (similarity) and the species guesses.
Text embeddings for the species list are cached in the project, so guesses can be
recomputed (after a split or merge) without loading the model again.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def zero_shot_topk(
    image_emb: np.ndarray, text_emb: np.ndarray, labels: list[str], scale: float, k: int = 3
) -> list[tuple[str, float]]:
    """Softmax over the label set; returns the top-k (label, probability)."""
    probs = softmax(scale * (text_emb @ l2_normalize(image_emb)))
    top = np.argsort(-probs)[:k]
    return [(labels[i], float(probs[i])) for i in top]


class Classifier(Protocol):
    name: str
    logit_scale: float

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """RGB crops in, (N, D) L2-normalised embeddings out."""
        ...

    def text_embeddings(self, labels: list[str]) -> np.ndarray | None:
        """(L, D) L2-normalised label embeddings, or None if this model can't guess."""
        ...


def histogram_embed(crops: list[np.ndarray], bins: int = 8) -> np.ndarray:
    from PIL import Image  # noqa: PLC0415

    out = []
    for crop in crops:
        hsv = np.asarray(Image.fromarray(crop).convert("HSV"), dtype=np.float32)
        # Weight by saturation so washed-out water pixels count less.
        weights = hsv[..., 1].ravel() / 255.0 + 1e-3
        hist, _, _ = np.histogram2d(
            hsv[..., 0].ravel(),
            hsv[..., 2].ravel(),
            bins=bins,
            range=[[0, 256], [0, 256]],
            weights=weights,
        )
        out.append(np.sqrt(hist.ravel()))
    return l2_normalize(np.array(out)) if out else np.zeros((0, bins**2), np.float32)


class HistogramClassifier:
    """Fallback when no model is available: colour histograms, no species guesses."""

    name = "hsv-histogram"
    logit_scale = 1.0

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        return histogram_embed(crops)

    def text_embeddings(self, labels: list[str]) -> np.ndarray | None:
        return None


class BioClip2Classifier:
    """BioCLIP 2 zero-shot via open_clip, restricted to the template's species list."""

    name = "bioclip-2"
    MODEL_ID = "hf-hub:imageomics/bioclip-2"

    def __init__(self, device: str = "cpu"):
        try:
            import open_clip  # noqa: PLC0415 — optional heavy dependency
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "open_clip/torch not installed. Run: pip install -r requirements-ml.txt"
            ) from exc
        self.torch = torch
        self.device = device
        # "squash" resizes the whole crop to 224x224, as Imageomics' pybioclip does; the
        # default centre crop would cut the head and tail off a long, thin fish.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.MODEL_ID, image_resize_mode="squash"
        )
        self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_ID)
        self.templates = open_clip.OPENAI_IMAGENET_TEMPLATES
        self.dim = int(self.model.visual.output_dim)
        self.logit_scale = float(self.model.logit_scale.exp().item())

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:  # pragma: no cover - needs weights
        from PIL import Image  # noqa: PLC0415

        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(crops), 16):
                batch = torch.stack(
                    [self.preprocess(Image.fromarray(c)) for c in crops[i : i + 16]]
                ).to(self.device)
                out.append(self.model.encode_image(batch).float().cpu().numpy())
        return l2_normalize(np.concatenate(out)) if out else np.zeros((0, self.dim), np.float32)

    def _encode_text(self, prompts: list[str]) -> np.ndarray:  # pragma: no cover - needs weights
        with self.torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            return self.model.encode_text(tokens).float().cpu().numpy()

    def text_embeddings(self, labels: list[str]) -> np.ndarray | None:  # pragma: no cover
        if not labels:
            return None
        return prompt_ensemble(self._encode_text, labels, self.templates)


def prompt_ensemble(encode, labels: list[str], templates) -> np.ndarray:
    """One embedding per label: the renormalised mean over every prompt template.

    This is how BioCLIP 2's zero-shot evaluation and pybioclip build class embeddings
    (CLIP's 80 OpenAI ImageNet templates), rather than a single "a photo of ..." prompt.
    """
    out = []
    for label in labels:
        emb = l2_normalize(encode([t(label) for t in templates]))
        out.append(l2_normalize(emb.mean(axis=0)))
    return np.stack(out)


def make_classifier(settings, device: str) -> Classifier:
    """Build the configured classifier, falling back to histograms if BioCLIP can't load."""
    if settings.classifier == "histogram":
        return HistogramClassifier()
    if settings.classifier == "bioclip":
        try:
            return BioClip2Classifier(device=device)
        except Exception as exc:  # noqa: BLE001 — any load failure means "use the fallback"
            log.warning("BioCLIP 2 unavailable (%s); using colour histograms, no guesses.", exc)
            return HistogramClassifier()
    raise ValueError(f"Unknown classifier: {settings.classifier}")


def crop_quality(crop: np.ndarray, score: float, area_frac: float) -> float:
    """Rank crops for classification: detection confidence x size x sharpness."""
    gray = crop.astype(np.float32).mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
    if min(gray.shape) < 3:
        return 0.0
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    sharpness = float(np.log1p(lap.var()))
    return float(score) * float(np.sqrt(area_frac)) * sharpness
