"""Crop embeddings and zero-shot species guesses.

The same image embeddings feed the continuity merge (similarity) and the species guesses.
Text embeddings for the species list are cached in the project, so guesses can be
recomputed (after a split or merge) without loading the model again.
"""

from __future__ import annotations

import logging
from pathlib import Path
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
        self.weights_path = None
        # "squash" resizes the whole crop to 224x224, as Imageomics' pybioclip does; the
        # default centre crop would cut the head and tail off a long, thin fish.
        try:
            self.model, self.preprocess, self._weights = self._build_mapped(open_clip, torch)
        except Exception as exc:  # noqa: BLE001 — fall back to open_clip's own loader
            log.info("Loading BioCLIP 2 the standard way (%s)", exc)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.MODEL_ID, image_resize_mode="squash"
            )
        self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(self.MODEL_ID)
        self.templates = open_clip.OPENAI_IMAGENET_TEMPLATES
        self.dim = int(self.model.visual.output_dim)
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self.short_text = self._short_text_matches()

    def _build_mapped(self, open_clip, torch):  # pragma: no cover - needs weights
        """The same model and preprocessing, without private memory for the weights.

        open_clip's own load fills the model with random weights and then reads a second
        full copy of the 1.7 GB checkpoint, and PyTorch keeps the freed memory reserved: it
        needs ~3.5 GB of Windows' commit limit, which crashed a 16 GB PC with other programs
        open. Here the model is built on the meta device (nothing allocated) and its weights
        are views of the checkpoint file mapped read-only, which uses no commit at all.
        """
        from open_clip.pretrained import download_pretrained_from_hf  # noqa: PLC0415

        # open_clip warns that the model is "initialized randomly"; the weights come next.
        def quiet(record):
            return "initialized randomly" not in record.getMessage()

        logging.getLogger().addFilter(quiet)
        try:
            with torch.device("meta"):
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.MODEL_ID, image_resize_mode="squash", load_weights=False, device="meta"
                )
        finally:
            logging.getLogger().removeFilter(quiet)
        self.weights_path = download_pretrained_from_hf(self.MODEL_ID.removeprefix("hf-hub:"))
        weights, mapping = map_safetensors(self.weights_path)
        model.load_state_dict(weights, strict=True, assign=True)
        # The one tensor not in the checkpoint: the text tower's causal mask, built as in
        # open_clip's build_causal_mask (-inf above the diagonal).
        n = model.attn_mask.shape[0]
        model.attn_mask = torch.full((n, n), float("-inf")).triu_(1)
        left = [k for k, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
        if left:
            raise RuntimeError(f"not in the checkpoint: {', '.join(left)}")
        return model, preprocess, mapping

    def describe(self) -> dict:  # pragma: no cover - needs weights
        """For run records: the model, the exact weights revision and file, and the device."""
        from .provenance import fingerprint  # noqa: PLC0415

        path = Path(self.weights_path) if self.weights_path else None
        return {
            "name": self.name,
            "model": self.MODEL_ID,
            "revision": path.parent.name if path and path.parent.parent.name == "snapshots" else None,
            "weights": fingerprint(path) if path and path.exists() else None,
            "device": self.device,
            "input": "224x224 squash",
        }

    def _encode_short(self, tokens):  # pragma: no cover - needs weights
        """encode_text over the prompts' own length instead of the full 77-token context.
        The text transformer is causal and pools at the end-of-text token (the highest token
        id), so the padding after it can't change the result; short prompts run ~3x faster."""
        m = self.model
        n = int(tokens.argmax(dim=-1).max()) + 1
        tokens = tokens[:, :n]
        x = m.token_embedding(tokens) + m.positional_embedding[:n]
        x = m.ln_final(m.transformer(x, attn_mask=m.attn_mask[:n, :n]))
        return x[self.torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ m.text_projection

    def _short_text_matches(self) -> bool:  # pragma: no cover - needs weights
        """Use the fast path only if it reproduces open_clip's own encode_text here."""
        try:
            tokens = self.tokenizer(["a photo of a lionfish.", "a bad photo of the yellowfin goatfish."])
            with self.torch.no_grad():
                full = self.model.encode_text(tokens.to(self.device))
                short = self._encode_short(tokens.to(self.device))
            return bool(self.torch.allclose(full, short, rtol=1e-3, atol=1e-4))
        except Exception:  # noqa: BLE001 — a different model layout: use the standard path
            return False

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
            out = self._encode_short(tokens) if self.short_text else self.model.encode_text(tokens)
            return out.float().cpu().numpy()

    def text_embeddings(self, labels: list[str]) -> np.ndarray | None:  # pragma: no cover
        if not labels:
            return None
        return prompt_ensemble(self._encode_text, labels, self.templates)


class BioClip25Classifier(BioClip2Classifier):
    """BioCLIP 2.5 Huge (February 2026): a ViT-H/14 image encoder, about twice BioCLIP 2's
    ViT-L/14, trained on TreeOfLife-200M plus 19M more images. Imageomics report 61.3 against
    55.6 for BioCLIP 2 on their zero-shot species benchmarks (none of them underwater video).
    Loaded the same way: its 3.9 GB checkpoint is mapped, not copied, so it needs disk cache
    and time rather than free memory. modeltest.py compares the two on reviewed cards."""

    name = "bioclip-2.5"
    MODEL_ID = "hf-hub:imageomics/bioclip-2.5-vith14"


def map_safetensors(path) -> tuple[dict, object]:
    """Tensors of a .safetensors file as read-only views of the file mapped into memory.

    A read-only file mapping isn't charged against Windows' commit limit (the pages are
    re-read from the file when needed), so a large checkpoint fits where a copy wouldn't.
    The views must never be written to. Returns (tensors, mapping); keep the mapping alive
    as long as the tensors are used.
    """
    import json  # noqa: PLC0415
    import mmap  # noqa: PLC0415
    import struct  # noqa: PLC0415
    import warnings  # noqa: PLC0415

    import torch  # noqa: PLC0415

    dtypes = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
              "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8, "BOOL": torch.bool}
    with open(path, "rb") as f:
        mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    (size,) = struct.unpack("<Q", mapping[:8])
    header = json.loads(mapping[8 : 8 + size])
    header.pop("__metadata__", None)
    tensors = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        for key, meta in header.items():
            begin, end = meta["data_offsets"]
            dtype = dtypes[meta["dtype"]]
            flat = torch.frombuffer(mapping, dtype=dtype, count=(end - begin) // dtype.itemsize, offset=8 + size + begin)
            tensors[key] = flat.reshape(meta["shape"])
    return tensors, mapping


def prompt_ensemble(encode, labels: list[str], templates) -> np.ndarray:
    """One embedding per label: the renormalised mean over every prompt template.

    This is how pybioclip builds class embeddings from CLIP's 80 OpenAI ImageNet templates
    (BioCLIP 2's own evaluation uses the same 80 plus "a photo of {c}."), rather than a
    single "a photo of ..." prompt.
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
    if settings.classifier in ("bioclip", "bioclip-2.5"):
        try:
            return (BioClip25Classifier if settings.classifier == "bioclip-2.5" else BioClip2Classifier)(device=device)
        except Exception as exc:  # noqa: BLE001 — any load failure means "use the fallback"
            hint = ""
            if isinstance(exc, MemoryError) or "paging file" in str(exc) or "not enough memory" in str(exc):
                hint = " Close other programs to free memory, then run again to get species guesses."
            log.warning("BioCLIP 2 unavailable (%s); using colour histograms, no guesses.%s", exc, hint)
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
