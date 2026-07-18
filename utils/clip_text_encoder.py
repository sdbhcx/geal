"""
Text encoders — configurable via YAML.

Architecture
------------
TextEncoder              # abstract base (nn.Module)
    ClipTextEncoder      # CLIP text encoder
    RobertaTextEncoder   # RoBERTa (AutoModel + AutoTokenizer)
    (future: other encoder types, ...)

Interface
---------
Every encoder implements:
    encode(text_input, device) -> (proj, mask)
        text_input: str or list[str]
        proj:       [*, C]   projected features (C = emb_dim)
        mask:       [*, L]   boolean attention mask
    set_freeze(mode: bool)       # toggle requires_grad at runtime
    encode_single(sentence, device) -> (proj, mask)

Factory
-------
build_text_encoder(cfg) -> TextEncoder
    cfg is a dict with at minimum:
        type: "clip"   (which encoder class to instantiate)
        model_name: str  # HuggingFace model id / local path
        emb_dim: int     # target projection dimension
        n_groups: int    # token sequence length
        freeze: bool     # freeze weights on init

Add a new encoder by:
    1. Subclass TextEncoder in this file
    2. Register it in ENCODER_REGISTRY below
    3. Use type: "<name>" in the config
"""
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
ENCODER_REGISTRY = {}  # name -> class


def register_encoder(name: str):
    """Decorator to register a TextEncoder subclass."""
    def wrapper(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return wrapper


def build_text_encoder(cfg: dict) -> "TextEncoder":
    """Create a TextEncoder from a config dict."""
    enc_type = cfg.get("type", "clip")
    cls = ENCODER_REGISTRY.get(enc_type)
    if cls is None:
        available = ", ".join(sorted(ENCODER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown text_encoder_type '{enc_type}'. Available: {available}"
        )
    # Pass only the keys the constructor expects.
    kwargs = {
        "model_name": cfg.get("model_name", "openai/clip-vit-large-patch14"),
        "emb_dim": cfg.get("emb_dim", 512),
        "n_groups": cfg.get("n_groups", 77),
        "freeze": cfg.get("freeze", True),
    }
    # Allow extra encoder-specific keys to pass through.
    for k, v in cfg.items():
        if k not in ("type",):
            kwargs.setdefault(k, v)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class TextEncoder(nn.Module):
    """Abstract base for all text encoders."""

    def __init__(self, model_name: str, emb_dim: int, n_groups: int, freeze: bool):
        super().__init__()
        self.model_name = model_name
        self.emb_dim = emb_dim
        self.n_groups = n_groups
        self.freeze = freeze

    def encode(self, text_input, device=None):
        raise NotImplementedError

    def set_freeze(self, mode: bool):
        raise NotImplementedError

    def encode_single(self, sentence: str, device=None):
        """Convenience: encode one sentence."""
        return self.encode(sentence, device)


# ---------------------------------------------------------------------------
# CLIP text encoder
# ---------------------------------------------------------------------------
@register_encoder("clip")
class ClipTextEncoder(TextEncoder):
    """
    CLIP text encoder with lazy loading and output projection.

    The CLIP model hidden dimension is auto-detected on first use,
    so this works for any CLIP variant (ViT-B/32 -> 512, ViT-L/14 -> 768, ...).
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 emb_dim: int = 512, n_groups: int = 77, freeze: bool = True):
        super().__init__(model_name, emb_dim, n_groups, freeze)

        # Lazy init so __init__ works on CPU without downloading weights.
        self._model: CLIPTextModel | None = None
        self._tokenizer: CLIPTokenizer | None = None

    @property
    def model(self) -> CLIPTextModel:
        if self._model is None:
            self._model = CLIPTextModel.from_pretrained(self.model_name)
        return self._model

    @property
    def tokenizer(self) -> CLIPTokenizer:
        if self._tokenizer is None:
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def set_freeze(self, mode: bool):
        self.freeze = mode
        if self._model is not None:
            for p in self._model.parameters():
                p.requires_grad = not mode

    def encode(self, text_input, device=None):
        """Tokenize -> CLIP forward -> project to emb_dim -> return (proj, mask)."""
        if device is None:
            device = next(self.model.parameters()).device

        tok = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="max_length",
            max_length=self.n_groups,
            truncation=True,
            return_attention_mask=True,
        )
        input_ids = tok["input_ids"].to(device)          # [B, n_groups]
        # CLIP uses id 49406 as padding (the [BOS] token duplicated)
        mask = input_ids.ne(49406).bool()                # [B, n_groups]

        self._model = self.model.to(device)
        with torch.no_grad() if self.freeze else torch.enable_grad():
            out = self.model(input_ids=input_ids)
        hidden = out.last_hidden_state                   # [B, n_groups, D]

        proj = self._project(hidden, device)             # [B, n_groups, C]
        return proj, mask

    def _project(self, hidden: torch.Tensor, device=None) -> torch.Tensor:
        if not hasattr(self, "_proj"):
            # Auto-detect CLIP hidden dimension from the model itself.
            clip_dim = self._model.config.hidden_size
            self._proj = nn.Sequential(
                nn.Linear(clip_dim, self.emb_dim, bias=True),
                nn.LayerNorm(self.emb_dim, eps=1e-12),
            ).to(device)
        return self._proj(hidden.to(device))


# ---------------------------------------------------------------------------
# RoBERTa text encoder (AutoModel + AutoTokenizer)
# ---------------------------------------------------------------------------
@register_encoder("roberta")
class RobertaTextEncoder(TextEncoder):
    """
    RoBERTa text encoder backed by AutoModel + AutoTokenizer.

    Mirrors the original Branch2D implementation exactly:
    AutoModel.from_pretrained + AutoTokenizer.from_pretrained + Linear projection.
    """

    def __init__(self, model_name: str, emb_dim: int, n_groups: int, freeze: bool = True):
        super().__init__(model_name, emb_dim, n_groups, freeze)
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_name)
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def set_freeze(self, mode: bool):
        self.freeze = mode
        if self._model is not None:
            for p in self._model.parameters():
                p.requires_grad = not mode

    def encode(self, text_input, device=None):
        """Tokenize -> RoBERTa forward -> project to emb_dim -> return (proj, mask)."""
        if device is None:
            device = next(self.model.parameters()).device

        tokens = self.tokenizer.batch_encode_plus(
            text_input,
            padding="max_length",
            truncation=True,
            max_length=self.n_groups,
            return_tensors="pt",
        ).to(device)

        self._model = self.model.to(device)
        with torch.inference_mode(mode=self.freeze):
            out = self.model(**tokens)
        hidden = out.last_hidden_state                  # [B, n_groups, D]

        proj = self._project(hidden, device)            # [B, n_groups, C]
        attn_mask = tokens.attention_mask.bool()        # [B, n_groups]
        return proj, attn_mask

    def _project(self, hidden: torch.Tensor, device=None) -> torch.Tensor:
        if not hasattr(self, "_proj"):
            roberta_dim = self._model.config.hidden_size
            self._proj = nn.Sequential(
                nn.Linear(roberta_dim, self.emb_dim, bias=True),
                nn.LayerNorm(self.emb_dim, eps=1e-12),
            ).to(device)
        return self._proj(hidden.to(device))
