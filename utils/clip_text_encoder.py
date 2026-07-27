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

Key naming (IMPORTANT for load_state_dict compatibility)
-------------------------------------------------------
The encoder is stored as `self.model` and `self.proj` inside TextEncoder,
so when a parent module holds it as `self.text_encoder = TextEncoder(...)`,
the state dict keys are:

    text_encoder.model.XXX   ← transformer model weights
    text_encoder.proj.X.X    ← projection head weights

Old checkpoints (saved with direct AutoModel) use:
    text_encoder.XXX         ← same keys, no wrapper
    text_resizer.X.X         ← projection head (now absorbed into text_encoder.proj)

New checkpoints use:
    text_encoder.model.XXX   ← same as above
    text_encoder.proj.X.X    ← projection head
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


def remap_text_encoder_keys(state_dict: dict) -> dict:
    """
    Map checkpoint keys from legacy formats to the current TextEncoder wrapper format.

    Three legacy key formats are supported:

    1. Pre-TextEncoder (direct AutoModel, no wrapper):
       text_encoder.XXX              -> text_encoder.model.XXX
       text_resizer.0.weight         -> text_encoder.proj.0.weight
       text_resizer.1.weight         -> text_encoder.proj.1.weight

    2. TextEncoder with private attribute names (lazy _model / _proj):
       text_encoder._model.XXX       -> text_encoder.model.XXX
       text_encoder._proj.0.weight   -> text_encoder.proj.0.weight

    3. TextEncoder with wrapper prefix already present (no change needed):
       text_encoder.model.XXX        -> text_encoder.model.XXX  (passthrough)
       text_encoder.proj.X.X         -> text_encoder.proj.X.X   (passthrough)

    This function should be called on the checkpoint state dict BEFORE
    passing it to model.load_state_dict(), so that the keys match.

    Args:
        state_dict: raw checkpoint state dict (may have legacy key format)

    Returns:
        remapped state dict (keys normalized to current format)
    """
    remapped = {}
    for k, v in state_dict.items():
        new_k = k

        # Handle text_resizer -> text_encoder.proj (format 1)
        if k.startswith("text_resizer."):
            new_k = k.replace("text_resizer.", "text_encoder.proj.", 1)

        # Handle text_encoder._model -> text_encoder.model (format 2)
        elif k.startswith("text_encoder._model."):
            new_k = k.replace("text_encoder._model.", "text_encoder.model.", 1)

        # Handle text_encoder._proj -> text_encoder.proj (format 2)
        elif k.startswith("text_encoder._proj."):
            new_k = k.replace("text_encoder._proj.", "text_encoder.proj.", 1)

        # Handle text_encoder.XXX (no wrapper, no underscore) -> text_encoder.model.XXX (format 1)
        elif k.startswith("text_encoder.") and not k.startswith("text_encoder.model.") and not k.startswith("text_encoder.proj."):
            new_k = k.replace("text_encoder.", "text_encoder.model.", 1)

        remapped[new_k] = v
    return remapped


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
    CLIP text encoder with output projection.

    The CLIP model hidden dimension is auto-detected from the model config,
    so this works for any CLIP variant (ViT-B/32 -> 512, ViT-L/14 -> 768, ...).

    The model and projection head are stored as nn.Module attributes
    (self.model, self.proj) so they are properly registered in state_dict
    and can be loaded from checkpoints.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 emb_dim: int = 512, n_groups: int = 77, freeze: bool = True):
        super().__init__(model_name, emb_dim, n_groups, freeze)

        # Store as nn.Module attributes (not lazy) so state_dict keys are
        # available for load_state_dict BEFORE any forward pass.
        self.model = CLIPTextModel.from_pretrained(model_name)
        clip_dim = self.model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, emb_dim, bias=True),
            nn.LayerNorm(emb_dim, eps=1e-12),
        )

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def set_freeze(self, mode: bool):
        self.freeze = mode
        for p in self.model.parameters():
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

        self.model = self.model.to(device)
        with torch.no_grad() if self.freeze else torch.enable_grad():
            out = self.model(input_ids=input_ids)
        hidden = out.last_hidden_state                   # [B, n_groups, D]

        proj = self.proj(hidden.to(device))             # [B, n_groups, C]
        return proj, mask

    @property
    def tokenizer(self):
        """Lazy-init tokenizer (not stored in state_dict, safe to lazy)."""
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        return self._tokenizer


# ---------------------------------------------------------------------------
# RoBERTa text encoder (AutoModel + AutoTokenizer)
# ---------------------------------------------------------------------------
@register_encoder("roberta")
class RobertaTextEncoder(TextEncoder):
    """
    RoBERTa text encoder backed by AutoModel + AutoTokenizer.

    Mirrors the original Branch2D implementation exactly:
    AutoModel.from_pretrained + AutoTokenizer.from_pretrained + Linear projection.

    The model and projection head are stored as nn.Module attributes
    (self.model, self.proj) so they are properly registered in state_dict
    and can be loaded from checkpoints.
    """

    def __init__(self, model_name: str, emb_dim: int, n_groups: int, freeze: bool = True):
        super().__init__(model_name, emb_dim, n_groups, freeze)

        # Store as nn.Module attributes (not lazy) so state_dict keys are
        # available for load_state_dict BEFORE any forward pass.
        self.model = AutoModel.from_pretrained(model_name)
        roberta_dim = self.model.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(roberta_dim, emb_dim, bias=True),
            nn.LayerNorm(emb_dim, eps=1e-12),
        )

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def set_freeze(self, mode: bool):
        self.freeze = mode
        for p in self.model.parameters():
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

        self.model = self.model.to(device)
        with torch.inference_mode(mode=self.freeze):
            out = self.model(**tokens)
        hidden = out.last_hidden_state                  # [B, n_groups, D]

        proj = self.proj(hidden.to(device))            # [B, n_groups, C]
        attn_mask = tokens.attention_mask.bool()        # [B, n_groups]
        return proj, attn_mask

    @property
    def tokenizer(self):
        """Lazy-init tokenizer (not stored in state_dict, safe to lazy)."""
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
