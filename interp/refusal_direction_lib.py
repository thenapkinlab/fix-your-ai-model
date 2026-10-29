"""Shared helper: residual stream at the final prompt token (mlx, Qwen-style models)."""
import numpy as np
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

def last_token_acts(model, tok, prompts, layers=None):
    """(n_prompts, n_layers_kept, d). layers=None keeps every layer; stops early after the last one asked for."""
    stop = max(layers) if layers else len(model.model.layers) - 1
    out = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        h = model.model.embed_tokens(mx.array([ids]))
        mask = create_attention_mask(h, None)
        keep = []
        for i, layer in enumerate(model.model.layers[:stop + 1]):
            h = layer(h, mask, None)
            if layers is None or i in layers: keep.append(h[0, -1])
        out.append(np.array(mx.stack(keep).astype(mx.float32)))
    return np.stack(out)
