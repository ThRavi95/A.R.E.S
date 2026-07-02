from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class SequenceAttentionXAI:
    def __init__(self, model):
        self.model = model
        self.attention_matrix: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._original_attn_forward = None
        self._register_hook()

    def _register_hook(self):
        attn_module = self.model.encoder.layers[-1].self_attn

        if self._hook_handle is not None:
            self._hook_handle.remove()

        self._original_attn_forward = attn_module.forward

        def _forward_with_weights(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return self._original_attn_forward(*args, **kwargs)

        def _capture_attention(module, inputs, output):
            del module, inputs
            if isinstance(output, tuple) and len(output) > 1:
                self.attention_matrix = output[1]
            else:
                self.attention_matrix = None

        self._hook_handle = attn_module.register_forward_hook(_capture_attention)
        attn_module.forward = _forward_with_weights

    def explain_sequence(self, input_tensor):
        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                # input_tensor: (batch, seq_len)
                self.model.encode(input_tensor)

            if self.attention_matrix is None:
                raise RuntimeError("Attention weights were not captured during encode().")

            attention_matrix = self.attention_matrix.detach().cpu()

            # Multi-head weights are typically (batch, heads, tgt_len, src_len).
            if attention_matrix.dim() == 4:
                attention_matrix = attention_matrix.mean(dim=1)  # (batch, tgt_len, src_len)

            # Reduce any remaining batch dimension to a single 2D map.
            if attention_matrix.dim() == 3:
                attention_matrix = (
                    attention_matrix.squeeze(0)
                    if attention_matrix.size(0) == 1
                    else attention_matrix.mean(dim=0)
                )

            if attention_matrix.dim() != 2:
                raise RuntimeError(
                    f"Expected a 2D attention map, got tensor with shape {tuple(attention_matrix.shape)}."
                )

            return attention_matrix.numpy()
        finally:
            self.attention_matrix = None
            if was_training:
                self.model.train()

    def __del__(self):
        try:
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None
        except Exception:
            pass

        try:
            if self._original_attn_forward is not None:
                self.model.encoder.layers[-1].self_attn.forward = self._original_attn_forward
                self._original_attn_forward = None
        except Exception:
            pass