#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared seq2seq supervision/forward protocol helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .inference_protocol import DEFAULT_ONESHOT_MODELS, is_oneshot_model


def align_source_position_scale(
    src,
    input_mean,
    input_scale,
    output_mean,
    output_scale,
    output_dim: int = 3,
):
    """
    Convert source position channels from input-scaler space into target-scaler space.

    Autoformer/FEDformer-style decoder context assumes that historical labels and
    supervised future labels live in the same normalized space. HGV source tensors
    contain extra kinematic channels, so only the first output_dim position channels
    are converted; the remaining source features are preserved.
    """
    output_dim = max(1, int(output_dim))
    if isinstance(src, torch.Tensor):
        converted = src.clone()
        in_mean = torch.as_tensor(input_mean, device=src.device, dtype=src.dtype)[:output_dim]
        in_scale = torch.as_tensor(input_scale, device=src.device, dtype=src.dtype)[:output_dim]
        out_mean = torch.as_tensor(output_mean, device=src.device, dtype=src.dtype)[:output_dim]
        out_scale = torch.as_tensor(output_scale, device=src.device, dtype=src.dtype)[:output_dim]
        raw_positions = converted[..., :output_dim] * in_scale + in_mean
        converted[..., :output_dim] = (raw_positions - out_mean) / out_scale
        return converted

    import numpy as np

    src_array = np.asarray(src)
    converted = np.array(src_array, copy=True)
    in_mean = np.asarray(input_mean, dtype=converted.dtype)[:output_dim]
    in_scale = np.asarray(input_scale, dtype=converted.dtype)[:output_dim]
    out_mean = np.asarray(output_mean, dtype=converted.dtype)[:output_dim]
    out_scale = np.asarray(output_scale, dtype=converted.dtype)[:output_dim]
    raw_positions = converted[..., :output_dim] * in_scale + in_mean
    converted[..., :output_dim] = (raw_positions - out_mean) / out_scale
    return converted


def build_supervision_windows(
    tgt: torch.Tensor,
    protocol: str,
    pred_len: int | None = None,
    label_len: int | None = None,
    src: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (decoder_input, supervision_target) under a unified protocol."""
    p = str(protocol).lower().strip()
    if p == "source_context_pred_window":
        if src is None:
            raise ValueError("source_context_pred_window requires src")
        use_pred_len = int(pred_len) if pred_len is not None else int(tgt.size(1))
        use_pred_len = max(1, min(use_pred_len, tgt.size(1)))
        use_label_len = int(label_len) if label_len is not None else max(1, min(48, src.size(1)))
        use_label_len = max(1, min(use_label_len, src.size(1)))
        target = tgt[:, :use_pred_len, :]
        context = src[:, -use_label_len:, : target.size(-1)]
        zeros = torch.zeros(
            tgt.size(0),
            use_pred_len,
            tgt.size(-1),
            device=tgt.device,
            dtype=tgt.dtype,
        )
        return torch.cat([context, zeros], dim=1), target

    if p == "pred_window":
        use_pred_len = int(pred_len) if pred_len is not None else int(tgt.size(1))
        use_pred_len = max(1, min(use_pred_len, tgt.size(1)))
        target = tgt[:, :use_pred_len, :]
        use_label_len = int(label_len) if label_len is not None else max(1, min(48, tgt.size(1)))
        use_label_len = max(1, min(use_label_len, tgt.size(1)))
        context = tgt[:, :use_label_len, :]
        zeros = torch.zeros(
            tgt.size(0),
            use_pred_len,
            tgt.size(-1),
            device=tgt.device,
            dtype=tgt.dtype,
        )
        return torch.cat([context, zeros], dim=1), target

    return tgt[:, :-1], tgt[:, 1:]


def training_forward(
    model,
    src: torch.Tensor,
    decoder_input: torch.Tensor,
    tgt_mask: torch.Tensor | None = None,
    supervision_protocol: str = "shifted_next_step",
    pred_len: int | None = None,
    oneshot_models: Iterable[str] | None = None,
    model_forward_kwargs: dict | None = None,
):
    """
    Unified train-time forward.

    For one-shot models:
    - pred_window: use decoder context + target_length
    - shifted_next_step: target_length equals decoder length
    """
    oneshot = set(oneshot_models or DEFAULT_ONESHOT_MODELS)
    forward_kwargs = dict(model_forward_kwargs or {})
    if is_oneshot_model(model, oneshot):
        protocol = str(supervision_protocol).lower().strip()
        if protocol in {"pred_window", "source_context_pred_window"}:
            use_pred_len = int(pred_len) if pred_len is not None else max(1, decoder_input.size(1) // 2)
            use_pred_len = max(1, use_pred_len)
            try:
                return model(
                    src,
                    target_length=use_pred_len,
                    decoder_context=decoder_input,
                    **forward_kwargs,
                )
            except TypeError:
                if forward_kwargs:
                    raise
                return model(src, target_length=use_pred_len)
        return model(src, target_length=decoder_input.size(1), **forward_kwargs)

    if tgt_mask is None:
        return model(src, decoder_input, **forward_kwargs)
    return model(src, decoder_input, tgt_mask=tgt_mask, **forward_kwargs)
