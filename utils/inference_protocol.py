#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified inference protocol helpers shared by experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch

DEFAULT_ONESHOT_MODELS = {
    "Informer",
    "Autoformer",
    "FEDformer",
    "TimesNet",
    "iTransformer",
    "CrossFormer",
    "PatchTST",
    "PIT",
    "AFCILNExternalAdapter",
    "RotatingEarth3DOFBaseline",
}


def default_causal_mask(size: int, device) -> torch.Tensor:
    """Build a standard causal mask."""
    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))


def is_oneshot_model(model, oneshot_models: Iterable[str] | None = None) -> bool:
    """Return True if model is expected to support one-shot prediction."""
    oneshot = set(oneshot_models or DEFAULT_ONESHOT_MODELS)
    return bool(getattr(model, "is_oneshot", False)) or (
        hasattr(model, "__class__") and model.__class__.__name__ in oneshot
    )


def build_decoder_context_for_eval(
    y_true_scaled: torch.Tensor,
    target_length: int,
    label_len: int,
) -> torch.Tensor:
    """Build decoder context using label_len + zero-future semantics."""
    label_len = max(1, min(int(label_len), y_true_scaled.size(1)))
    context = y_true_scaled[:, :label_len, :]
    zeros = torch.zeros(
        y_true_scaled.size(0),
        max(1, int(target_length)),
        y_true_scaled.size(-1),
        device=y_true_scaled.device,
        dtype=y_true_scaled.dtype,
    )
    return torch.cat([context, zeros], dim=1)


def build_source_context_for_eval(
    x: torch.Tensor,
    target_length: int,
    label_len: int,
    output_dim: int,
) -> torch.Tensor:
    """Build decoder context from the observed source trajectory without future leakage."""
    label_len = max(1, min(int(label_len), x.size(1)))
    context = x[:, -label_len:, :output_dim]
    zeros = torch.zeros(
        x.size(0),
        max(1, int(target_length)),
        output_dim,
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([context, zeros], dim=1)


def predict_by_eval_protocol(
    model,
    x: torch.Tensor,
    y_true_scaled: torch.Tensor,
    pred_length: int,
    device,
    eval_protocol: str,
    eval_ar_seed_mode: str,
    label_len: int,
    causal_mask_builder: Callable[[int, object], torch.Tensor] | None = None,
    oneshot_models: Iterable[str] | None = None,
) -> torch.Tensor:
    """Unified eval-time prediction for one-shot/autoregressive models."""
    protocol = str(eval_protocol).lower().strip()
    pred_length = max(1, int(pred_length))
    if is_oneshot_model(model, oneshot_models):
        if protocol == "source_context_decoder":
            decoder_context = build_source_context_for_eval(
                x=x,
                target_length=pred_length,
                label_len=label_len,
                output_dim=y_true_scaled.size(-1),
            )
            try:
                return model(x, target_length=pred_length, decoder_context=decoder_context)
            except TypeError:
                return model(x, target_length=pred_length)
        if protocol == "official_like_decoder":
            decoder_context = build_decoder_context_for_eval(
                y_true_scaled=y_true_scaled,
                target_length=pred_length,
                label_len=label_len,
            )
            try:
                return model(x, target_length=pred_length, decoder_context=decoder_context)
            except TypeError:
                return model(x, target_length=pred_length)
        return model(x, target_length=pred_length)

    if protocol == "source_context_decoder":
        y_input = build_source_context_for_eval(
            x=x,
            target_length=pred_length,
            label_len=label_len,
            output_dim=y_true_scaled.size(-1),
        )
        build_mask = causal_mask_builder or default_causal_mask
        tgt_mask = build_mask(y_input.size(1), device)
        pred = model(x, y_input, tgt_mask=tgt_mask)
        return pred[:, -pred_length:, :]

    seed_mode = str(eval_ar_seed_mode).lower().strip()
    if seed_mode == "gt_first":
        y_input = y_true_scaled[:, :1, :].clone()
    else:
        y_input = torch.zeros(
            y_true_scaled.size(0),
            1,
            y_true_scaled.size(-1),
            device=device,
            dtype=y_true_scaled.dtype,
        )

    build_mask = causal_mask_builder or default_causal_mask
    for _ in range(pred_length):
        tgt_mask = build_mask(y_input.size(1), device)
        pred = model(x, y_input, tgt_mask=tgt_mask)
        y_input = torch.cat([y_input, pred[:, -1:, :]], dim=1)
    return y_input[:, 1:, :]
