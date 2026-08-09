import torch
from torch.utils.data import DataLoader, TensorDataset

from models.plgaformer import PLGAFormerTransformer
from utils.inference_protocol import build_source_context_for_eval, default_causal_mask
from utils.physics_prior_cache import (
    PhysicsPriorDataset,
    build_cached_physics_prior_loader,
    clear_process_physics_prior_cache,
    unpack_batch_with_optional_prior,
)


def _model():
    input_mean = torch.tensor([6_430_000.0, 0.0, 0.0, 5_000.0, -0.1, 0.0])
    input_scale = torch.tensor([10_000.0, 0.1, 0.1, 1_000.0, 0.1, 0.1])
    return PLGAFormerTransformer(
        input_dim=6,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        use_sparse_attention=False,
        use_physics_corrector=False,
        use_multi_head_output=True,
        use_prior_fusion=True,
        input_scaler_mean=input_mean,
        input_scaler_scale=input_scale,
        output_scaler_mean=input_mean[:3],
        output_scaler_scale=input_scale[:3],
    ).eval()


def test_cached_prior_loader_preserves_samples_and_batch_order():
    torch.manual_seed(7)
    model = _model()
    source = torch.randn(5, 12, 6) * 0.05
    target = torch.randn(5, 8, 3)
    loader = DataLoader(TensorDataset(source, target), batch_size=2, shuffle=False)

    cached_loader, stats = build_cached_physics_prior_loader(
        loader,
        model,
        device="cpu",
        target_length=8,
        cache_batch_size=3,
    )

    assert stats.enabled
    assert stats.samples == 5
    assert stats.target_length == 8
    assert isinstance(cached_loader.dataset, PhysicsPriorDataset)
    cached_source, cached_target, cached_prior = unpack_batch_with_optional_prior(
        next(iter(cached_loader))
    )
    expected_prior = model.physics_prior(source[:2], target_length=8)
    assert torch.equal(cached_source, source[:2])
    assert torch.equal(cached_target, target[:2])
    assert torch.allclose(cached_prior, expected_prior, rtol=0.0, atol=0.0)


def test_cached_prior_override_is_numerically_equivalent_to_inline_prior():
    torch.manual_seed(11)
    model = _model()
    source = torch.randn(2, 12, 6) * 0.05
    decoder_input = build_source_context_for_eval(
        x=source,
        target_length=8,
        label_len=4,
        output_dim=3,
    )
    mask = default_causal_mask(decoder_input.size(1), source.device)

    with torch.no_grad():
        inline = model(source, decoder_input, tgt_mask=mask)
        cached_prior = model.physics_prior(source, target_length=8)
        cached = model(
            source,
            decoder_input,
            tgt_mask=mask,
            physics_prior_override=cached_prior,
        )

    assert torch.allclose(cached, inline, rtol=0.0, atol=0.0)


def test_identical_prior_and_dataset_reuse_process_cache():
    clear_process_physics_prior_cache()
    torch.manual_seed(13)
    model = _model()
    source = torch.randn(5, 12, 6) * 0.05
    target = torch.randn(5, 8, 3)
    loader = DataLoader(TensorDataset(source, target), batch_size=2, shuffle=False)

    _, first = build_cached_physics_prior_loader(
        loader, model, device="cpu", target_length=8, cache_batch_size=3
    )
    _, second = build_cached_physics_prior_loader(
        loader, model, device="cpu", target_length=8, cache_batch_size=3
    )

    assert first.reason == "precomputed_in_memory"
    assert second.reason == "reused_process_cache"
    assert second.elapsed_sec == 0.0
