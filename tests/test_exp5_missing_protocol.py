import torch

from experiments.exp5_missing_data.missing_data_experiment import create_mask


def test_random_missing_mask_is_shared_by_seed():
    first = create_mask(
        3, 8, 6, 0.2, "R", generator=torch.Generator().manual_seed(1234)
    )
    second = create_mask(
        3, 8, 6, 0.2, "R", generator=torch.Generator().manual_seed(1234)
    )

    assert torch.equal(first, second)


def test_continuous_missing_mask_has_requested_centered_block():
    mask = create_mask(1, 10, 2, 0.4, "C")

    assert torch.equal(mask[:, 3:7, :], torch.zeros(1, 4, 2))
    assert torch.equal(mask[:, :3, :], torch.ones(1, 3, 2))
    assert torch.equal(mask[:, 7:, :], torch.ones(1, 3, 2))
