import pytest
import torch

from miles.backends.training_utils.data import _concatenate_multimodal_tensors


def test_concatenate_multimodal_tensors_pads_variable_audio_time_dimension():
    first = torch.ones((1, 80, 3))
    second = torch.full((1, 80, 5), 2.0)

    result = _concatenate_multimodal_tensors("input_features", [first, second])

    assert result.shape == (2, 80, 5)
    assert torch.equal(result[0, :, :3], first[0])
    assert torch.count_nonzero(result[0, :, 3:]) == 0
    assert torch.equal(result[1], second[0])


def test_concatenate_multimodal_tensors_keeps_strict_vision_shapes():
    with pytest.raises(RuntimeError):
        _concatenate_multimodal_tensors(
            "pixel_values",
            [torch.ones((1, 3, 3)), torch.ones((1, 3, 4))],
        )
