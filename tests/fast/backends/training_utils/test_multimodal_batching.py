import torch

from miles.backends.training_utils.data import _concatenate_audio_tensors


def test_concatenate_audio_tensors_pads_variable_length_dimension():
    first = torch.ones((1, 80, 3))
    second = torch.full((1, 80, 5), 2.0)

    result = _concatenate_audio_tensors([first, second])

    assert result.shape == (2, 80, 5)
    assert torch.equal(result[0, :, :3], first[0])
    assert torch.count_nonzero(result[0, :, 3:]) == 0
    assert torch.equal(result[1], second[0])
