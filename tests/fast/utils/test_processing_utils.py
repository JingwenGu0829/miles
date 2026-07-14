import sys
import types
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from miles.utils.processing_utils import (
    get_prompt_ids_and_multimodal_train_inputs,
    process_multimodal_info,
)


def test_get_prompt_ids_keeps_only_tensor_convertible_training_inputs():
    class Processor:
        def __call__(self, text=None, **kwargs):
            return {
                "input_ids": [[10, 11, 12]],
                "attention_mask": [[1, 1, 1]],
                "input_features": torch.ones((1, 80, 7)),
                "feature_attention_mask": np.ones((1, 7), dtype=np.int64),
                "video_second_per_grid": [0.5],
                "video_metadata": [{"fps": 2.0}],
            }

    prompt_ids, train_inputs = get_prompt_ids_and_multimodal_train_inputs(
        Processor(), "prompt", {"audio": [np.zeros(10)]}
    )

    assert prompt_ids == [10, 11, 12]
    assert set(train_inputs) == {"input_features", "feature_attention_mask", "video_second_per_grid"}
    assert train_inputs["feature_attention_mask"].dtype == torch.int64
    assert train_inputs["video_second_per_grid"].tolist() == [0.5]


def test_process_multimodal_info_uses_omni_loader_for_audio_processor(monkeypatch):
    calls = {}

    def process_mm_info(messages, **kwargs):
        calls["messages"] = messages
        calls["kwargs"] = kwargs
        return (
            [np.zeros(4)],
            [Image.new("RGB", (2, 2))],
            [torch.zeros((4, 3, 2, 2))],
            {
                "fps": [2.0],
                "do_sample_frames": False,
            },
        )

    monkeypatch.setitem(sys.modules, "qwen_omni_utils", types.SimpleNamespace(process_mm_info=process_mm_info))

    class Processor:
        image_processor = SimpleNamespace(patch_size=16)
        audio_token = "<|audio|>"
        feature_extractor = object()

        def __call__(self, text=None, audio=None, **kwargs):
            raise AssertionError("not called while loading media")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                {"type": "audio_url", "audio_url": {"url": "https://example.test/audio.wav"}},
                {"type": "video", "video": "clip.mp4"},
            ],
        }
    ]
    processor_inputs, rollout_inputs = process_multimodal_info(messages, Processor())

    assert set(processor_inputs) == {"audio", "images", "videos", "fps", "do_sample_frames"}
    assert rollout_inputs == {
        "images": ["https://example.test/image.png"],
        "videos": ["clip.mp4"],
        "audio": ["https://example.test/audio.wav"],
    }
    assert calls["kwargs"] == {
        "use_audio_in_video": False,
        "return_video_kwargs": True,
        "image_patch_size": 16,
    }
