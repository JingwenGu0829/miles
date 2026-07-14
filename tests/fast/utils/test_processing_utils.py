import base64
import io
import sys
import types
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from miles.rollout.generate_utils.multimodal import build_rollout_engine_multimodal_payload
from miles.utils.processing_utils import (
    extract_multimodal_rollout_inputs,
    get_prompt_ids_and_multimodal_train_inputs,
    process_multimodal_info,
)


def _decode_data_uri(uri: str) -> bytes:
    return base64.b64decode(uri.split(",", 1)[1])


def test_extract_multimodal_rollout_inputs_preserves_sources_in_modality_order():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image.png"},
                {"type": "audio_url", "audio_url": {"url": "https://example.test/a.wav"}},
                {"type": "video", "video": "data:video/mp4;base64,AAAA"},
                {"type": "image_url", "image_url": {"url": "https://example.test/b.png"}},
                {"type": "text", "text": "What happened?"},
            ],
        }
    ]

    assert extract_multimodal_rollout_inputs(messages) == {
        "images": ["image.png", "https://example.test/b.png"],
        "videos": ["data:video/mp4;base64,AAAA"],
        "audio": ["https://example.test/a.wav"],
    }


def test_build_rollout_payload_serializes_image_video_and_audio():
    image = Image.new("RGB", (4, 3), color="red")
    waveform = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    video_uri = "data:video/mp4;base64,AAAA"

    payload = build_rollout_engine_multimodal_payload(
        {
            "images": [image],
            "videos": [torch.zeros((4, 3, 8, 8))],
            "audio": [waveform],
            "audio_kwargs": {"sampling_rate": 8000},
        },
        {"videos": [video_uri]},
    )

    assert payload["video_data"] == [video_uri]
    assert payload["image_data"][0].startswith("data:image/png;base64,")
    with Image.open(io.BytesIO(_decode_data_uri(payload["image_data"][0]))) as decoded:
        assert decoded.mode == "RGB"
        assert decoded.size == (4, 3)

    assert payload["audio_data"][0].startswith("data:audio/wav;base64,")
    with wave.open(io.BytesIO(_decode_data_uri(payload["audio_data"][0])), "rb") as decoded:
        assert decoded.getframerate() == 8000
        assert decoded.getnchannels() == 1
        assert decoded.getnframes() == 3


def test_video_frames_require_an_original_rollout_source():
    with pytest.raises(TypeError, match="video rollout source"):
        build_rollout_engine_multimodal_payload({"videos": [torch.zeros((4, 3, 8, 8))]})


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
                {"type": "audio", "audio": "clip.wav"},
                {"type": "video", "video": "clip.mp4"},
            ],
        }
    ]
    processor_inputs, rollout_inputs = process_multimodal_info(messages, Processor())

    assert set(processor_inputs) == {"audio", "images", "videos", "fps", "do_sample_frames"}
    assert rollout_inputs == {"videos": ["clip.mp4"], "audio": ["clip.wav"]}
    assert calls["kwargs"] == {
        "use_audio_in_video": False,
        "return_video_kwargs": True,
        "image_patch_size": 16,
    }
