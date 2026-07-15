from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload
from miles.rollout.generate_utils.multimodal import build_rollout_engine_multimodal_payload


def test_compute_request_payload_includes_all_conditioning_modalities():
    args = SimpleNamespace(
        rollout_max_response_len=32,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_path=None,
    )

    payload, status = compute_request_payload(
        args,
        input_ids=[1, 2, 3],
        sampling_params={"temperature": 1.0},
        multimodal_inputs={
            "images": [Image.new("RGB", (2, 2))],
            "audio": [np.zeros(4, dtype=np.float32)],
            "videos": [object()],
        },
        multimodal_rollout_inputs={
            "video_data": ["https://example.test/video.mp4"],
        },
    )

    assert status is None
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["image_data"][0].startswith("data:image/png;base64,")
    assert payload["video_data"] == ["https://example.test/video.mp4"]
    assert payload["audio_data"][0].startswith("data:audio/wav;base64,")


def test_rollout_inputs_reject_unknown_fields():
    with pytest.raises(ValueError, match="Unsupported multimodal rollout fields"):
        build_rollout_engine_multimodal_payload(None, {"audio": ["audio.wav"]})


def test_rollout_inputs_require_one_source_per_processed_video():
    with pytest.raises(ValueError, match="same length"):
        build_rollout_engine_multimodal_payload({"videos": [object()]})
