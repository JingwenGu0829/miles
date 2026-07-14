from types import SimpleNamespace

import numpy as np

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload


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
            "audio": [np.zeros(16, dtype=np.float32)],
            "audio_kwargs": {"sampling_rate": 16000},
        },
        multimodal_rollout_inputs={
            "images": ["https://example.test/image.png"],
            "videos": ["https://example.test/video.mp4"],
        },
    )

    assert status is None
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["image_data"] == ["https://example.test/image.png"]
    assert payload["video_data"] == ["https://example.test/video.mp4"]
    assert payload["audio_data"][0].startswith("data:audio/wav;base64,")
