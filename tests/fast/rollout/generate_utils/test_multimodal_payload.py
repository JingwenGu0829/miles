import base64
import io
import wave
from types import SimpleNamespace

import numpy as np
from PIL import Image

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
            "images": [Image.new("RGB", (4, 3), color="red")],
            "audio": [np.zeros(16, dtype=np.float32)],
            "audio_kwargs": {"sampling_rate": 8000},
        },
        multimodal_rollout_inputs={"videos": ["https://example.test/video.mp4"]},
    )

    assert status is None
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["image_data"][0].startswith("data:image/png;base64,")
    assert payload["video_data"] == ["https://example.test/video.mp4"]
    _, encoded_audio = payload["audio_data"][0].split(",", 1)
    with wave.open(io.BytesIO(base64.b64decode(encoded_audio)), "rb") as decoded:
        assert (decoded.getframerate(), decoded.getnchannels(), decoded.getnframes()) == (8000, 1, 16)
