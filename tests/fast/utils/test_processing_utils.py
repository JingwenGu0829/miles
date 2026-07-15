import sys
from types import SimpleNamespace

import pytest

from miles.utils.processing_utils import process_multimodal_info


def test_process_multimodal_info_retains_video_sources_in_prompt_order(monkeypatch):
    calls = {}

    def fake_process_vision_info(prompt, image_patch_size):
        calls["prompt"] = prompt
        calls["image_patch_size"] = image_patch_size
        return ["resolved-image"], ["processed-video-1", "processed-video-2"]

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=fake_process_vision_info),
    )
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "first.mp4"},
                {"type": "image", "image": "image.png"},
                {"type": "video", "video": "https://example.test/second.mp4"},
            ],
        }
    ]
    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))

    processor_inputs, rollout_inputs = process_multimodal_info(prompt, processor)

    assert processor_inputs == {
        "images": ["resolved-image"],
        "videos": ["processed-video-1", "processed-video-2"],
    }
    assert rollout_inputs == {"video_data": ["first.mp4", "https://example.test/second.mp4"]}
    assert calls == {"prompt": prompt, "image_patch_size": 16}


@pytest.mark.parametrize(
    "video_item, error_type, message",
    [
        ({"type": "video", "video": ["frame-1.png"]}, TypeError, "path, URL, or data URI"),
        (
            {"type": "video", "video": "video.mp4", "fps": 4},
            ValueError,
            "per-item processing options",
        ),
    ],
)
def test_process_multimodal_info_rejects_video_sources_the_engine_cannot_replay(video_item, error_type, message):
    prompt = [{"role": "user", "content": [video_item]}]

    with pytest.raises(error_type, match=message):
        process_multimodal_info(prompt, object())
