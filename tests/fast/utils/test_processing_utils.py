import sys
from types import SimpleNamespace

import pytest

from miles.utils.processing_utils import process_vision_info_with_sources


def test_process_vision_info_retains_video_sources_in_prompt_order(monkeypatch):
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

    processor_inputs, rollout_media_sources = process_vision_info_with_sources(prompt, processor)

    assert processor_inputs == {
        "images": ["resolved-image"],
        "videos": ["processed-video-1", "processed-video-2"],
    }
    assert rollout_media_sources == {"videos": ["first.mp4", "https://example.test/second.mp4"]}
    assert calls == {"prompt": prompt, "image_patch_size": 16}


def test_process_vision_info_rejects_video_sources_the_engine_cannot_replay():
    invalid_items = [
        ({"type": "video", "video": ["frame-1.png"]}, TypeError),
        ({"type": "video", "video": "video.mp4", "fps": 4}, ValueError),
    ]
    for item, error_type in invalid_items:
        with pytest.raises(error_type):
            process_vision_info_with_sources([{"role": "user", "content": [item]}], object())
