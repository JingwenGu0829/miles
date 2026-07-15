"""SGLang request fields for multimodal conditioning inputs."""

import base64
import io
import os
import wave
from urllib.parse import unquote, urlparse

from miles.utils.processing_utils import encode_image_for_rollout_engine

_ROLLOUT_INPUT_KEYS = {"video_data"}
_AUDIO_SAMPLE_RATE = 16000  # qwen-omni-utils resamples audio to 16 kHz.


def _encode_audio(audio) -> str:
    import numpy as np

    samples = np.asarray(audio)
    if samples.ndim != 1 or samples.dtype.kind != "f":
        raise TypeError("Audio processor output must be a one-dimensional floating-point waveform")

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(_AUDIO_SAMPLE_RATE)
        output.writeframes(pcm.tobytes())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _validate_video(source: str) -> str:
    if not isinstance(source, str):
        raise TypeError("Video rollout input must be a path, URL, or data URI")
    if source.startswith(("http://", "https://", "data:")):
        return source

    path = unquote(urlparse(source).path) if source.startswith("file://") else source
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video rollout input does not exist: {path}")
    return source


def build_rollout_engine_multimodal_payload(
    multimodal_inputs: dict | None,
    multimodal_rollout_inputs: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Serialize processor outputs into SGLang request fields."""
    payload = dict(multimodal_rollout_inputs or {})
    unknown_keys = payload.keys() - _ROLLOUT_INPUT_KEYS
    if unknown_keys:
        raise ValueError(f"Unsupported multimodal rollout fields: {sorted(unknown_keys)}")

    videos = (multimodal_inputs or {}).get("videos") or []
    video_data = payload.get("video_data") or []
    if len(videos) != len(video_data):
        raise ValueError("Processor video inputs and rollout video sources must have the same length")
    if video_data:
        payload["video_data"] = [_validate_video(source) for source in video_data]

    if multimodal_inputs and multimodal_inputs.get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in multimodal_inputs["images"]]
    if multimodal_inputs and multimodal_inputs.get("audio"):
        payload["audio_data"] = [_encode_audio(audio) for audio in multimodal_inputs["audio"]]
    return payload


def has_multimodal_inputs(
    multimodal_inputs: dict | None,
    multimodal_rollout_inputs: dict[str, list[str]] | None = None,
) -> bool:
    return bool(multimodal_rollout_inputs) or bool(
        multimodal_inputs and (multimodal_inputs.get("images") or multimodal_inputs.get("audio"))
    )
