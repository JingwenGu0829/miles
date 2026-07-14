"""SGLang request serialization for multimodal conditioning inputs."""

import base64
import io
import mimetypes
import os
import wave
from urllib.parse import unquote, urlparse

from miles.utils.processing_utils import encode_image_for_rollout_engine


def encode_audio_for_rollout_engine(audio, sampling_rate: int = 16000) -> str:
    """Encode one mono waveform as a lossless PCM WAV data URI."""
    import numpy as np

    samples = np.asarray(audio)
    if samples.ndim == 2 and 1 in samples.shape:
        samples = samples.reshape(-1)
    if samples.ndim != 1:
        raise ValueError(f"Audio rollout input must be mono, got shape {samples.shape}")
    if samples.dtype.kind == "f":
        samples = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    elif samples.dtype.kind in ("i", "u"):
        if samples.size and (samples.min() < -32768 or samples.max() > 32767):
            raise ValueError("Integer audio rollout input must fit in signed 16-bit PCM")
        samples = samples.astype("<i2")
    else:
        raise ValueError(f"Unsupported audio dtype: {samples.dtype}")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sampling_rate)
        output.writeframes(samples.tobytes())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _encode_bytes(data: bytes, media_type: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _unwrap_source(value):
    if isinstance(value, dict):
        if "url" in value:
            return value["url"]
        if "path" in value:
            return value["path"]
    return value


def _encode_media_reference(value, modality: str) -> str:
    value = _unwrap_source(value)
    default_type = {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}[modality]
    if isinstance(value, bytes):
        return _encode_bytes(value, default_type)
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        raise TypeError(f"{modality} rollout source must be a path, URL, data URI, or bytes; got {type(value)}")
    if value.startswith(("http://", "https://", "data:")):
        return value

    path = unquote(urlparse(value).path) if value.startswith("file://") else value
    if os.path.isfile(path):
        media_type = mimetypes.guess_type(path)[0] or default_type
        with open(path, "rb") as media_file:
            return _encode_bytes(media_file.read(), media_type)

    # SGLang also accepts an unprefixed base64 string.
    return value


def _media_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _first_present(inputs: dict | None, *keys: str):
    if not inputs:
        return None
    for key in keys:
        value = inputs.get(key)
        if value is not None and (not hasattr(value, "__len__") or len(value) > 0):
            return value
    return None


def build_rollout_engine_multimodal_payload(
    multimodal_inputs: dict | None,
    multimodal_rollout_inputs: dict | None = None,
) -> dict[str, list[str] | bool]:
    """Serialize processor media into SGLang's image/video/audio request fields."""
    payload: dict[str, list[str] | bool] = {}

    # Preserve the existing image behavior: SGLang receives the exact resized pixels
    # used by the local processor, serialized losslessly as PNG.
    images = _first_present(multimodal_inputs, "images")
    if images is None:
        images = _first_present(multimodal_rollout_inputs, "images")
    if images is not None:
        payload["image_data"] = [
            (
                encode_image_for_rollout_engine(image)
                if hasattr(image, "save") and hasattr(image, "mode")
                else _encode_media_reference(image, "image")
            )
            for image in _media_list(images)
        ]

    # Qwen's local video utility returns sampled frame tensors. Retain and send the
    # original source because tensors cannot cross the JSON API.
    videos = _first_present(multimodal_rollout_inputs, "videos")
    if videos is None:
        videos = _first_present(multimodal_inputs, "videos")
    if videos is not None:
        payload["video_data"] = [_encode_media_reference(video, "video") for video in _media_list(videos)]

    audios = _first_present(multimodal_inputs, "audio", "audios")
    if audios is not None:
        audio_kwargs = (multimodal_inputs or {}).get("audio_kwargs", {})
        sampling_rate = int(audio_kwargs.get("sampling_rate", 16000))
        encoded_audios = []
        for audio in _media_list(audios):
            if isinstance(audio, dict) and "array" in audio:
                encoded_audios.append(
                    encode_audio_for_rollout_engine(audio["array"], int(audio.get("sampling_rate", sampling_rate)))
                )
            elif isinstance(audio, tuple) and len(audio) == 2:
                encoded_audios.append(encode_audio_for_rollout_engine(audio[0], int(audio[1])))
            elif isinstance(audio, (str, bytes, os.PathLike)):
                encoded_audios.append(_encode_media_reference(audio, "audio"))
            else:
                encoded_audios.append(encode_audio_for_rollout_engine(audio, sampling_rate))
        payload["audio_data"] = encoded_audios
    else:
        raw_audios = _first_present(multimodal_rollout_inputs, "audio", "audios")
        if raw_audios is not None:
            payload["audio_data"] = [_encode_media_reference(audio, "audio") for audio in _media_list(raw_audios)]

    if (multimodal_inputs or {}).get("use_audio_in_video"):
        payload["use_audio_in_video"] = True
    return payload


def has_multimodal_inputs(
    multimodal_inputs: dict | None,
    multimodal_rollout_inputs: dict | None = None,
) -> bool:
    modality_keys = ("images", "videos", "audio", "audios")
    return any(
        _first_present(inputs, *modality_keys) is not None for inputs in (multimodal_inputs, multimodal_rollout_inputs)
    )
