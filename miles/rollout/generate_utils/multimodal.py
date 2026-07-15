from typing import Any

from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import RolloutMediaSources


def build_rollout_engine_multimodal_payload(
    multimodal_inputs: dict[str, Any] | None,
    rollout_media_sources: RolloutMediaSources | None,
) -> dict[str, list[str]]:
    """Build the JSON-safe multimodal portion of a rollout request."""
    multimodal_inputs = multimodal_inputs or {}
    rollout_media_sources = rollout_media_sources or {}

    unsupported_modalities = set(rollout_media_sources) - {"videos"}
    if unsupported_modalities:
        raise ValueError(f"Unsupported rollout media modalities: {sorted(unsupported_modalities)}")

    payload = {}
    if image_data := multimodal_inputs.get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    processed_videos = multimodal_inputs.get("videos")
    video_data = rollout_media_sources.get("videos")
    if video_data is not None and any(not isinstance(source, str) for source in video_data):
        raise TypeError("Rollout video sources must be paths, URLs, or data URIs")
    processed_video_count = len(processed_videos) if processed_videos is not None else 0
    rollout_video_count = len(video_data) if video_data is not None else 0
    if processed_video_count != rollout_video_count:
        raise ValueError(
            "Video processor inputs and rollout sources must have the same length: "
            f"processed={processed_video_count}, rollout={rollout_video_count}"
        )
    if video_data:
        payload["video_data"] = list(video_data)

    return payload


def has_multimodal_inputs(
    multimodal_inputs: dict[str, Any] | None,
    rollout_media_sources: RolloutMediaSources | None,
) -> bool:
    processor_media = ((multimodal_inputs or {}).get(key) for key in ("images", "videos", "audio", "audios"))
    return any(value is not None and len(value) > 0 for value in processor_media) or bool(rollout_media_sources)


def build_rollout_input_ids(
    input_ids: list[int],
    *,
    processor_prompt_ids: list[int],
    rollout_prompt_ids: list[int] | None,
) -> list[int]:
    """Replace a locally expanded prompt prefix with its tokenizer-only form."""
    input_ids = list(input_ids)
    processor_prompt_ids = list(processor_prompt_ids)
    if rollout_prompt_ids is None:
        return input_ids

    if input_ids[: len(processor_prompt_ids)] != processor_prompt_ids:
        raise ValueError("Cannot build rollout_input_ids: input IDs do not start with the processed prompt IDs")

    return list(rollout_prompt_ids) + input_ids[len(processor_prompt_ids) :]
