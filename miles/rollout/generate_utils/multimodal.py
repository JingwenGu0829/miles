from typing import Any

from miles.utils.processing_utils import encode_image_for_rollout_engine


def build_rollout_engine_multimodal_payload(
    multimodal_inputs: dict[str, Any] | None,
    multimodal_rollout_inputs: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    """Build the JSON-safe multimodal portion of a rollout request."""
    multimodal_inputs = multimodal_inputs or {}
    multimodal_rollout_inputs = multimodal_rollout_inputs or {}

    unsupported_fields = set(multimodal_rollout_inputs) - {"video_data"}
    if unsupported_fields:
        raise ValueError(f"Unsupported multimodal rollout fields: {sorted(unsupported_fields)}")

    payload = {}
    if image_data := multimodal_inputs.get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    processed_videos = multimodal_inputs.get("videos")
    video_data = multimodal_rollout_inputs.get("video_data")
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
    multimodal_rollout_inputs: dict[str, list[str]] | None,
) -> bool:
    return any(value is not None and len(value) > 0 for value in (multimodal_inputs or {}).values()) or any(
        value is not None and len(value) > 0 for value in (multimodal_rollout_inputs or {}).values()
    )


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
