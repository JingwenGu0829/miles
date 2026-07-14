import base64
import inspect
import io
import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer as RawTokenizer
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin

from miles.utils.hf_config import register_hf_config_aliases

logger = logging.getLogger(__name__)


def _fix_v5_tokenizer_components(tokenizer: PreTrainedTokenizerBase, model_name_or_path: str) -> None:
    # transformers v5's LlamaTokenizerFast rebuilds pre_tokenizer/decoder in
    # __init__, discarding the originals from tokenizer.json.  DeepSeek-V3.2
    # declares LlamaTokenizerFast but actually uses ByteLevel, so without this
    # fix the loaded tokenizer decodes Metaspace ▁ instead of ByteLevel Ġ/Ċ
    # and diverges from the sglang-served tokenizer.  Mirrors sglang's
    # _fix_v5_tokenizer_components (hf_transformers_utils.py).
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return

    try:
        local_path = Path(model_name_or_path) / "tokenizer.json"
        if local_path.is_file():
            tok_file = str(local_path)
        else:
            tok_file = hf_hub_download(model_name_or_path, "tokenizer.json", local_files_only=True)
        raw = RawTokenizer.from_file(tok_file)
    except Exception as e:
        logger.warning("Could not load tokenizer.json for %s: %s", model_name_or_path, e)
        return

    raw_pre = type(raw.pre_tokenizer).__name__ if raw.pre_tokenizer else None
    loaded_pre = type(backend.pre_tokenizer).__name__ if backend.pre_tokenizer else None

    if raw_pre and loaded_pre and raw_pre != loaded_pre:
        logger.info(
            "Fixing v5 tokenizer component mismatch for %s: pre_tokenizer %s -> %s, decoder %s -> %s",
            model_name_or_path,
            loaded_pre,
            raw_pre,
            type(backend.decoder).__name__ if backend.decoder else None,
            type(raw.decoder).__name__ if raw.decoder else None,
        )
        backend.pre_tokenizer = raw.pre_tokenizer
        backend.decoder = raw.decoder


# Default image patch size for vision-language models
# Note: Qwen3-VL uses 16, Qwen2.5-VL uses 14
# Reference: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/README.md
DEFAULT_PATCH_SIZE = 14


_TOKENIZER_CACHE: dict[tuple, PreTrainedTokenizerBase] = {}


def _make_cache_key(name_or_path: str, chat_template_path: str | None, kwargs: dict) -> tuple | None:
    try:
        kwargs_items = tuple(sorted(kwargs.items()))
        hash(kwargs_items)
    except TypeError:
        return None
    return (name_or_path, chat_template_path, kwargs_items)


def load_tokenizer(name_or_path: str, chat_template_path: str | None = None, **kwargs) -> PreTrainedTokenizerBase:
    # Cache keyed by (name, chat_template_path, kwargs) — the fast suite creates
    # hundreds of SessionServer / MockSGLangServer fixtures and each previously
    # triggered a fresh AutoTokenizer.from_pretrained, tripping HF Hub rate limits.
    cache_key = _make_cache_key(name_or_path, chat_template_path, kwargs)
    if cache_key is not None and cache_key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[cache_key]

    register_hf_config_aliases()
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    _fix_v5_tokenizer_components(tokenizer, name_or_path)
    if chat_template_path:
        assert os.path.isfile(chat_template_path), (
            f"chat_template_path not found: {chat_template_path}. "
            f"Ensure the path is accessible on this node (e.g. inside the miles repo or on a shared filesystem)."
        )
        with open(chat_template_path) as f:
            tokenizer.chat_template = f.read()
        logger.info("Loaded custom chat template from %s", chat_template_path)

    if cache_key is not None:
        _TOKENIZER_CACHE[cache_key] = tokenizer
    return tokenizer


def build_processor_kwargs(multimodal_inputs: dict | None = None) -> dict:

    modality_forced = {"return_tensors": "pt"}

    result = dict(multimodal_inputs) if multimodal_inputs else {}

    # return_tensors=None for text (input_ids), "pt" for modality-specific outputs.
    # Use per-modality dicts to avoid transformers >=5.0 duplicate kwarg error.
    result["text_kwargs"] = {**result.get("text_kwargs", {}), "return_tensors": None}
    for key in ("audio_kwargs", "images_kwargs", "videos_kwargs"):
        if key in result:
            result[key] = {**result[key], **modality_forced}
        else:
            result[key] = modality_forced.copy()

    return result


def processor_requires_medias(processor) -> bool:
    try:
        params = inspect.signature(processor).parameters
        return "medias" in params and "text" in params
    except (TypeError, ValueError):
        return hasattr(processor, "media_processor")


def call_processor(processor, text, multimodal_inputs: dict | None = None):
    multimodal_inputs = multimodal_inputs or {}

    # for kimi-vl & kimi-2.5
    if processor_requires_medias(processor):
        medias = []
        if images := multimodal_inputs.get("images"):
            medias.extend({"type": "image", "image": image} for image in images)
        if videos := multimodal_inputs.get("videos"):
            medias.extend({"type": "video", "video": video} for video in videos)
        if audios := multimodal_inputs.get("audio"):
            medias.extend({"type": "audio", "audio": audio} for audio in audios)
        return processor(text=text, medias=medias)

    kwargs = build_processor_kwargs(multimodal_inputs)
    return processor(text=text, **kwargs)


def load_processor(name_or_path: str, **kwargs):
    try:
        proc = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to load processor from {name_or_path}: {e}")
        proc = None

    # If HF returned a tokenizer, discard it.
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        proc = None

    return proc


def processor_supports_audio(processor) -> bool:
    """Whether a Transformers-style processor exposes an audio input."""
    try:
        if "audio" in inspect.signature(processor.__call__).parameters:
            return True
    except (TypeError, ValueError):
        pass
    return getattr(processor, "audio_token", None) is not None and hasattr(processor, "feature_extractor")


def _unwrap_media_source(value):
    if isinstance(value, dict):
        if "url" in value:
            return value["url"]
        if "path" in value:
            return value["path"]
    return value


def extract_multimodal_rollout_inputs(conversations) -> dict[str, list]:
    """Extract original media sources from structured chat messages.

    Local preprocessing turns videos into frame tensors and audio into waveforms. Those
    values are useful to the training processor but cannot always be sent through the
    SGLang HTTP API, so retain the original source alongside them.
    """
    if not isinstance(conversations, list) or not conversations:
        return {}
    if isinstance(conversations[0], dict):
        conversations = [conversations]

    result = {"images": [], "videos": [], "audio": []}
    aliases = {
        "image": ("images", ("image", "image_url")),
        "image_url": ("images", ("image_url", "image")),
        "video": ("videos", ("video", "video_url")),
        "video_url": ("videos", ("video_url", "video")),
        "audio": ("audio", ("audio", "audio_url")),
        "audio_url": ("audio", ("audio_url", "audio")),
        "input_audio": ("audio", ("input_audio", "audio")),
    }
    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") not in aliases:
                    continue
                output_key, source_keys = aliases[item["type"]]
                source = next((item[key] for key in source_keys if key in item), None)
                if source is not None:
                    result[output_key].append(_unwrap_media_source(source))
    return {key: values for key, values in result.items() if values}


def process_multimodal_info(prompt, processor, *, use_audio_in_video: bool = False):
    """Build processor-ready and rollout-ready image/video/audio inputs."""
    rollout_inputs = extract_multimodal_rollout_inputs(prompt)
    image_processor = getattr(processor, "image_processor", None)
    image_patch_size = getattr(image_processor, "patch_size", DEFAULT_PATCH_SIZE)
    if image_processor is not None and not hasattr(image_processor, "patch_size"):
        logger.info("Using default patch size: %s", DEFAULT_PATCH_SIZE)

    if processor_supports_audio(processor):
        from qwen_omni_utils import process_mm_info

        audios, images, videos, video_kwargs = process_mm_info(
            prompt,
            use_audio_in_video=use_audio_in_video,
            return_video_kwargs=True,
            image_patch_size=image_patch_size,
        )
        multimodal_inputs = {"audio": audios, "images": images, "videos": videos}
        if videos is not None:
            multimodal_inputs.update(video_kwargs)
        if use_audio_in_video:
            multimodal_inputs["use_audio_in_video"] = True
    else:
        # TODO: temporary solution, will write model-independent media loaders later.
        from qwen_vl_utils import process_vision_info as qwen_process_vision_info

        images, videos = qwen_process_vision_info(prompt, image_patch_size=image_patch_size)
        multimodal_inputs = {"images": images, "videos": videos}

    multimodal_inputs = {key: value for key, value in multimodal_inputs.items() if value is not None}
    return multimodal_inputs, rollout_inputs


def process_vision_info(prompt, processor):
    """Backward-compatible wrapper returning only processor-ready media."""
    multimodal_inputs, _ = process_multimodal_info(prompt, processor)
    return multimodal_inputs


def encode_image_for_rollout_engine(image) -> str:
    """Load an image from path, ensure RGB, encode as PNG base64 string."""
    buffer = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def get_prompt_ids_and_multimodal_train_inputs(processor, text, multimodal_inputs):
    """Run a processor and separate prompt IDs from tensor-like model inputs."""
    import numpy as np
    import torch

    processor_output = call_processor(processor, text, multimodal_inputs)
    prompt_ids = processor_output["input_ids"][0]
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    prompt_ids = [int(token_id) for token_id in prompt_ids]

    train_inputs = {}
    for key, value in processor_output.items():
        if key in ("input_ids", "attention_mask"):
            continue
        if isinstance(value, torch.Tensor):
            train_inputs[key] = value
        elif isinstance(value, np.ndarray):
            train_inputs[key] = torch.from_numpy(value)
        elif isinstance(value, (list, tuple)):
            try:
                train_inputs[key] = torch.as_tensor(value)
            except (TypeError, ValueError, RuntimeError):
                logger.debug("Dropping non-tensor processor output %s (%s)", key, type(value).__name__)
    return prompt_ids, train_inputs or None
