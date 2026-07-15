from types import SimpleNamespace

import pytest
from PIL import Image

from miles.rollout import sglang_rollout
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import multi_turn, single_turn
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
)
from miles.rollout.generate_utils.multimodal import (
    build_rollout_engine_multimodal_payload,
    build_rollout_input_ids,
)
from miles.utils.types import Sample


def _rollout_args(**overrides):
    defaults = dict(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_response_len=16,
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_path=None,
        sglang_speculative_algorithm=None,
        partial_rollout=False,
    )
    return SimpleNamespace(**(defaults | overrides))


class _VideoProcessor:
    def __call__(self, text, **kwargs):
        return {"input_ids": [[100, 101, 102]], "pixel_values_videos": "train-only"}


class _VideoTokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [1, 2]


def test_multimodal_payload_keeps_resolved_images_and_raw_video_sources():
    image = Image.new("RGB", (2, 2), color="red")

    payload = build_rollout_engine_multimodal_payload(
        {"images": [image], "videos": [object()]},
        {"video_data": ["https://example.test/video.mp4"]},
    )

    assert payload["image_data"][0].startswith("data:image/png;base64,")
    assert payload["video_data"] == ["https://example.test/video.mp4"]


def test_multimodal_payload_requires_one_raw_source_per_processed_video():
    with pytest.raises(ValueError, match="same length"):
        build_rollout_engine_multimodal_payload({"videos": [object()]}, None)


def test_build_rollout_input_ids_replaces_only_the_processed_prompt_prefix():
    rollout_input_ids = build_rollout_input_ids(
        [100, 101, 102, 20, 21],
        processor_prompt_ids=[100, 101, 102],
        rollout_prompt_ids=[1, 2],
    )

    assert rollout_input_ids == [1, 2, 20, 21]


def test_build_rollout_input_ids_rejects_a_noncanonical_prefix():
    with pytest.raises(ValueError, match="do not start"):
        build_rollout_input_ids(
            [999, 20],
            processor_prompt_ids=[100],
            rollout_prompt_ids=[1],
        )


def test_compute_prompt_ids_keeps_processor_and_rollout_representations_separate():
    class Processor:
        def __call__(self, text, **kwargs):
            return {"input_ids": [[100, 101, 102]], "pixel_values": "train-only"}

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert text == "<video>describe it"
            assert add_special_tokens is False
            return [1, 2]

    sample = Sample(
        prompt="<video>describe it",
        multimodal_inputs={"videos": [object()]},
        multimodal_rollout_inputs={"video_data": ["video.mp4"]},
    )
    state = SimpleNamespace(processor=Processor(), tokenizer=Tokenizer())

    prompt_ids = compute_prompt_ids_from_sample(state, sample)

    assert prompt_ids == [100, 101, 102]
    assert sample.rollout_prompt_ids == [1, 2]
    assert sample.multimodal_train_inputs == {"pixel_values": "train-only"}


def test_image_only_keeps_the_existing_processor_expanded_request_ids():
    image = Image.new("RGB", (2, 2), color="red")

    class Processor:
        def __call__(self, text, **kwargs):
            return {"input_ids": [[100, 101, 102]], "pixel_values": "train-only"}

    state = SimpleNamespace(processor=Processor(), tokenizer=_VideoTokenizer())
    sample = Sample(prompt="<image>describe it", multimodal_inputs={"images": [image]})

    prompt_ids = compute_prompt_ids_from_sample(state, sample)
    payload, status = compute_request_payload(
        _rollout_args(),
        input_ids=prompt_ids,
        sampling_params={"max_new_tokens": 4},
        multimodal_inputs=sample.multimodal_inputs,
    )

    assert status is None
    assert sample.rollout_prompt_ids is None
    assert payload["input_ids"] == [100, 101, 102]
    assert payload["image_data"][0].startswith("data:image/png;base64,")


def test_context_limit_uses_canonical_ids_while_request_uses_rollout_ids():
    args = SimpleNamespace(
        rollout_max_response_len=16,
        rollout_max_context_len=6,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        lora_rank=0,
        lora_adapter_path=None,
    )

    payload, status = compute_request_payload(
        args,
        input_ids=[100, 101, 102, 103],
        rollout_input_ids=[1, 2],
        sampling_params={"max_new_tokens": 10},
    )

    assert status is None
    assert payload["input_ids"] == [1, 2]
    assert payload["sampling_params"]["max_new_tokens"] == 2


@pytest.mark.asyncio
async def test_single_turn_sends_collapsed_video_ids_but_keeps_canonical_sample_tokens(monkeypatch):
    requests = []

    async def fake_post(url, payload):
        requests.append(payload)
        return {
            "text": "answer",
            "meta_info": {
                "output_token_logprobs": [(-0.1, 20)],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr(single_turn, "post", fake_post)
    args = _rollout_args()
    state = SimpleNamespace(args=args, processor=_VideoProcessor(), tokenizer=_VideoTokenizer())
    sample = Sample(
        prompt="<video>describe it",
        multimodal_inputs={"videos": [object()]},
        multimodal_rollout_inputs={"video_data": ["video.mp4"]},
    )

    output = await single_turn.generate(
        GenerateFnInput(
            state=state,
            sample=sample,
            sampling_params={"max_new_tokens": 4},
            evaluation=False,
        )
    )

    assert requests[0]["input_ids"] == [1, 2]
    assert requests[0]["video_data"] == ["video.mp4"]
    assert output.samples.tokens == [100, 101, 102, 20]
    assert output.samples.rollout_prompt_ids == [1, 2]


@pytest.mark.asyncio
async def test_legacy_rollout_uses_the_same_video_request_contract(monkeypatch):
    requests = []

    async def fake_post(url, payload, headers=None):
        requests.append(payload)
        return {
            "text": "answer",
            "meta_info": {
                "output_token_logprobs": [(-0.1, 20)],
                "finish_reason": {"type": "stop"},
            },
        }

    args = _rollout_args(
        ci_test=False,
        use_opd=False,
        sglang_router_policy="round_robin",
    )
    state = SimpleNamespace(args=args, processor=_VideoProcessor(), tokenizer=_VideoTokenizer())
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    sample = Sample(
        prompt="<video>describe it",
        multimodal_inputs={"videos": [object()]},
        multimodal_rollout_inputs={"video_data": ["video.mp4"]},
    )

    output = await sglang_rollout.generate(args, sample, {"max_new_tokens": 4})

    assert requests[0]["input_ids"] == [1, 2]
    assert requests[0]["video_data"] == ["video.mp4"]
    assert output.tokens == [100, 101, 102, 20]


@pytest.mark.asyncio
async def test_multi_turn_reuses_collapsed_video_prefix_and_appends_turn_tokens(monkeypatch):
    requests = []
    outputs = iter(
        [
            {
                "text": "tool call",
                "meta_info": {
                    "output_token_logprobs": [(-0.1, 20)],
                    "finish_reason": {"type": "stop"},
                },
            },
            {
                "text": "answer",
                "meta_info": {
                    "output_token_logprobs": [(-0.2, 21)],
                    "finish_reason": {"type": "stop"},
                },
            },
        ]
    )

    async def fake_post(url, payload):
        requests.append(payload)
        return next(outputs)

    class Parser:
        calls = 0

        def parse_non_stream(self, text):
            self.calls += 1
            return None, [object()] if self.calls == 1 else []

    async def fake_execute_tool_calls(tool_calls, execute_tool_function):
        return [{"role": "tool", "content": "observation"}]

    def fake_add_tool_response(sample, tool_messages, tokenizer):
        sample.tokens.append(30)
        sample.response += "observation"
        sample.response_length += 1
        sample.loss_mask.append(0)
        sample.rollout_log_probs.append(0.0)

    monkeypatch.setattr(multi_turn, "post", fake_post)
    monkeypatch.setattr(multi_turn, "create_tool_call_parser", lambda *args: Parser())
    monkeypatch.setattr(multi_turn, "execute_tool_calls", fake_execute_tool_calls)
    monkeypatch.setattr(multi_turn, "update_sample_with_tool_responses", fake_add_tool_response)
    monkeypatch.setattr(multi_turn, "load_function", lambda path: [] if path == "tools" else object())

    args = _rollout_args(
        generate_execute_tool_function_path="execute",
        generate_tool_specs_path="tools",
        generate_tool_call_parser="fake",
        generate_max_turns=2,
        generate_multi_samples=False,
    )
    state = SimpleNamespace(args=args, processor=_VideoProcessor(), tokenizer=_VideoTokenizer())
    sample = Sample(
        prompt="<video>describe it",
        multimodal_inputs={"videos": [object()]},
        multimodal_rollout_inputs={"video_data": ["video.mp4"]},
    )

    output = await multi_turn.generate(
        GenerateFnInput(
            state=state,
            sample=sample,
            sampling_params={"max_new_tokens": 4},
            evaluation=False,
        )
    )

    assert [request["input_ids"] for request in requests] == [[1, 2], [1, 2, 20, 30]]
    assert [request["video_data"] for request in requests] == [["video.mp4"], ["video.mp4"]]
    assert output.samples.tokens == [100, 101, 102, 20, 30, 21]


@pytest.mark.asyncio
async def test_multi_turn_multi_sample_output_keeps_canonical_video_tokens(monkeypatch):
    requests = []

    async def fake_post(url, payload):
        requests.append(payload)
        return {
            "text": "answer",
            "meta_info": {
                "output_token_logprobs": [(-0.1, 20)],
                "finish_reason": {"type": "stop"},
            },
        }

    class Parser:
        def parse_non_stream(self, text):
            return None, []

    monkeypatch.setattr(multi_turn, "post", fake_post)
    monkeypatch.setattr(multi_turn, "create_tool_call_parser", lambda *args: Parser())
    monkeypatch.setattr(multi_turn, "load_function", lambda path: [] if path == "tools" else object())

    args = _rollout_args(
        generate_execute_tool_function_path="execute",
        generate_tool_specs_path="tools",
        generate_tool_call_parser="fake",
        generate_max_turns=1,
        generate_multi_samples=True,
    )
    state = SimpleNamespace(args=args, processor=_VideoProcessor(), tokenizer=_VideoTokenizer())
    sample = Sample(
        prompt="<video>describe it",
        multimodal_inputs={"videos": [object()]},
        multimodal_rollout_inputs={"video_data": ["video.mp4"]},
    )

    output = await multi_turn.generate(
        GenerateFnInput(
            state=state,
            sample=sample,
            sampling_params={"max_new_tokens": 4},
            evaluation=False,
        )
    )

    assert requests[0]["input_ids"] == [1, 2]
    assert output.samples[0].tokens == [100, 101, 102, 20]
    assert output.samples[0].rollout_prompt_ids == [1, 2]
