"""Unit tests for Layered Prefill V2 sampling / frontier lifecycle (V3)."""

from unittest.mock import MagicMock

import torch

from vllm.v1.core.layered_prefill import LayeredFrontier, LayeredPrefillStateStore
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.model_runner import ExecuteModelState

from vllm_ascend.worker.v2.layered_prefill import (
    LayeredV2ExecuteModelState,
    detach_execute_model_state,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _empty_cached() -> CachedRequestData:
    return CachedRequestData.make_empty()


def _scheduler_output(num_scheduled_tokens: dict[str, int]) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=_empty_cached(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def test_merge_layered_v2_outputs_preserves_scheduler_order():
    d_out = ModelRunnerOutput(
        req_ids=["d0"],
        req_id_to_index={"d0": 0},
        sampled_token_ids=[[11]],
    )
    p_out = ModelRunnerOutput(
        req_ids=["p0"],
        req_id_to_index={"p0": 0},
        sampled_token_ids=[[22]],
    )
    merged = NPUModelRunner._merge_layered_v2_outputs(
        _scheduler_output({"p0": 8, "d0": 1}),
        [d_out, p_out],
    )
    assert merged.req_ids == ["p0", "d0"]
    assert merged.sampled_token_ids == [[22], [11]]


def test_merge_intermediate_p_emits_empty_sampled_tokens():
    d_out = ModelRunnerOutput(
        req_ids=["d0"],
        req_id_to_index={"d0": 0},
        sampled_token_ids=[[7]],
    )
    p_out = ModelRunnerOutput(
        req_ids=["p0"],
        req_id_to_index={"p0": 0},
        sampled_token_ids=[[]],
    )
    merged = NPUModelRunner._merge_layered_v2_outputs(
        _scheduler_output({"d0": 1, "p0": 8}),
        [d_out, p_out],
    )
    assert merged.sampled_token_ids == [[7], []]


def test_frontier_store_clears_finished_and_preempted():
    store = LayeredPrefillStateStore()
    hs = torch.zeros(2, 4)
    store.put(
        LayeredFrontier(
            req_id="a",
            group_id=1,
            query_len=2,
            hidden_states=hs,
            residual=None,
        )
    )
    store.put(
        LayeredFrontier(
            req_id="b",
            group_id=2,
            query_len=2,
            hidden_states=hs.clone(),
            residual=None,
        )
    )
    store.clear_many({"a"})
    assert store.get("a") is None
    assert store.get("b") is not None
    store.clear_many(["b"])
    assert store.get("b") is None


def test_detach_keeps_d_hidden_alive_after_alias_write():
    hidden = torch.arange(4, dtype=torch.float32)
    state = ExecuteModelState(
        input_batch=object(),  # type: ignore[arg-type]
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=hidden,
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    detached = detach_execute_model_state(state)
    hidden.fill_(-1)
    assert detached.hidden_states is not None
    assert detached.hidden_states.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_layered_v2_state_sample_p_flag():
    state = LayeredV2ExecuteModelState(
        scheduler_output=_scheduler_output({"p0": 4}),
        d_state=None,
        p_state=MagicMock(),
        sample_p=False,
    )
    assert state.sample_p is False
    assert isinstance(state.scheduler_output, SchedulerOutput)
