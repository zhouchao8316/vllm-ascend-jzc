# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Layered Prefill helpers for Ascend Model Runner V2.

See /home/jzc/gjc/layered_prefill_v2_migration_plan.md §§3–4.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu.model_runner import ExecuteModelState

if TYPE_CHECKING:
    from vllm.v1.core.layered_prefill import LayeredPrefillPlan


@dataclass
class LayeredV2ExecuteModelState:
    """Combined D/P execute state handed to sample_tokens.

    ``sample_p`` is False for intermediate layer groups so the worker skips
    sampler / ``postprocess_num_computed_tokens`` for the P rows (plan §3.3).
    """

    scheduler_output: SchedulerOutput
    d_state: ExecuteModelState | None
    p_state: ExecuteModelState | None
    sample_p: bool


def detach_execute_model_state(state: ExecuteModelState) -> ExecuteModelState:
    """Clone tensors that P forward may overwrite in shared model workspaces.

    ``input_buffers`` are already isolated via a second AscendInputBuffers set.
    Model activations / aux hidden states are not: the P sub-batch reuses the
    same layer scratch as D, so sampling D after P without a clone reads P's
    last write (first token OK on P-only steps, then decode collapses to
    commas / noise under true P+D).
    """
    hidden = state.hidden_states
    aux = state.aux_hidden_states
    return state._replace(
        hidden_states=None if hidden is None else hidden.clone(),
        aux_hidden_states=(
            None if aux is None else [tensor.clone() for tensor in aux]
        ),
    )


def subset_cached_request_data(
    data: CachedRequestData, req_ids: list[str]
) -> CachedRequestData:
    indices = [data.req_ids.index(req_id) for req_id in req_ids if req_id in data.req_ids]
    selected_ids = [data.req_ids[index] for index in indices]

    def aligned(values: list) -> list:
        return [values[index] for index in indices] if len(values) == len(data.req_ids) else []

    return CachedRequestData(
        req_ids=selected_ids,
        resumed_req_ids=data.resumed_req_ids.intersection(selected_ids),
        new_token_ids=aligned(data.new_token_ids),
        all_token_ids={
            req_id: data.all_token_ids[req_id]
            for req_id in selected_ids
            if req_id in data.all_token_ids
        },
        new_block_ids=aligned(data.new_block_ids),
        num_computed_tokens=aligned(data.num_computed_tokens),
        num_output_tokens=aligned(data.num_output_tokens),
    )


def subset_scheduler_output(
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    *,
    layered_plan: Any = None,
    include_one_time_updates: bool = True,
) -> SchedulerOutput:
    """Crop a SchedulerOutput to ``req_ids``.

    One-time lifecycle fields (finished / preempted / connector / block zeroing)
    must appear on exactly one sub-batch per scheduler step.
    """
    req_id_set = set(req_ids)
    num_scheduled_tokens = {
        req_id: scheduler_output.num_scheduled_tokens[req_id]
        for req_id in req_ids
        if req_id in scheduler_output.num_scheduled_tokens
    }
    scheduled_new_reqs = [
        data for data in scheduler_output.scheduled_new_reqs if data.req_id in req_id_set
    ]
    scheduled_cached_reqs = subset_cached_request_data(
        scheduler_output.scheduled_cached_reqs, req_ids
    )
    scheduled_spec_decode_tokens = {
        req_id: tokens
        for req_id, tokens in scheduler_output.scheduled_spec_decode_tokens.items()
        if req_id in req_id_set
    }
    scheduled_encoder_inputs = {
        req_id: inputs
        for req_id, inputs in scheduler_output.scheduled_encoder_inputs.items()
        if req_id in req_id_set
    }
    num_invalid_spec_tokens = None
    if scheduler_output.num_invalid_spec_tokens is not None:
        num_invalid_spec_tokens = {
            req_id: value
            for req_id, value in scheduler_output.num_invalid_spec_tokens.items()
            if req_id in req_id_set
        }
    partial_tail_offloads = None
    if scheduler_output.partial_tail_offloads is not None:
        partial_tail_offloads = {
            req_id: value
            for req_id, value in scheduler_output.partial_tail_offloads.items()
            if req_id in req_id_set
        }
    return replace(
        scheduler_output,
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=scheduled_cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        scheduled_encoder_input_stats=(
            scheduler_output.scheduled_encoder_input_stats
            if include_one_time_updates
            else None
        ),
        finished_req_ids=(
            scheduler_output.finished_req_ids if include_one_time_updates else set()
        ),
        preempted_req_ids=(
            scheduler_output.preempted_req_ids if include_one_time_updates else set()
        ),
        free_encoder_mm_hashes=(
            scheduler_output.free_encoder_mm_hashes if include_one_time_updates else []
        ),
        new_block_ids_to_zero=(
            scheduler_output.new_block_ids_to_zero if include_one_time_updates else None
        ),
        kv_cache_block_copies=(
            scheduler_output.kv_cache_block_copies if include_one_time_updates else None
        ),
        kv_connector_metadata=(
            scheduler_output.kv_connector_metadata if include_one_time_updates else None
        ),
        ec_connector_metadata=(
            scheduler_output.ec_connector_metadata if include_one_time_updates else None
        ),
        ec_manager_metadata=(
            scheduler_output.ec_manager_metadata if include_one_time_updates else None
        ),
        partial_tail_offloads=partial_tail_offloads,
        num_invalid_spec_tokens=num_invalid_spec_tokens,
        layered_prefill_plan=layered_plan,
    )


def split_d_p_req_ids(
    scheduler_output: SchedulerOutput,
    plan: "LayeredPrefillPlan",
) -> tuple[list[str], list[str]]:
    all_req_ids = list(scheduler_output.num_scheduled_tokens)
    p_req_ids = list(plan.prefill_req_ids)
    if len(p_req_ids) != 1:
        raise RuntimeError("Layered prefill Phase 1 supports exactly one P request")
    p_req_set = set(p_req_ids)
    if not p_req_set.issubset(all_req_ids):
        raise RuntimeError("Layered plan contains an unscheduled P request")
    d_req_ids = [req_id for req_id in all_req_ids if req_id not in p_req_set]
    return d_req_ids, p_req_ids
