# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU configuration tests without model/worker registration fixtures."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig, VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode

from vllm_ascend.platform import NPUPlatform, _setup_compile_backend


def _make_pp_capture_config(
    tp_size=4,
    pp_size=2,
    max_num_seqs=20,
    max_num_batched_tokens=8192,
    capture_sizes=None,
    capture_max=None,
    speculative_tokens=0,
    enforce_eager=False,
    graph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(),
        model_config=SimpleNamespace(),
        performance_mode="throughput",
    )
    compilation_kwargs = {"mode": CompilationMode.DYNAMO_TRACE_ONCE, "cudagraph_mode": graph_mode}
    if capture_sizes is not None:
        compilation_kwargs["cudagraph_capture_sizes"] = capture_sizes
    if capture_max is not None:
        compilation_kwargs["max_cudagraph_capture_size"] = capture_max
    config.compilation_config = CompilationConfig(**compilation_kwargs)
    config.parallel_config.tensor_parallel_size = tp_size
    config.parallel_config.pipeline_parallel_size = pp_size
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    config.model_config.enforce_eager = enforce_eager
    config.uniform_decode_query_len = speculative_tokens + 1
    config.num_speculative_tokens = speculative_tokens
    config.speculative_config = (
        SimpleNamespace(num_speculative_tokens=speculative_tokens, uses_dynamic_speculative_decoding=lambda: False)
        if speculative_tokens
        else None
    )
    config.lora_config = None
    config.update_sizes_for_sequence_parallelism = lambda sizes: [size for size in sizes if size % tp_size == 0]
    config.additional_config = {"ascend_compilation_config": {"enable_npugraph_ex": True}}
    config._set_cudagraph_sizes = lambda: VllmConfig._set_cudagraph_sizes(config)
    return config


def _initialize_pp_capture_sizes(config):
    NPUPlatform.apply_config_platform_defaults(config)
    config._set_cudagraph_sizes()
    with patch("vllm_ascend.platform.enable_sp", return_value=False):
        _setup_compile_backend(config, compile_backend="test_backend", enable_dsa_cp=True)
    return config.compilation_config.cudagraph_capture_sizes


@pytest.mark.parametrize(
    ("tp_size", "pp_size", "max_num_seqs", "token_budget", "expected"),
    [
        (4, 2, 8, 8192, [4, 8]),
        (4, 2, 12, 8192, [4, 8, 12]),
        (4, 2, 13, 8192, [4, 8, 12]),
        (4, 2, 20, 8192, [4, 8, 12, 16, 20]),
        (4, 4, 20, 8192, [4, 8, 12, 16, 20]),
        (2, 2, 10, 8192, [2, 4, 6, 8, 10]),
        (8, 2, 20, 8192, [8, 16]),
        (4, 1, 20, 8192, [4, 8, 16, 20]),
        (8, 1, 20, 8192, [8, 16]),
        (4, 2, 20, 10, [4, 8]),
    ],
)
def test_pp_default_capture_sizes_cover_tp_multiples(tp_size, pp_size, max_num_seqs, token_budget, expected):
    config = _make_pp_capture_config(
        tp_size=tp_size, pp_size=pp_size, max_num_seqs=max_num_seqs, max_num_batched_tokens=token_budget
    )
    assert _initialize_pp_capture_sizes(config) == expected


def test_pp_capture_sizes_preserve_explicit_graph_list():
    config = _make_pp_capture_config(capture_sizes=[4, 8, 16, 20])
    assert _initialize_pp_capture_sizes(config) == [4, 8, 16, 20]


def test_pp_capture_sizes_preserve_explicit_capture_ceiling():
    config = _make_pp_capture_config(capture_max=16)
    assert _initialize_pp_capture_sizes(config) == [4, 8, 16]


def test_pp_capture_sizes_leave_speculative_graphs_unchanged():
    pp_config = _make_pp_capture_config(pp_size=2, speculative_tokens=2)
    tp_config = _make_pp_capture_config(pp_size=1, speculative_tokens=2)
    assert _initialize_pp_capture_sizes(pp_config) == _initialize_pp_capture_sizes(tp_config)


@pytest.mark.parametrize(
    "enforce_eager,graph_mode", [(True, CUDAGraphMode.FULL_DECODE_ONLY), (False, CUDAGraphMode.NONE)]
)
def test_pp_capture_sizes_stay_empty_when_graphs_are_disabled(enforce_eager, graph_mode):
    config = _make_pp_capture_config(enforce_eager=enforce_eager, graph_mode=graph_mode)
    assert _initialize_pp_capture_sizes(config) == []


def test_pp_default_capture_sizes_are_idempotent():
    config = _make_pp_capture_config()
    expected = _initialize_pp_capture_sizes(config)
    with patch("vllm_ascend.platform.enable_sp", return_value=False):
        _setup_compile_backend(config, compile_backend="test_backend", enable_dsa_cp=True)
    assert config.compilation_config.cudagraph_capture_sizes == expected
    assert not hasattr(config.compilation_config, "reduced_cg_cap")


