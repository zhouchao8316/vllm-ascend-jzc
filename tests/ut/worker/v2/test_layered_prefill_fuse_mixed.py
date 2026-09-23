"""Unit tests for fused mixed-batch frontier concat / split."""

import numpy as np
import torch

from vllm.v1.core.layered_prefill import LayeredPrefillStateStore
from vllm_ascend.worker.v2.layered_prefill import (
    concat_req_frontiers,
    pad_activation_rows,
    slice_req_activations,
    store_req_frontiers,
)


def test_store_and_concat_req_frontiers_roundtrip():
    store = LayeredPrefillStateStore()
    hidden = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    residual = hidden + 0.5
    query_start_loc = np.array([0, 1, 6], dtype=np.int32)
    req_ids = ["d0", "p0"]
    store_req_frontiers(
        store, req_ids, query_start_loc, hidden, residual, next_group_id=1
    )
    d_front = store.get("d0")
    p_front = store.get("p0")
    assert d_front is not None and p_front is not None
    assert d_front.query_len == 1
    assert p_front.query_len == 5
    assert d_front.group_id == 1
    restored, restored_res = concat_req_frontiers(
        store, req_ids, expected_group_id=1, num_tokens_padded=8
    )
    assert restored.shape[0] == 8
    assert torch.equal(restored[:6], hidden)
    assert restored_res is not None
    assert torch.equal(restored_res[:6], residual)
    assert torch.equal(restored[6:], torch.zeros(2, 4))


def test_slice_req_activations_keeps_decode_rows():
    hidden = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3)
    residual = hidden + 1
    query_start_loc = np.array([0, 1, 6], dtype=np.int32)
    d_h, d_r = slice_req_activations(
        ["d0", "p0"],
        ["d0"],
        query_start_loc,
        hidden,
        residual,
    )
    assert d_h.shape == (1, 3)
    assert torch.equal(d_h, hidden[:1])
    assert d_r is not None
    assert torch.equal(d_r, residual[:1])
    padded, padded_r = pad_activation_rows(d_h, d_r, 4)
    assert padded.shape[0] == 4
    assert torch.equal(padded[:1], d_h)
    assert padded_r is not None
    assert torch.equal(padded[1:], torch.zeros(3, 3))


def test_store_keep_ids_uses_full_batch_loc():
    store = LayeredPrefillStateStore()
    hidden = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    query_start_loc = np.array([0, 1, 6], dtype=np.int32)
    store_req_frontiers(
        store,
        ["p0"],
        query_start_loc,
        hidden,
        None,
        next_group_id=2,
        batch_req_ids=["d0", "p0"],
        keep_ids=["p0"],
    )
    p_front = store.get("p0")
    assert p_front is not None
    assert p_front.query_len == 5
    assert torch.equal(p_front.hidden_states, hidden[1:6])
    assert store.get("d0") is None


def test_store_keeps_trailing_pad_rows_on_last_request():
    store = LayeredPrefillStateStore()
    hidden = torch.arange(10 * 4 * 3, dtype=torch.float32).reshape(10, 4, 3)
    query_start_loc = np.array([0, 3, 8], dtype=np.int32)
    req_ids = ["a", "b"]
    store_req_frontiers(
        store, req_ids, query_start_loc, hidden, None, next_group_id=1
    )
    first = store.get("a")
    last = store.get("b")
    assert first is not None and last is not None
    assert first.query_len == 3
    assert last.query_len == 5
    assert first.hidden_states.shape == (3, 4, 3)
    assert last.hidden_states.shape == (7, 4, 3)
    assert torch.equal(first.hidden_states, hidden[:3])
    assert torch.equal(last.hidden_states, hidden[3:])
    restored, restored_res = concat_req_frontiers(
        store, req_ids, expected_group_id=1, num_tokens_padded=10
    )
    assert restored_res is None
    assert restored.shape == (10, 4, 3)
    assert torch.equal(restored, hidden)


def test_pad_activation_rows_pads_token_dim_for_rank3():
    hidden = torch.ones(5, 4, 3)
    padded, residual = pad_activation_rows(hidden, None, 8)
    assert residual is None
    assert padded.shape == (8, 4, 3)
    assert torch.equal(padded[:5], hidden)
    assert torch.equal(padded[5:], torch.zeros(3, 4, 3))
