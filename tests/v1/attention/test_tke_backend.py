# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TKE attention backend reorder_batch functionality."""

import pytest
import torch

from tests.v1.attention.utils import create_standard_kv_cache_spec, create_vllm_config
from vllm.v1.attention.backends.tke import TkeMetadataBuilder
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput


class MockInputBatch:
    """Mock InputBatch for testing reorder_batch method."""

    def __init__(self, req_ids: list[str]):
        self.req_ids = req_ids
        self._swapped_pairs = []  # Track swaps for verification

    def swap_states(self, idx1: int, idx2: int) -> None:
        """Mock implementation that records swaps."""
        # Swap the req_ids
        self.req_ids[idx1], self.req_ids[idx2] = self.req_ids[idx2], self.req_ids[idx1]
        self._swapped_pairs.append((idx1, idx2))


@pytest.fixture
def tke_metadata_builder():
    """Create a TKE metadata builder for testing."""
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    # Use a non-gated model that's publicly available
    vllm_config = create_vllm_config(model_name="facebook/opt-125m")
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    return TkeMetadataBuilder(kv_cache_spec, vllm_config, device)


def test_reorder_batch_no_reordering_needed(tke_metadata_builder):
    """Test case where decode sequences are already at the front."""
    # Create input batch where decode sequences (1 token) are already at front
    req_ids = ["req1", "req2", "req3", "req4"]
    input_batch = MockInputBatch(req_ids.copy())

    # Create scheduler output with decode sequences first
    num_scheduled_tokens = {
        "req1": 1,  # decode
        "req2": 1,  # decode
        "req3": 5,  # prefill
        "req4": 8,  # prefill
    }
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=15,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    # Call reorder_batch
    result = tke_metadata_builder.reorder_batch(input_batch, scheduler_output)

    # Should return False since no reordering was needed
    assert result is False
    # No swaps should have occurred
    assert len(input_batch._swapped_pairs) == 0
    # Order should remain unchanged
    assert input_batch.req_ids == ["req1", "req2", "req3", "req4"]

    # Verify internal state
    assert tke_metadata_builder._num_generation_sequences == 2
    assert tke_metadata_builder._num_generation_tokens == 2
    assert tke_metadata_builder._num_context_sequences == 2
    assert tke_metadata_builder._num_context_tokens == 13


def test_reorder_batch_simple_reordering(tke_metadata_builder):
    """Test simple case where some reordering is needed."""
    # Create input batch where prefill sequences are mixed with decode
    req_ids = ["req1", "req2", "req3", "req4"]
    input_batch = MockInputBatch(req_ids.copy())

    # Create scheduler output with mixed pattern
    num_scheduled_tokens = {
        "req1": 5,  # prefill
        "req2": 1,  # decode
        "req3": 8,  # prefill
        "req4": 1,  # decode
    }
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=15,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    # Call reorder_batch
    result = tke_metadata_builder.reorder_batch(input_batch, scheduler_output)

    # Should return True since reordering was performed
    assert result is True
    # Should have made some swaps
    assert len(input_batch._swapped_pairs) > 0

    # Verify decode sequences are now at the front
    # The exact order depends on the algorithm implementation
    decode_positions = []
    prefill_positions = []
    for i, req_id in enumerate(input_batch.req_ids):
        if num_scheduled_tokens[req_id] == 1:
            decode_positions.append(i)
        else:
            prefill_positions.append(i)

    # All decode positions should be before all prefill positions
    if decode_positions and prefill_positions:
        assert max(decode_positions) < min(prefill_positions)

    # Verify internal state
    assert tke_metadata_builder._num_generation_sequences == 2
    assert tke_metadata_builder._num_generation_tokens == 2
    assert tke_metadata_builder._num_context_sequences == 2
    assert tke_metadata_builder._num_context_tokens == 13


def test_reorder_batch_all_decode(tke_metadata_builder):
    """Test case with all decode sequences."""
    req_ids = ["req1", "req2", "req3"]
    input_batch = MockInputBatch(req_ids.copy())

    num_scheduled_tokens = {
        "req1": 1,  # decode
        "req2": 1,  # decode
        "req3": 1,  # decode
    }
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    result = tke_metadata_builder.reorder_batch(input_batch, scheduler_output)

    # No reordering needed since all are decode
    assert result is False
    assert len(input_batch._swapped_pairs) == 0

    # Verify internal state
    assert tke_metadata_builder._num_generation_sequences == 3
    assert tke_metadata_builder._num_generation_tokens == 3
    assert tke_metadata_builder._num_context_sequences == 0
    assert tke_metadata_builder._num_context_tokens == 0


def test_reorder_batch_all_prefill(tke_metadata_builder):
    """Test case with all prefill sequences."""
    req_ids = ["req1", "req2", "req3"]
    input_batch = MockInputBatch(req_ids.copy())

    num_scheduled_tokens = {
        "req1": 10,  # prefill
        "req2": 15,  # prefill
        "req3": 8,  # prefill
    }
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=33,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    result = tke_metadata_builder.reorder_batch(input_batch, scheduler_output)

    # No reordering needed since all are prefill
    assert result is False
    assert len(input_batch._swapped_pairs) == 0

    # Verify internal state
    assert tke_metadata_builder._num_generation_sequences == 0
    assert tke_metadata_builder._num_generation_tokens == 0
    assert tke_metadata_builder._num_context_sequences == 3
    assert tke_metadata_builder._num_context_tokens == 33


def test_reorder_batch_complex_pattern(tke_metadata_builder):
    """Test complex reordering pattern."""
    req_ids = ["req1", "req2", "req3", "req4", "req5", "req6"]
    input_batch = MockInputBatch(req_ids.copy())

    # Pattern: prefill, decode, prefill, decode, prefill, decode
    num_scheduled_tokens = {
        "req1": 10,  # prefill
        "req2": 1,  # decode
        "req3": 5,  # prefill
        "req4": 1,  # decode
        "req5": 12,  # prefill
        "req6": 1,  # decode
    }
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=30,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    result = tke_metadata_builder.reorder_batch(input_batch, scheduler_output)

    # Should return True since reordering was performed
    assert result is True

    # Verify decode sequences are at the front
    decode_count = 0
    prefill_count = 0
    decode_phase = True

    for req_id in input_batch.req_ids:
        tokens = num_scheduled_tokens[req_id]
        if tokens == 1:  # decode
            decode_count += 1
            assert decode_phase, (
                f"Found decode sequence {req_id} after prefill sequences"
            )
        else:  # prefill
            prefill_count += 1
            decode_phase = False

    # Verify counts
    assert decode_count == 3
    assert prefill_count == 3

    # Verify internal state
    assert tke_metadata_builder._num_generation_sequences == 3
    assert tke_metadata_builder._num_generation_tokens == 3
    assert tke_metadata_builder._num_context_sequences == 3
    assert tke_metadata_builder._num_context_tokens == 27


def test_reorder_batch_empty_batch(tke_metadata_builder):
    """Test edge case with empty batch."""
    req_ids = []
    input_batch = MockInputBatch(req_ids)

    num_scheduled_tokens = {}
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_input_ids=[],
        structured_output_request_ids={},
        grammar_bitmask=None,
    )

    result = tke_metadata_builder.reorder_batch(input_batch, scheduler_output)

    # No reordering needed for empty batch
    assert result is False
    assert len(input_batch._swapped_pairs) == 0

    # Verify internal state
    assert tke_metadata_builder._num_generation_sequences == 0
    assert tke_metadata_builder._num_generation_tokens == 0
    assert tke_metadata_builder._num_context_sequences == 0
    assert tke_metadata_builder._num_context_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__])
