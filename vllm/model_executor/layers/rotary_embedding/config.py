# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration abstractions for RoPE."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RotaryEmbeddingConfig:
    base: float
    max_positions: int
    dimension: int
