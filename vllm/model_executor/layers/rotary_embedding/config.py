# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration abstractions for RoPE."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SimpleRotaryEmbedding:
    base: float
    max_positions: int
    dimension: int


@dataclass(frozen=True)
class YarnScalingConfig:
    extended_max_positions: int
    beta_fast: float
    beta_slow: float
    scaling_factor: float
    extrapolation_factor: Optional[float] = None
    attn_factor: Optional[float] = None


@dataclass(frozen=True)
class YarnRotaryEmbeddingConfig:
    """
    Configuration for Yarn Rotary Embedding, made up of a base
    """
    rotary_embedding_config: SimpleRotaryEmbedding
    yarn_scaling_config: YarnScalingConfig


RotaryEmbeddingConfig = SimpleRotaryEmbedding | YarnRotaryEmbeddingConfig
