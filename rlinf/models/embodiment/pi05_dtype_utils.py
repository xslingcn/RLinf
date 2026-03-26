# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for PI05 mixed-dtype parameter handling."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

PI05_FLOAT32_PARAM_SELECTORS = (
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
    "value_head",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)


def _matches_any_selector(name: str, selectors: Iterable[str]) -> bool:
    return any(selector in name for selector in selectors)


def normalize_pi05_float32_parameters_for_training(
    model: nn.Module,
    target_dtype: torch.dtype,
) -> list[str]:
    """Cast only the known PI05 float32-stability params to the target dtype."""
    converted_param_names: list[str] = []
    for name, param in model.named_parameters():
        if (
            param.dtype.is_floating_point
            and param.dtype != target_dtype
            and _matches_any_selector(name, PI05_FLOAT32_PARAM_SELECTORS)
        ):
            param.data = param.data.to(dtype=target_dtype)
            converted_param_names.append(name)
    return converted_param_names


def list_pi05_params_with_unexpected_dtype(
    model: nn.Module,
    target_dtype: torch.dtype,
) -> list[tuple[str, torch.dtype]]:
    """Return floating PI05 params that still do not match the target dtype."""
    return [
        (name, param.dtype)
        for name, param in model.named_parameters()
        if param.dtype.is_floating_point
        and param.dtype != target_dtype
        and not _matches_any_selector(name, PI05_FLOAT32_PARAM_SELECTORS)
    ]
