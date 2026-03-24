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

from pathlib import Path

import torch
from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration

from rlinf.models.embodiment.openpi import (
    _get_model_weight_paths,
    _inject_missing_paligemma_embeddings,
    _normalize_openpi_state_dict_keys,
    _retie_paligemma_embeddings,
)


def _build_tiny_paligemma_model():
    config = PaliGemmaConfig()
    config.text_config.hidden_size = 16
    config.text_config.intermediate_size = 32
    config.text_config.num_hidden_layers = 1
    config.text_config.num_attention_heads = 2
    config.text_config.num_key_value_heads = 2
    config.text_config.head_dim = 8
    config.text_config.vocab_size = 32
    config.vision_config.hidden_size = 16
    config.vision_config.intermediate_size = 32
    config.vision_config.num_hidden_layers = 1
    config.vision_config.num_attention_heads = 2
    config.vision_config.image_size = 14
    config.vision_config.patch_size = 14
    config.hidden_size = 16
    config.projection_dim = 16
    config.vocab_size = 32
    return PaliGemmaForConditionalGeneration(config)


def test_normalize_openpi_state_dict_keys_strips_model_prefix():
    state_dict = {
        "model.foo": torch.randn(2, 2),
        "model.bar": torch.randn(2, 2),
    }

    normalized = _normalize_openpi_state_dict_keys(state_dict)

    assert set(normalized) == {"foo", "bar"}


def test_normalize_openpi_state_dict_keys_strips_compiled_prefix():
    state_dict = {
        "_orig_mod.foo": torch.randn(2, 2),
        "_orig_mod.bar": torch.randn(2, 2),
    }

    normalized = _normalize_openpi_state_dict_keys(state_dict)

    assert set(normalized) == {"foo", "bar"}


def test_get_model_weight_paths_prefers_model_weights(tmp_path: Path):
    (tmp_path / "policy_preprocessor.safetensors").touch()
    (tmp_path / "policy_postprocessor.safetensors").touch()
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "aux.safetensors").touch()

    weight_paths = _get_model_weight_paths(str(tmp_path))

    assert weight_paths == [str(tmp_path / "model.safetensors")]


def test_paligemma_embedding_repair_backfills_and_reties():
    state_dict = {
        "paligemma_with_expert.paligemma.lm_head.weight": torch.randn(8, 4),
    }
    repaired = _inject_missing_paligemma_embeddings(state_dict)
    assert (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        in repaired
    )
    assert torch.equal(
        repaired["paligemma_with_expert.paligemma.lm_head.weight"],
        repaired[
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        ],
    )

    paligemma = _build_tiny_paligemma_model()
    paligemma.model.language_model.embed_tokens = torch.nn.Embedding(32, 16)
    wrapper = type(
        "_FakeWrapper",
        (),
        {"paligemma_with_expert": type("_FakePali", (), {"paligemma": paligemma})()},
    )()

    _retie_paligemma_embeddings(wrapper)

    assert (
        wrapper.paligemma_with_expert.paligemma.lm_head.weight.data_ptr()
        == wrapper.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.data_ptr()
    )
