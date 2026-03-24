# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rlinf.models.embodiment.lerobot_pi05 import (
    BundleSpec,
    LeRobotPI05ActionModel,
    build_raw_frames,
    load_bundle_spec,
    resolve_camera_bindings,
)


def _write_bundle(
    tmp_path: Path,
    *,
    image_feature_names: tuple[str, ...] = (
        "observation.images.left",
        "observation.images.right",
        "observation.images.top",
    ),
) -> Path:
    config = {
        "type": "pi05",
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [14]},
            **{
                name: {"type": "VISUAL", "shape": [3, 360, 640]}
                for name in image_feature_names
            },
        },
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
        "image_resolution": [224, 224],
        "chunk_size": 50,
        "n_action_steps": 50,
        "num_inference_steps": 10,
    }
    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {"registry_name": "rename_observations_processor", "config": {}},
            {
                "registry_name": "tokenizer_processor",
                "config": {"task_key": "task", "max_length": 200},
            },
        ],
    }
    postprocessor = {
        "name": "policy_postprocessor",
        "steps": [{"registry_name": "device_processor", "config": {"device": "cpu"}}],
    }

    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(preprocessor))
    (tmp_path / "policy_postprocessor.json").write_text(json.dumps(postprocessor))
    return tmp_path


def _make_bundle_spec(tmp_path: Path) -> BundleSpec:
    return load_bundle_spec(_write_bundle(tmp_path))


def test_load_bundle_spec_reads_pi05_bundle_metadata(tmp_path: Path):
    spec = _make_bundle_spec(tmp_path)

    assert spec.policy_type == "pi05"
    assert spec.task_key == "task"
    assert spec.state_feature_name == "observation.state"
    assert spec.state_shape == (14,)
    assert spec.action_feature_name == "action"
    assert spec.action_shape == (14,)
    assert spec.image_feature_names == (
        "observation.images.left",
        "observation.images.right",
        "observation.images.top",
    )
    assert spec.chunk_size == 50
    assert spec.num_inference_steps == 10


def test_resolve_camera_bindings_uses_safe_default_for_top_left_right_bundle(
    tmp_path: Path,
):
    spec = _make_bundle_spec(tmp_path)

    bindings = resolve_camera_bindings(spec, camera_bindings=None)

    assert bindings["observation.images.top"].name == "main_images"
    assert bindings["observation.images.left"].name == "wrist_images"
    assert bindings["observation.images.left"].index == 0
    assert bindings["observation.images.right"].index == 1


def test_resolve_camera_bindings_requires_explicit_mapping_for_unknown_layout(
    tmp_path: Path,
):
    model_path = _write_bundle(
        tmp_path,
        image_feature_names=("observation.images.front", "observation.images.side"),
    )
    spec = load_bundle_spec(model_path)

    with pytest.raises(ValueError, match="camera_bindings is required"):
        resolve_camera_bindings(spec, camera_bindings=None)


def test_build_raw_frames_maps_rlinf_canonical_observation_to_lerobot_inputs(
    tmp_path: Path,
):
    spec = _make_bundle_spec(tmp_path)
    camera_bindings = resolve_camera_bindings(spec, None)

    env_obs = {
        "states": torch.arange(28, dtype=torch.float32).reshape(2, 14),
        "main_images": torch.full((2, 4, 5, 3), 10, dtype=torch.uint8),
        "wrist_images": torch.stack(
            [
                torch.full((2, 4, 5, 3), 20, dtype=torch.uint8),
                torch.full((2, 4, 5, 3), 30, dtype=torch.uint8),
            ],
            dim=1,
        ),
        "task_descriptions": ["fold towel", "flatten towel"],
    }

    raw_frames = build_raw_frames(env_obs, spec, camera_bindings)

    assert len(raw_frames) == 2
    first_frame, first_task = raw_frames[0]
    second_frame, second_task = raw_frames[1]
    assert first_task == "fold towel"
    assert second_task == "flatten towel"
    assert np.allclose(first_frame["observation.state"], np.arange(14, dtype=np.float32))
    assert np.allclose(
        second_frame["observation.state"], np.arange(14, 28, dtype=np.float32)
    )
    assert np.all(first_frame["observation.images.top"] == 10)
    assert np.all(first_frame["observation.images.left"] == 20)
    assert np.all(first_frame["observation.images.right"] == 30)


def test_default_forward_raises_for_inference_only_model():
    model = object.__new__(LeRobotPI05ActionModel)

    with pytest.raises(NotImplementedError, match="Training/logprob recomputation"):
        LeRobotPI05ActionModel.default_forward(model)


def test_predict_action_batch_uses_per_sample_preprocess_and_chunk_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    spec = _make_bundle_spec(tmp_path)
    model = object.__new__(LeRobotPI05ActionModel)
    nn.Module.__init__(model)

    class _FakePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(device="cpu")
            self.seen_batch = None
            self.seen_num_steps = None

        def predict_action_chunk(self, batch, num_steps=None):
            self.seen_batch = batch
            self.seen_num_steps = num_steps
            batch_size = batch["processed_state"].shape[0]
            return torch.arange(batch_size * 4 * 14, dtype=torch.float32).reshape(
                batch_size, 4, 14
            )

    fake_policy = _FakePolicy()
    model.bundle_spec = spec
    model.camera_bindings = resolve_camera_bindings(spec, None)
    model.missing_camera = "error"
    model.override_num_inference_steps = 7
    model.truncate_action_chunk_to = 2
    model.policy = fake_policy
    model.preprocessor = lambda prepared: {
        "processed_state": prepared["observation.state"] + 1.0
    }
    model.postprocessor = lambda action_chunk: action_chunk * 2.0

    def _fake_prepare(raw_frame, device, task):
        del device, task
        return {
            "observation.state": torch.from_numpy(raw_frame["observation.state"]).unsqueeze(0)
        }

    monkeypatch.setattr(
        "lerobot.policies.utils.prepare_observation_for_inference",
        _fake_prepare,
    )

    env_obs = {
        "states": torch.arange(28, dtype=torch.float32).reshape(2, 14),
        "main_images": torch.full((2, 4, 5, 3), 10, dtype=torch.uint8),
        "wrist_images": torch.stack(
            [
                torch.full((2, 4, 5, 3), 20, dtype=torch.uint8),
                torch.full((2, 4, 5, 3), 30, dtype=torch.uint8),
            ],
            dim=1,
        ),
        "task_descriptions": ["fold towel", "flatten towel"],
    }

    actions, result = model.predict_action_batch(env_obs, mode="eval")

    assert fake_policy.seen_num_steps == 7
    assert tuple(fake_policy.seen_batch["processed_state"].shape) == (2, 14)
    assert tuple(actions.shape) == (2, 2, 14)
    assert torch.equal(actions[0, 0], torch.arange(14, dtype=torch.float32) * 2.0)
    assert result == {
        "prev_logprobs": None,
        "prev_values": None,
        "forward_inputs": {},
    }
