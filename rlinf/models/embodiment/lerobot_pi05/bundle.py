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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BundleSpec:
    model_path: Path
    policy_type: str
    task_key: str
    state_feature_name: str
    state_shape: tuple[int, ...]
    action_feature_name: str
    action_shape: tuple[int, ...]
    image_feature_names: tuple[str, ...]
    image_shapes: dict[str, tuple[int, ...]]
    image_resolution: tuple[int, int]
    chunk_size: int
    n_action_steps: int
    num_inference_steps: int
    input_features: dict[str, dict[str, Any]]
    output_features: dict[str, dict[str, Any]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _find_step_config(
    processor_spec: dict[str, Any], registry_name: str
) -> dict[str, Any] | None:
    for step in processor_spec.get("steps", []):
        if step.get("registry_name") == registry_name:
            return step.get("config", {})
    return None


def load_bundle_spec(model_path: str | Path) -> BundleSpec:
    model_path = Path(model_path).expanduser().resolve()
    config_path = model_path / "config.json"
    preprocessor_path = model_path / "policy_preprocessor.json"
    postprocessor_path = model_path / "policy_postprocessor.json"

    if not config_path.exists():
        raise FileNotFoundError(f"LeRobot bundle is missing {config_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"LeRobot bundle is missing {preprocessor_path}")
    if not postprocessor_path.exists():
        raise FileNotFoundError(f"LeRobot bundle is missing {postprocessor_path}")

    config = _load_json(config_path)
    if config.get("type") != "pi05":
        raise ValueError(
            f"Only LeRobot pi05 bundles are supported, got type={config.get('type')!r}"
        )

    preprocessor = _load_json(preprocessor_path)
    tokenizer_cfg = _find_step_config(preprocessor, "tokenizer_processor") or {}
    task_key = tokenizer_cfg.get("task_key", "task")

    input_features = config["input_features"]
    output_features = config["output_features"]

    state_features = [
        (name, feature)
        for name, feature in input_features.items()
        if feature.get("type") == "STATE"
    ]
    image_features = [
        (name, feature)
        for name, feature in input_features.items()
        if feature.get("type") == "VISUAL"
    ]
    action_features = [
        (name, feature)
        for name, feature in output_features.items()
        if feature.get("type") == "ACTION"
    ]

    if len(state_features) != 1:
        raise ValueError(
            f"Expected exactly one state feature in {config_path}, got {len(state_features)}"
        )
    if len(action_features) != 1:
        raise ValueError(
            f"Expected exactly one action feature in {config_path}, got {len(action_features)}"
        )

    state_feature_name, state_feature = state_features[0]
    action_feature_name, action_feature = action_features[0]

    return BundleSpec(
        model_path=model_path,
        policy_type=config["type"],
        task_key=task_key,
        state_feature_name=state_feature_name,
        state_shape=tuple(state_feature["shape"]),
        action_feature_name=action_feature_name,
        action_shape=tuple(action_feature["shape"]),
        image_feature_names=tuple(name for name, _ in image_features),
        image_shapes={
            name: tuple(feature["shape"]) for name, feature in image_features
        },
        image_resolution=tuple(config["image_resolution"]),
        chunk_size=int(config["chunk_size"]),
        n_action_steps=int(config["n_action_steps"]),
        num_inference_steps=int(config["num_inference_steps"]),
        input_features=input_features,
        output_features=output_features,
    )
