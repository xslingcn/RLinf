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

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .bundle import BundleSpec

_SOURCE_PATTERN = re.compile(r"^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)(?:\[(?P<index>\d+)\])?$")
_DEFAULT_BINDINGS = {
    "observation.images.top": "main_images",
    "observation.images.left": "wrist_images[0]",
    "observation.images.right": "wrist_images[1]",
}


@dataclass(frozen=True)
class CameraSource:
    name: str
    index: int | None = None


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_camera_source(spec: str) -> CameraSource:
    match = _SOURCE_PATTERN.fullmatch(spec)
    if match is None:
        raise ValueError(
            f"Invalid camera binding {spec!r}. Expected forms like 'main_images' or 'wrist_images[0]'."
        )
    index = match.group("index")
    return CameraSource(
        name=match.group("name"),
        index=int(index) if index is not None else None,
    )


def _default_camera_bindings(spec: BundleSpec) -> dict[str, str]:
    if tuple(spec.image_feature_names) == (
        "observation.images.left",
        "observation.images.right",
        "observation.images.top",
    ):
        return dict(_DEFAULT_BINDINGS)
    raise ValueError(
        "camera_bindings is required for this LeRobot pi05 bundle because no safe default mapping exists."
    )


def resolve_camera_bindings(
    spec: BundleSpec,
    camera_bindings: dict[str, str] | None,
) -> dict[str, CameraSource]:
    raw_bindings = (
        _default_camera_bindings(spec)
        if not camera_bindings
        else {str(key): str(value) for key, value in camera_bindings.items()}
    )
    missing = set(spec.image_feature_names) - set(raw_bindings)
    extra = set(raw_bindings) - set(spec.image_feature_names)
    if missing:
        raise ValueError(
            f"camera_bindings is missing bundle image keys: {sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"camera_bindings contains unknown bundle image keys: {sorted(extra)}"
        )
    return {key: _parse_camera_source(value) for key, value in raw_bindings.items()}


def _get_batch_size(env_obs: dict[str, Any]) -> int:
    for key in ("states", "main_images", "wrist_images", "extra_view_images"):
        value = env_obs.get(key)
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return int(value.shape[0])
    task_descriptions = env_obs.get("task_descriptions")
    if task_descriptions is not None:
        return len(task_descriptions)
    raise ValueError("Cannot infer batch size from env_obs.")


def _get_task(env_obs: dict[str, Any], sample_idx: int) -> str:
    tasks = env_obs.get("task_descriptions")
    if tasks is None:
        return ""
    if isinstance(tasks, str):
        return tasks
    if sample_idx >= len(tasks):
        raise ValueError(
            f"task_descriptions has length {len(tasks)} but sample_idx={sample_idx}"
        )
    return str(tasks[sample_idx])


def _extract_state(
    env_obs: dict[str, Any], sample_idx: int, expected_shape: tuple[int, ...]
) -> np.ndarray:
    if "states" not in env_obs:
        raise ValueError("env_obs is missing 'states' required by the LeRobot bundle.")
    state = _as_numpy(env_obs["states"])
    if state.shape[0] <= sample_idx:
        raise ValueError(
            f"states batch has size {state.shape[0]} but sample_idx={sample_idx}"
        )
    sample_state = np.asarray(state[sample_idx], dtype=np.float32)
    if sample_state.shape != expected_shape:
        raise ValueError(
            f"Expected state shape {expected_shape}, got {sample_state.shape} for sample {sample_idx}"
        )
    return sample_state


def _normalize_image(sample: np.ndarray) -> np.ndarray:
    if sample.ndim != 3:
        raise ValueError(
            f"Expected an image with 3 dimensions (HWC or CHW), got shape {sample.shape}"
        )
    if sample.shape[0] in (1, 3, 4) and sample.shape[-1] not in (1, 3, 4):
        sample = np.moveaxis(sample, 0, -1)
    if sample.dtype == np.uint8:
        return sample
    if np.issubdtype(sample.dtype, np.floating) and sample.size > 0:
        sample = np.clip(sample, 0.0, 1.0) * 255.0 if sample.max() <= 1.0 else sample
    return np.clip(sample, 0.0, 255.0).astype(np.uint8)


def _extract_camera_sample(
    env_obs: dict[str, Any],
    source: CameraSource,
    sample_idx: int,
    missing_camera: str,
) -> np.ndarray:
    source_value = env_obs.get(source.name)
    if source_value is None:
        raise ValueError(
            f"env_obs is missing camera source {source.name!r} required by the LeRobot bundle."
        )

    array = _as_numpy(source_value)
    if array.shape[0] <= sample_idx:
        raise ValueError(
            f"{source.name} batch has size {array.shape[0]} but sample_idx={sample_idx}"
        )
    sample = array[sample_idx]

    if source.index is not None:
        if sample.ndim < 4:
            if source.index != 0:
                raise ValueError(
                    f"Camera source {source.name!r} does not expose multiple views but index {source.index} was requested."
                )
        else:
            if source.index >= sample.shape[0]:
                if missing_camera == "error":
                    raise ValueError(
                        f"Camera source {source.name!r} has {sample.shape[0]} views but index {source.index} was requested."
                    )
                raise ValueError(
                    f"Unsupported missing_camera={missing_camera!r}; only 'error' is implemented for faithful inference."
                )
            sample = sample[source.index]
    elif sample.ndim == 4:
        raise ValueError(
            f"Camera source {source.name!r} contains multiple views; specify an explicit index like '{source.name}[0]'."
        )

    return _normalize_image(sample)


def build_raw_frame(
    env_obs: dict[str, Any],
    sample_idx: int,
    spec: BundleSpec,
    camera_bindings: dict[str, CameraSource],
    missing_camera: str = "error",
) -> tuple[dict[str, np.ndarray], str]:
    raw_frame = {
        spec.state_feature_name: _extract_state(env_obs, sample_idx, spec.state_shape)
    }
    for bundle_key, source in camera_bindings.items():
        raw_frame[bundle_key] = _extract_camera_sample(
            env_obs=env_obs,
            source=source,
            sample_idx=sample_idx,
            missing_camera=missing_camera,
        )
    return raw_frame, _get_task(env_obs, sample_idx)


def build_raw_frames(
    env_obs: dict[str, Any],
    spec: BundleSpec,
    camera_bindings: dict[str, CameraSource],
    missing_camera: str = "error",
) -> list[tuple[dict[str, np.ndarray], str]]:
    batch_size = _get_batch_size(env_obs)
    return [
        build_raw_frame(
            env_obs=env_obs,
            sample_idx=sample_idx,
            spec=spec,
            camera_bindings=camera_bindings,
            missing_camera=missing_camera,
        )
        for sample_idx in range(batch_size)
    ]
