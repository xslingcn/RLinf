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

import numbers
from typing import Any

import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy

from .bundle import BundleSpec, load_bundle_spec
from .obs_adapter import build_raw_frames, resolve_camera_bindings


def _collate_processed_samples(samples: list[Any]) -> Any:
    first = samples[0]
    if torch.is_tensor(first):
        return torch.cat(samples, dim=0)
    if isinstance(first, dict):
        return {
            key: _collate_processed_samples([sample[key] for sample in samples])
            for key in first
        }
    if isinstance(first, list):
        return [item for sample in samples for item in sample]
    if isinstance(first, tuple):
        return tuple(
            _collate_processed_samples([sample[idx] for sample in samples])
            for idx in range(len(first))
        )
    if isinstance(first, (str, bytes, numbers.Number, bool)):
        if all(sample == first for sample in samples):
            return first
        return list(samples)
    if first is None:
        return None
    raise TypeError(f"Unsupported preprocessed batch value type: {type(first)!r}")


class LeRobotPI05ActionModel(nn.Module, BasePolicy):
    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        camera_bindings: dict[str, str] | None = None,
        missing_camera: str = "error",
        override_num_inference_steps: int | None = None,
        truncate_action_chunk_to: int | None = None,
    ):
        super().__init__()

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import PI05Policy

        self.bundle_spec: BundleSpec = load_bundle_spec(model_path)
        self.camera_bindings = resolve_camera_bindings(
            self.bundle_spec, camera_bindings
        )
        self.missing_camera = missing_camera
        self.override_num_inference_steps = override_num_inference_steps
        self.truncate_action_chunk_to = truncate_action_chunk_to

        policy_config = PreTrainedConfig.from_pretrained(model_path)
        policy_config.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy_config.use_amp = False

        self.policy = PI05Policy.from_pretrained(model_path, config=policy_config)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.bundle_spec.model_path),
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)}
            },
        )

    def _sync_preprocessor_device(self) -> None:
        device = str(next(self.policy.parameters()).device)
        self.policy.config.device = device
        for step in getattr(self.preprocessor, "steps", []):
            if step.__class__.__name__ == "DeviceProcessorStep" and hasattr(
                step, "device"
            ):
                step.device = device

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        self._sync_preprocessor_device()
        return model

    def default_forward(self, **kwargs):
        del kwargs
        raise NotImplementedError(
            "Training/logprob recomputation is not supported for lerobot_pi05 yet."
        )

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        compute_values: bool = False,
        num_steps: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del mode, compute_values, kwargs

        from lerobot.policies.utils import prepare_observation_for_inference

        raw_frames = build_raw_frames(
            env_obs=env_obs,
            spec=self.bundle_spec,
            camera_bindings=self.camera_bindings,
            missing_camera=self.missing_camera,
        )
        device = torch.device(str(self.policy.config.device))
        processed_samples = []
        for raw_frame, task in raw_frames:
            prepared = prepare_observation_for_inference(
                raw_frame,
                device=device,
                task=task,
            )
            processed_samples.append(self.preprocessor(prepared))

        batch = _collate_processed_samples(processed_samples)
        inference_steps = (
            num_steps
            if num_steps is not None
            else self.override_num_inference_steps
        )

        with torch.inference_mode():
            if inference_steps is None:
                action_chunk = self.policy.predict_action_chunk(batch)
            else:
                action_chunk = self.policy.predict_action_chunk(
                    batch, num_steps=inference_steps
                )
            action_chunk = self.postprocessor(action_chunk)

        if self.truncate_action_chunk_to is not None:
            action_chunk = action_chunk[:, : self.truncate_action_chunk_to]

        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {},
        }
        return action_chunk, result


def get_model(cfg, torch_dtype=None):
    del torch_dtype
    lerobot_cfg = cfg.get("lerobot_pi05", {}) or {}
    return LeRobotPI05ActionModel(
        model_path=cfg.model_path,
        device=lerobot_cfg.get("device"),
        camera_bindings=lerobot_cfg.get("camera_bindings"),
        missing_camera=lerobot_cfg.get("missing_camera", "error"),
        override_num_inference_steps=lerobot_cfg.get("override_num_inference_steps"),
        truncate_action_chunk_to=lerobot_cfg.get("truncate_action_chunk_to"),
    )
