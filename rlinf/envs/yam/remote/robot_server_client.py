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

"""RobotServer gRPC client for the desktop remote robot runtime."""

from __future__ import annotations

import time
from typing import Optional

import grpc
import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.yam.remote.proto import robot_env_pb2, robot_env_pb2_grpc
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

_DEFAULT_MAX_MESSAGE_SIZE = 16 * 1024 * 1024


def _decompress_image(data: bytes, height: int, width: int) -> np.ndarray:
    """Decode JPEG bytes to uint8 HWC numpy array."""
    import cv2

    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img = img[..., ::-1].copy()
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    return img


def _decode_image(
    data: bytes, height: int, width: int, is_compressed: bool
) -> np.ndarray:
    """Decode one image payload into a uint8 HWC numpy array."""
    if is_compressed:
        return _decompress_image(data, height, width)
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def _decode_image_stack(
    image_payloads, height: int, width: int, is_compressed: bool
) -> np.ndarray | None:
    """Decode repeated image payloads into a batched NHWC image tensor."""
    if not image_payloads:
        return None
    images = [
        _decode_image(image_payload, height, width, is_compressed)
        for image_payload in image_payloads
    ]
    return np.stack(images, axis=0)[np.newaxis, :]


def _proto_to_obs(proto_obs: robot_env_pb2.Observation) -> dict:
    """Convert a protobuf Observation to an RLinf-compatible dict."""
    state_shape = tuple(proto_obs.state_shape)
    states = np.frombuffer(proto_obs.states, dtype=np.float32).reshape(state_shape)

    h, w = proto_obs.img_height, proto_obs.img_width
    img = _decode_image(proto_obs.main_image, h, w, proto_obs.is_compressed)
    img = img[np.newaxis, :]

    obs = {
        "states": to_tensor(states),
        "main_images": to_tensor(img),
        "task_descriptions": [proto_obs.task_description],
    }
    wrist_images = _decode_image_stack(
        proto_obs.wrist_images, h, w, proto_obs.is_compressed
    )
    if wrist_images is not None:
        obs["wrist_images"] = to_tensor(wrist_images)
    extra_view_images = _decode_image_stack(
        proto_obs.extra_view_images, h, w, proto_obs.is_compressed
    )
    if extra_view_images is not None:
        obs["extra_view_images"] = to_tensor(extra_view_images)
    return obs


class RobotServerClient:
    """Thin gRPC client for the desktop RobotServer."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        worker_info: Optional[WorkerInfo] = None,
    ) -> None:
        assert num_envs == 1, (
            f"RobotServerClient supports exactly 1 environment, got {num_envs}."
        )
        self._logger = get_logger()
        self.cfg = cfg
        self.num_envs = num_envs
        self.worker_info = worker_info

        server_url = str(cfg.get("remote_server_url", "localhost:50051"))
        max_msg = int(cfg.get("grpc_max_message_size", _DEFAULT_MAX_MESSAGE_SIZE))
        self._timeout = float(cfg.get("grpc_timeout", 30.0))

        channel_options = [
            ("grpc.max_send_message_length", max_msg),
            ("grpc.max_receive_message_length", max_msg),
        ]
        self._logger.info(f"[RobotServerClient] Connecting to server at {server_url}")
        self._channel = grpc.insecure_channel(server_url, options=channel_options)
        self._stub = robot_env_pb2_grpc.RobotEnvServiceStub(self._channel)

        connect_timeout = float(cfg.get("grpc_connect_timeout", 300.0))
        retry_interval = 5.0
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                spaces = self._stub.GetSpaces(
                    robot_env_pb2.Empty(), timeout=self._timeout
                )
                break
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.UNAVAILABLE:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                wait = min(retry_interval, remaining)
                self._logger.info(
                    "[RobotServerClient] Server not ready at %s, retrying in %.0fs "
                    "(%.0fs remaining of %.0fs connect timeout) ...",
                    server_url,
                    wait,
                    remaining,
                    connect_timeout,
                )
                time.sleep(wait)

        self._state_dim = spaces.state_dim
        self._action_dim = spaces.action_dim
        self._img_h = spaces.img_height
        self._img_w = spaces.img_width
        self._img_c = spaces.img_channels
        self._max_episode_steps = spaces.max_episode_steps
        self._control_rate_hz = spaces.control_rate_hz
        self.auto_reset = bool(cfg.get("auto_reset", spaces.auto_reset))
        self.ignore_terminations = bool(
            cfg.get("ignore_terminations", spaces.ignore_terminations)
        )

        self._task_description = str(cfg.get("task_description", ""))
        self._is_start = True
        self._num_steps = 0
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.last_obs: dict | None = None

        self._init_metrics()

        if self._task_description:
            self._stub.SetTaskDescription(
                robot_env_pb2.TaskDescriptionRequest(
                    task_description=self._task_description
                ),
                timeout=self._timeout,
            )

        self._logger.info(
            "[RobotServerClient] Connected. state_dim=%s, action_dim=%s, img=(%s,%s,%s)",
            self._state_dim,
            self._action_dim,
            self._img_h,
            self._img_w,
            self._img_c,
        )

    def reset(self, *, seed: int | None = None) -> tuple[dict, dict]:
        """Reset the remote robot episode and return the initial observation."""
        self._num_steps = 0
        self._elapsed_steps[:] = 0
        self._reset_metrics()
        self._is_start = True

        req = robot_env_pb2.ResetRequest()
        if seed is not None:
            req.seed = seed
        proto_obs = self._stub.Reset(req, timeout=self._timeout)
        obs = _proto_to_obs(proto_obs)
        obs["task_descriptions"] = [self._task_description]
        self.last_obs = obs
        return obs, {}

    def chunk_step(self, chunk_actions):
        """Execute one chunk of actions on the desktop RobotServer."""
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        num_envs, chunk_size, action_dim = chunk_actions.shape

        req = robot_env_pb2.ChunkStepRequest(
            actions=chunk_actions.tobytes(),
            num_envs=num_envs,
            chunk_size=chunk_size,
            action_dim=action_dim,
        )
        resp = self._stub.ChunkStep(req, timeout=self._timeout * chunk_size)

        obs_list = []
        rewards = []
        terminations = []
        truncations = []
        infos_list = []

        for sr in resp.step_results:
            obs = _proto_to_obs(sr.observation)
            obs["task_descriptions"] = [self._task_description]
            obs_list.append(obs)

            self._elapsed_steps += 1
            self._num_steps += 1

            step_reward = np.array([sr.reward], dtype=np.float32)
            step_term = np.array([sr.terminated], dtype=bool)
            step_trunc = np.array([sr.truncated], dtype=bool)

            infos = self._record_metrics(
                step_reward, step_term, np.zeros_like(step_term), {}
            )
            infos_list.append(infos)
            rewards.append(step_reward)
            terminations.append(step_term)
            truncations.append(step_trunc)

        chunk_rewards = torch.stack(
            [to_tensor(r) if not isinstance(r, torch.Tensor) else r for r in rewards],
            dim=1,
        )
        raw_chunk_terminations = torch.stack(
            [
                to_tensor(t) if not isinstance(t, torch.Tensor) else t
                for t in terminations
            ],
            dim=1,
        )
        raw_chunk_truncations = torch.stack(
            [
                to_tensor(t) if not isinstance(t, torch.Tensor) else t
                for t in truncations
            ],
            dim=1,
        )

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        if obs_list:
            self.last_obs = obs_list[-1]

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def close(self) -> None:
        """Close the remote robot session."""
        self._stub.Close(robot_env_pb2.Empty(), timeout=self._timeout)
        self._channel.close()

    def _init_metrics(self) -> None:
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)
        self.prev_step_reward = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None) -> None:
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0.0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, intervene_current_step, infos):
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info = {
            "success_once": self.success_once.copy(),
            "return": self.returns.copy(),
            "episode_len": self._elapsed_steps.copy(),
            "reward": np.where(
                self._elapsed_steps > 0, self.returns / self._elapsed_steps, 0.0
            ),
            "intervened_once": self.intervened_once,
            "intervened_steps": self.intervened_steps,
            "success_no_intervened": self.success_once.copy() & (~self.intervened_once),
        }
        infos["episode"] = to_tensor(episode_info)
        return infos

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = value

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def task_description(self) -> str:
        return self._task_description

    @task_description.setter
    def task_description(self, value: str) -> None:
        self._task_description = str(value)
        self._stub.SetTaskDescription(
            robot_env_pb2.TaskDescriptionRequest(
                task_description=self._task_description
            ),
            timeout=self._timeout,
        )

    @property
    def task_descriptions(self) -> list[str]:
        return [self._task_description]

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def control_rate_hz(self) -> float:
        return self._control_rate_hz

    def update_reset_state_ids(self) -> None:
        """No-op. Episode-state tracking is managed by the remote server."""
