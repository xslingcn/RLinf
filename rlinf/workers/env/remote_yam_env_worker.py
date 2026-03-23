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

"""Dedicated env-side worker for the marl-backed remote robot training path."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
)
from rlinf.envs.action_utils import prepare_actions
from rlinf.integrations.marl import MarlClient, MarlImage
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.envs.yam.remote.robot_server_client import RobotServerClient


class RemoteYamEnvWorker(Worker):
    """Dedicated env worker for remote robot execution plus marl orchestration."""

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.rollout_epoch = int(self.cfg.algorithm.get("rollout_epoch", 1))
        self.collect_transitions = bool(
            self.cfg.rollout.get("collect_transitions", False)
        )
        self.collect_prev_infos = bool(self.cfg.rollout.get("collect_prev_infos", True))
        self.stage_num = int(self.cfg.rollout.pipeline_stage_num)
        self.only_eval = bool(getattr(self.cfg.runner, "only_eval", False))
        self.enable_eval = bool(
            self.cfg.runner.val_check_interval > 0 or self.only_eval
        )
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self._validate_supported_cfg()

        self._marl_cfg = self.cfg.marl
        planner_cfg = self._marl_cfg.get("planner", {})
        topreward_cfg = self._marl_cfg.get("topreward", {})

        self._subtask_interval = int(planner_cfg.get("interval", 0))
        self._steps_since_subtask_update = 0
        self._top_reward_enabled = bool(topreward_cfg.get("enabled", False))
        self._top_reward_max_frames = int(topreward_cfg.get("max_frames", 16))

        self._marl_client: MarlClient | None = None
        self._marl_image_format = str(self._marl_cfg.get("image_format", "jpeg"))
        self._marl_topreward_camera_name = topreward_cfg.get("camera_name", "main")
        self._marl_topreward_fps = float(topreward_cfg.get("fps", 2.0))
        marl_memory_entries = int(planner_cfg.get("max_memory_entries", 20))

        self._marl_run_ids = [
            self._build_marl_run_id(stage_id) for stage_id in range(self.stage_num)
        ]
        self._marl_episode_indices = [0 for _ in range(self.stage_num)]
        self._marl_episode_ids = ["" for _ in range(self.stage_num)]
        self._marl_step_ids = [-1 for _ in range(self.stage_num)]
        self._marl_topreward_start_step_ids = [0 for _ in range(self.stage_num)]
        self._marl_prev_top_scores = [0.0 for _ in range(self.stage_num)]
        self._marl_memories = [
            deque(maxlen=marl_memory_entries) for _ in range(self.stage_num)
        ]

        if not self.only_eval:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )

        self.n_train_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        self.actor_split_num = self.get_actor_split_num()

        self.robot_clients: list[RobotServerClient] = []
        self.eval_robot_clients: list[RobotServerClient] = []
        self.last_obs_list: list[dict] = []
        self.last_intervened_info_list: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    def _validate_supported_cfg(self) -> None:
        if str(self.cfg.env.train.env_type) != "remote":
            raise ValueError(
                "RemoteYamEnvWorker only supports env.train.env_type=remote, got "
                f"{self.cfg.env.train.env_type!r}."
            )
        if self.enable_eval and str(self.cfg.env.eval.env_type) != "remote":
            raise ValueError(
                "RemoteYamEnvWorker only supports env.eval.env_type=remote, got "
                f"{self.cfg.env.eval.env_type!r}."
            )
        if not bool(self.cfg.marl.get("enabled", True)):
            raise ValueError("RemoteYamEnvWorker requires marl.enabled=true.")

    def _build_marl_run_id(self, stage_id: int) -> str:
        experiment_name = str(
            getattr(getattr(self.cfg.runner, "logger", {}), "experiment_name", "rlinf")
        )
        return f"{experiment_name}-rank{self._rank}-stage{stage_id}"

    @staticmethod
    def _to_numpy_image_array(image: Any) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        return np.asarray(image)

    def _obs_to_marl_images(self, obs: dict[str, Any]) -> list[MarlImage]:
        images: list[MarlImage] = []

        main_images = obs.get("main_images")
        if main_images is not None:
            main_array = self._to_numpy_image_array(main_images)
            if main_array.ndim == 4:
                images.append(MarlImage(camera_name="main", image=main_array[0]))
            elif main_array.ndim == 3:
                images.append(MarlImage(camera_name="main", image=main_array))

        wrist_images = obs.get("wrist_images")
        if wrist_images is not None:
            wrist_array = self._to_numpy_image_array(wrist_images)
            if wrist_array.ndim == 5:
                for index, image in enumerate(wrist_array[0]):
                    images.append(MarlImage(camera_name=f"wrist_{index}", image=image))

        extra_view_images = obs.get("extra_view_images")
        if extra_view_images is not None:
            extra_array = self._to_numpy_image_array(extra_view_images)
            if extra_array.ndim == 5:
                for index, image in enumerate(extra_array[0]):
                    images.append(MarlImage(camera_name=f"extra_{index}", image=image))

        return images

    def _get_marl_memory_text(self, stage_id: int) -> str:
        memory = self._marl_memories[stage_id]
        return "\n".join(memory) if memory else "(no history yet)"

    def _sync_task_descriptions_to_obs(
        self, *, stage_id: int, env_output: EnvOutput | None = None
    ) -> None:
        task_descriptions = self.robot_clients[stage_id].task_descriptions
        last_obs = self.robot_clients[stage_id].last_obs
        if isinstance(last_obs, dict):
            last_obs["task_descriptions"] = list(task_descriptions)
        if env_output is not None and isinstance(env_output.obs, dict):
            env_output.obs["task_descriptions"] = list(task_descriptions)
        if env_output is not None and isinstance(env_output.final_obs, dict):
            env_output.final_obs["task_descriptions"] = list(task_descriptions)

    def _start_marl_episode(self, stage_id: int) -> None:
        self._marl_episode_ids[stage_id] = (
            f"ep_{self._marl_episode_indices[stage_id]:06d}"
        )
        self._marl_episode_indices[stage_id] += 1
        self._marl_step_ids[stage_id] = -1
        self._marl_topreward_start_step_ids[stage_id] = 0
        self._marl_prev_top_scores[stage_id] = 0.0
        self._marl_memories[stage_id].clear()

    def _record_marl_step(self, stage_id: int, env_output: EnvOutput) -> bool:
        if self._marl_client is None or env_output.obs is None:
            return False

        images = self._obs_to_marl_images(env_output.obs)
        if not images:
            return False

        client = self.robot_clients[stage_id]
        task_description = client.task_description
        next_step_id = self._marl_step_ids[stage_id] + 1
        try:
            self._marl_client.create_image_set(
                run_id=self._marl_run_ids[stage_id],
                episode_id=self._marl_episode_ids[stage_id],
                step_id=next_step_id,
                task_description=task_description,
                images=images,
                metadata={
                    "worker_rank": self._rank,
                    "stage_id": stage_id,
                },
            )
        except Exception as exc:  # pragma: no cover - service failure
            self.log_warning(
                f"[RemoteYamEnvWorker] marl create_image_set failed for stage {stage_id}: {exc}"
            )
            return False

        self._marl_step_ids[stage_id] = next_step_id
        if task_description:
            self._marl_memories[stage_id].append(
                f"Step {next_step_id}: task={task_description}"
            )
        return True

    def init_worker(self):
        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.cfg.env.train.total_num_envs // self.stage_num
            ),
        }
        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.cfg.env.eval.total_num_envs // self.stage_num
            )
        self.log_info(f"Robot worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Robot worker initialized with src_ranks: {self.src_ranks}")

        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        if not self.only_eval:
            self.robot_clients = [
                RobotServerClient(
                    cfg=self.cfg.env.train,
                    num_envs=self.train_num_envs_per_stage,
                    worker_info=self.worker_info,
                )
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_robot_clients = [
                RobotServerClient(
                    cfg=self.cfg.env.eval,
                    num_envs=self.eval_num_envs_per_stage,
                    worker_info=self.worker_info,
                )
                for _ in range(self.stage_num)
            ]

        base_url = str(self._marl_cfg.get("base_url", "http://127.0.0.1:8080"))
        timeout_s = float(self._marl_cfg.get("timeout_s", 30.0))
        self._marl_client = MarlClient(
            base_url=base_url,
            timeout_s=timeout_s,
            image_format=self._marl_image_format,
        )
        if bool(self._marl_cfg.get("healthcheck", True)):
            self._marl_client.healthz()
        self.log_info(f"[RemoteYamEnvWorker] marl enabled at {base_url}.")

        if not self.only_eval:
            self._init_robot_clients()

    def _maybe_update_subtask(self, stage_id: int) -> None:
        if self._subtask_interval <= 0:
            return

        self._steps_since_subtask_update += 1
        if self._steps_since_subtask_update < self._subtask_interval:
            return

        self._steps_since_subtask_update = 0
        if self._marl_client is None or self._marl_step_ids[stage_id] < 0:
            return

        try:
            response = self._marl_client.plan(
                run_id=self._marl_run_ids[stage_id],
                episode_id=self._marl_episode_ids[stage_id],
                step_id=self._marl_step_ids[stage_id],
                memory_text=self._get_marl_memory_text(stage_id),
            )
            new_subtask = str(response.get("subtask_text", "")).strip()
        except Exception as exc:  # pragma: no cover - service failure
            self.log_warning(
                f"[RemoteYamEnvWorker] marl planner request failed: {exc}. "
                "Keeping current task description."
            )
            return

        if new_subtask:
            self.robot_clients[stage_id].task_description = new_subtask
            if self._top_reward_enabled:
                self._reset_top_reward_state(stage_id)
            self._marl_memories[stage_id].append(
                f"Planner updated task to: {new_subtask}"
            )
            self.log_info(
                f"[RemoteYamEnvWorker] Subtask updated for stage {stage_id}: '{new_subtask}'"
            )

    def _compute_top_reward(self, env_output: EnvOutput, stage_id: int) -> EnvOutput:
        if not self._top_reward_enabled or self._marl_client is None:
            return env_output
        if self._marl_step_ids[stage_id] < 0:
            return env_output

        instruction = self.robot_clients[stage_id].task_description
        try:
            score_response = self._marl_client.score_topreward(
                run_id=self._marl_run_ids[stage_id],
                episode_id=self._marl_episode_ids[stage_id],
                start_step_id=self._marl_topreward_start_step_ids[stage_id],
                end_step_id=self._marl_step_ids[stage_id],
                instruction=instruction,
                max_frames=self._top_reward_max_frames,
                camera_name=self._marl_topreward_camera_name,
                fps=self._marl_topreward_fps,
            )
            score_t = float(score_response["reward"])
        except Exception as exc:  # pragma: no cover - service failure
            self.log_warning(
                f"[RemoteYamEnvWorker] marl TOPReward failed for stage {stage_id}: {exc}"
            )
            return env_output

        reward = score_t - self._marl_prev_top_scores[stage_id]
        self._marl_prev_top_scores[stage_id] = score_t
        if env_output.rewards is not None:
            env_output.rewards[:, -1] = reward
        self.log_info(
            f"[RemoteYamEnvWorker] marl TOPReward: score={score_t:.4f}, delta={reward:.4f}"
        )
        return env_output

    def _reset_top_reward_state(self, stage_id: int | None = None) -> None:
        if stage_id is None:
            for stage_index in range(self.stage_num):
                self._marl_prev_top_scores[stage_index] = 0.0
                self._marl_topreward_start_step_ids[stage_index] = (
                    self._marl_step_ids[stage_index] + 1
                )
        else:
            self._marl_prev_top_scores[stage_id] = 0.0
            self._marl_topreward_start_step_ids[stage_id] = (
                self._marl_step_ids[stage_id] + 1
            )

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        env_world_size = self._component_placement.get_world_size("env")
        rollout_world_size = self._component_placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            dst_rank=self._rank,
        )

    def _init_robot_clients(self):
        if self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                extracted_obs, _ = self.robot_clients[stage_id].reset()
                self.last_obs_list.append(extracted_obs)
                self.last_intervened_info_list.append((None, None))

    @Worker.timer("robot_interact_step")
    def robot_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.robot_clients[stage_id].chunk_step(chunk_actions)
        )
        extracted_obs = obs_list[-1] if obs_list else None
        infos = infos_list[-1] if infos_list else {}
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos.get("final_info", {}):
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )

        self._record_marl_step(stage_id, env_output)
        env_output = self._compute_top_reward(env_output, stage_id)

        if self._top_reward_enabled and chunk_dones[:, -1].any():
            self._reset_top_reward_state(stage_id)
            if self.cfg.env.train.auto_reset:
                self._start_marl_episode(stage_id)

        return env_output, env_info

    def robot_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = (
            self.eval_robot_clients[stage_id].chunk_step(chunk_actions)
        )
        extracted_obs = obs_list[-1] if obs_list else None
        infos = infos_list[-1] if infos_list else {}
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key].cpu()
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"]
            if "final_observation" in infos
            else None,
        )
        return env_output, env_info

    def recv_chunk_actions(self, input_channel: Channel, mode="train") -> np.ndarray:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        chunk_action = []
        for src_rank, expected_size in src_ranks_and_sizes:
            action_i = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_actions"
                ),
            )
            if isinstance(action_i, torch.Tensor):
                action_i = action_i.detach().cpu().numpy()
            else:
                action_i = np.asarray(action_i)
            assert action_i.shape[0] == expected_size, (
                f"Expected action shard size {expected_size} from rollout rank {src_rank}, "
                f"got shape {action_i.shape}."
            )
            chunk_action.append(action_i)
        chunk_action = np.concatenate(chunk_action, axis=0)
        expected_total_size = sum(size for _, size in src_ranks_and_sizes)
        assert chunk_action.shape[0] == expected_total_size, (
            f"Expected concatenated action size {expected_total_size}, got {chunk_action.shape[0]}."
        )
        return chunk_action

    def recv_rollout_results(
        self, input_channel: Channel, mode="train"
    ) -> RolloutResult:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        rollout_results: list[RolloutResult] = []

        def _infer_rollout_batch_size(rollout_result: RolloutResult) -> int:
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(rollout_result, field_name, None)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            if rollout_result.forward_inputs:
                first_tensor = next(iter(rollout_result.forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return first_tensor.shape[0]
            raise ValueError("Cannot infer batch size from rollout result.")

        for src_rank, expected_size in src_ranks_and_sizes:
            rollout_result = input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_rollout_results"
                ),
            )

            actual_size = _infer_rollout_batch_size(rollout_result)
            assert actual_size == expected_size, (
                f"Expected rollout result size {expected_size} from rollout rank {src_rank}, "
                f"got batch size {actual_size}."
            )
            rollout_results.append(rollout_result)

        return RolloutResult.merge_rollout_results(rollout_results)

    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def finish_rollout(self, mode="train"):
        if mode == "train":
            for client in self.robot_clients:
                if hasattr(client, "update_reset_state_ids"):
                    client.update_reset_state_ids()
        elif mode == "eval":
            for client in self.eval_robot_clients:
                if not self.cfg.env.eval.auto_reset and hasattr(
                    client, "update_reset_state_ids"
                ):
                    client.update_reset_state_ids()

    def split_env_batch(
        self,
        env_batch: dict[str, Any],
        sizes: list[int],
        mode: Literal["train", "eval"],
    ) -> list[dict[str, Any]]:
        count = len(sizes)
        total_size = sum(sizes)
        splitted_env_batches = [{} for _ in range(count)]
        for key, value in env_batch.items():
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == total_size, (
                    f"Tensor field '{key}' expected batch size {total_size}, got {value.shape[0]}."
                )
                splitted_values = torch.split(value, sizes, dim=0)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_values[i].contiguous()
            elif isinstance(value, list):
                length = len(value)
                if mode == "train":
                    assert length == self.train_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.train_num_envs_per_stage}, got {length}"
                    )
                elif mode == "eval":
                    assert length == self.eval_num_envs_per_stage, (
                        f"Mode {mode}: key '{key}' expected length {self.eval_num_envs_per_stage}, got {length}"
                    )
                assert length == total_size, (
                    f"List field '{key}' expected length {total_size}, got {length}."
                )
                begin = 0
                for i, size in enumerate(sizes):
                    splitted_env_batches[i][key] = value[begin : begin + size]
                    begin += size
            elif isinstance(value, dict):
                splitted_sub_batches = self.split_env_batch(value, sizes, mode)
                for i in range(count):
                    splitted_env_batches[i][key] = splitted_sub_batches[i]
            else:
                for i in range(count):
                    splitted_env_batches[i][key] = value

        return splitted_env_batches

    def send_env_batch(
        self,
        output_channel: Channel,
        env_batch: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> None:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        env_batches = self.split_env_batch(env_batch, split_sizes, mode)
        for (rank, _), env_batch_i in zip(dst_ranks_and_sizes, env_batches):
            output_channel.put(
                item=env_batch_i,
                key=CommMapper.build_channel_key(self._rank, rank, extra=f"{mode}_obs"),
            )

    def bootstrap_step(self) -> list[EnvOutput]:
        self._steps_since_subtask_update = 0
        if self._top_reward_enabled:
            self._reset_top_reward_state()
        for stage_id in range(self.stage_num):
            self._start_marl_episode(stage_id)

        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.cfg.actor.model.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.robot_clients[stage_id].is_start = True
                extracted_obs, infos = self.robot_clients[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()
                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=infos["final_observation"]
                    if "final_observation" in infos
                    else None,
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()
            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def record_env_metrics(
        self, env_metrics: dict[str, list], env_info: dict[str, Any], epoch: int
    ):
        for key, value in env_info.items():
            if (
                not self.cfg.env.train.auto_reset
                and not self.cfg.env.train.ignore_terminations
            ):
                if key in env_metrics and len(env_metrics[key]) > epoch:
                    env_metrics[key][epoch] = value
                else:
                    env_metrics[key].append(value)
            else:
                env_metrics[key].append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    def _log_task_text_for_rollout(
        self, *, stage_id: int, obs: dict[str, Any] | None, phase: str
    ) -> None:
        if obs is None:
            return
        task_descriptions = obs.get("task_descriptions", None)
        if not task_descriptions:
            self.log_info(
                f"[RemoteYamEnvWorker] Rollout obs stage={stage_id} phase={phase} has no task_descriptions."
            )
            return
        self.log_info(
            f"[RemoteYamEnvWorker] Rollout obs stage={stage_id} phase={phase} "
            f"task={task_descriptions[0]!r}"
        )

    async def send_rollout_trajectories(
        self, rollout_result: EmbodiedRolloutResult, channel: Channel
    ):
        trajectories: Trajectory = rollout_result.to_splited_trajectories(
            self.actor_split_num
        )
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)

    async def _run_interact_once(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
            )
            for _ in range(self.stage_num)
        ]
        env_metrics = defaultdict(list)

        for epoch in range(self.rollout_epoch):
            env_outputs = self.bootstrap_step()
            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                env_batch = env_output.to_dict()
                self._log_task_text_for_rollout(
                    stage_id=stage_id,
                    obs=env_batch["obs"],
                    phase="bootstrap",
                )
                self.send_env_batch(
                    output_channel,
                    {
                        "obs": env_batch["obs"],
                        "final_obs": env_batch["final_obs"],
                    },
                )

            for _ in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    rollout_result = self.recv_rollout_results(
                        input_channel, mode="train"
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=rollout_result.prev_logprobs
                        if self.collect_prev_infos
                        else None,
                        prev_values=rollout_result.prev_values
                        if self.collect_prev_infos
                        else None,
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)

                    env_output, env_info = self.robot_interact_step(
                        rollout_result.actions, stage_id
                    )
                    self._maybe_update_subtask(stage_id)
                    self._sync_task_descriptions_to_obs(
                        stage_id=stage_id, env_output=env_output
                    )
                    env_batch = env_output.to_dict()
                    self._log_task_text_for_rollout(
                        stage_id=stage_id,
                        obs=env_batch["obs"],
                        phase="post_step",
                    )
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                    )
                    if self.collect_transitions:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output
                    self.record_env_metrics(env_metrics, env_info, epoch)

            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                if env_output.intervene_actions is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

                rollout_result = self.recv_rollout_results(input_channel, mode="train")
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values
                )
                chunk_step_result = ChunkStepResult(
                    prev_values=rollout_result.prev_values
                    if self.collect_prev_infos
                    else None,
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)

            self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if actor_channel is not None:
            for stage_id in range(self.stage_num):
                await self.send_rollout_trajectories(
                    self.rollout_results[stage_id], actor_channel
                )

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel | None = None,
    ):
        return await self._run_interact_once(
            input_channel,
            output_channel,
            actor_channel,
            cooperative_yield=False,
        )

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        eval_metrics = defaultdict(list)

        for eval_rollout_epoch in range(self.cfg.algorithm.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_robot_clients[stage_id].is_start = True
                    extracted_obs, infos = self.eval_robot_clients[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=infos["final_observation"]
                        if "final_observation" in infos
                        else None,
                    )
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(
                        input_channel, mode="eval"
                    )
                    env_output, env_info = self.robot_evaluate_step(
                        raw_chunk_actions, stage_id
                    )
                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch
                            == self.cfg.algorithm.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_env_batch(
                        output_channel,
                        {
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                    )

            self.finish_rollout(mode="eval")

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        return compute_split_num(recv_num, send_num)

    def close_clients(self):
        for client in self.robot_clients:
            try:
                client.close()
            except Exception:
                pass
        for client in self.eval_robot_clients:
            try:
                client.close()
            except Exception:
                pass
