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

from collections import UserDict
from contextlib import contextmanager
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import Self

from rlinf.scheduler import Worker
from rlinf.utils.timers import NamedTimer


def compute_rollout_metrics_dynamic(
    rollout_batch: dict[str, torch.Tensor],
    max_prompt_len: int,
    response_len: int,
    data_parallel_group: Optional[ProcessGroup] = None,
    use_critic: bool = False,
):
    """Compute rollout metrics for dynamic multi-turn scenarios.

    Key features:
    - Reward scores computed at TRAJECTORY level (not turn level)
    - Uses idx_to_traj to map turns to trajectories
    - Aggregates turn rewards to trajectory level before averaging
    - Supports tool call metrics, eval metrics, and MAS turn metrics

    Args:
        rollout_batch: Batch containing turn-level data with idx_to_traj mapping
        max_prompt_len: Maximum prompt length
        response_len: Response length
        data_parallel_group: Data parallel group for distributed training
        use_critic: Whether using critic (unused)

    Returns:
        Tuple of (rollout_metrics dict, total_prompt_lengths tensor, total_decode_lengths tensor)
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Extract basic tensors
    advantages = rollout_batch["advantages"].to(device=device)
    mask = rollout_batch["response_mask"].to(device=device)
    prompt_lengths = rollout_batch["prompt_lengths"].clone().to(device=device)
    response_lengths = rollout_batch["response_lengths"].clone().to(device=device)
    reward_scores = rollout_batch["rewards"].clone().to(device=device)
    is_end = rollout_batch["is_end"].clone().float().to(device=device)
    idx_to_traj = rollout_batch["idx_to_traj"]
    num_trajectories = max(idx_to_traj) + 1
    num_seq = prompt_lengths.numel()

    dp_world_size = torch.distributed.get_world_size(data_parallel_group)

    # Gather prompt/response lengths across DPs
    prompt_lengths_list: list[list[int]] = [None for _ in range(dp_world_size)]
    decode_lengths_list: list[list[int]] = [None for _ in range(dp_world_size)]
    torch.distributed.all_gather_object(
        prompt_lengths_list, prompt_lengths.tolist(), group=data_parallel_group
    )
    torch.distributed.all_gather_object(
        decode_lengths_list, response_lengths.tolist(), group=data_parallel_group
    )

    total_prompt_lengths = torch.tensor(sum(prompt_lengths_list, []), device=device)
    total_decode_lengths = torch.tensor(sum(decode_lengths_list, []), device=device)

    # Compute trajectory-level rewards
    trajectory_rewards = torch.zeros(
        num_trajectories, dtype=reward_scores.dtype, device=device
    )
    trajectory_counts = torch.zeros(num_trajectories, dtype=torch.long, device=device)
    for turn_idx, traj_idx in enumerate(idx_to_traj):
        trajectory_rewards[traj_idx] += reward_scores[turn_idx]
        trajectory_counts[traj_idx] += 1
    trajectory_rewards = trajectory_rewards / trajectory_counts.clamp(min=1).float()

    # Compute local sums
    sum_plen = prompt_lengths.sum().detach().item()
    sum_rlen = response_lengths.sum().detach().item()
    sum_turn_rewards = reward_scores.sum().detach().item()
    sum_traj_rewards = trajectory_rewards.sum().detach().item()
    sum_end = is_end.sum().detach().item()

    valid_adv = torch.masked_select(advantages, mask)
    n_valid_token = mask.sum().detach().item()
    sum_adv = valid_adv.to(torch.float64).sum().detach().item()

    # All-reduce basic metrics
    reduce_metrics = torch.as_tensor(
        [
            sum_plen,
            sum_rlen,
            sum_traj_rewards,
            sum_turn_rewards,
            sum_end,
            sum_adv,
            num_seq,
            n_valid_token,
            num_trajectories,
        ],
        device=device,
        dtype=torch.float32,
    )
    torch.distributed.all_reduce(
        reduce_metrics, torch.distributed.ReduceOp.SUM, group=data_parallel_group
    )
    (
        sum_plen,
        sum_rlen,
        sum_traj_rewards,
        sum_turn_rewards,
        sum_end,
        sum_adv,
        num_seq,
        n_valid_token,
        num_trajectories,
    ) = reduce_metrics.tolist()

    # All-reduce advantage min/max
    adv_max = torch.max(valid_adv).detach().item()
    adv_min = torch.min(valid_adv).detach().item()
    reduce_tensor = torch.as_tensor(
        [-adv_min, adv_max], device=device, dtype=torch.float32
    )
    torch.distributed.all_reduce(
        reduce_tensor, torch.distributed.ReduceOp.MAX, group=data_parallel_group
    )
    adv_min, adv_max = reduce_tensor.tolist()

    # All-reduce max lengths
    local_max_prompt = prompt_lengths.max().item()
    local_max_response = response_lengths.max().item()
    local_max_total = (prompt_lengths + response_lengths).max().item()
    max_length_metrics = torch.as_tensor(
        [local_max_prompt, local_max_response, local_max_total],
        device=device,
        dtype=torch.float32,
    )
    torch.distributed.all_reduce(
        max_length_metrics, torch.distributed.ReduceOp.MAX, group=data_parallel_group
    )
    global_max_prompt, global_max_response, global_max_total = (
        max_length_metrics.tolist()
    )

    # Build final metrics dict
    rollout_metrics = {
        "total_num_sequence": num_seq,
        "prompt_length": sum_plen / num_seq,
        "response_length": sum_rlen / num_seq,
        "total_length": (sum_plen + sum_rlen) / num_seq,
        "max_prompt_length": int(global_max_prompt),
        "max_response_length": int(global_max_response),
        "max_total_length": int(global_max_total),
        "reward_scores_traj": sum_traj_rewards / num_trajectories,
        "reward_scores_turn": sum_turn_rewards / num_seq,
        "avg_turns_per_traj": num_seq / num_trajectories,
        "fraction_of_samples_properly_ended": sum_end / num_seq,
        "advantages_mean": sum_adv / n_valid_token,
        "advantages_max": adv_max,
        "advantages_min": -adv_min,
    }
    return rollout_metrics, total_prompt_lengths, total_decode_lengths


def compute_rollout_metrics(
    rollout_batch: dict[str, torch.Tensor],
    max_prompt_len: int,
    response_len: int,
    data_parallel_group: Optional[ProcessGroup] = None,
    use_critic: bool = False,
):
    device = Worker.torch_platform.current_device()
    advantages = rollout_batch["advantages"].to(device=device)
    mask = rollout_batch["response_mask"][:, -response_len:].to(device=device)
    prompt_lengths = rollout_batch["prompt_lengths"].clone().to(device=device)
    response_lengths = rollout_batch["response_lengths"].clone().to(device=device)
    reward_scores = rollout_batch["rewards"].clone().to(device=device)
    is_end = rollout_batch["is_end"].clone().float().to(device=device)

    dp_world_size = torch.distributed.get_world_size(data_parallel_group)

    prompt_lengths_list: list[list[int]] = [None for _ in range(dp_world_size)]
    decode_lengths_list: list[list[int]] = [None for _ in range(dp_world_size)]
    torch.distributed.all_gather_object(
        prompt_lengths_list,
        prompt_lengths.tolist(),
        group=data_parallel_group,
    )
    torch.distributed.all_gather_object(
        decode_lengths_list,
        response_lengths.tolist(),
        group=data_parallel_group,
    )

    total_prompt_lengths = torch.tensor(sum(prompt_lengths_list, []), device="cpu")
    total_decode_lengths = torch.tensor(sum(decode_lengths_list, []), device="cpu")

    sum_plen = prompt_lengths.sum().detach().item()
    sum_rlen = response_lengths.sum().detach().item()
    sum_rewards = reward_scores.sum().detach().item()
    sum_end = is_end.sum().detach().item()

    mean_rlen = total_decode_lengths.float().mean().detach().item()
    var_rlen = total_decode_lengths.float().var().detach().item()
    min_rlen, max_rlen = total_decode_lengths.float().aminmax()
    min_rlen, max_rlen = min_rlen.detach().item(), max_rlen.detach().item()

    valid_adv = torch.masked_select(advantages, mask)
    n_valid_token = mask.sum().detach().item()
    sum_adv = valid_adv.to(torch.float64).sum().detach().item()

    num_seq = prompt_lengths.numel()
    reduce_metrics = torch.as_tensor(
        [sum_plen, sum_rlen, sum_rewards, sum_end, sum_adv, num_seq, n_valid_token],
        device=device,
        dtype=torch.float32,
    )

    torch.distributed.all_reduce(
        reduce_metrics,
        torch.distributed.ReduceOp.SUM,
        group=data_parallel_group,
    )

    sum_plen, sum_rlen, sum_rewards, sum_end, sum_adv, num_seq, n_valid_token = (
        reduce_metrics.tolist()
    )

    adv_max = torch.max(valid_adv).detach().item()
    adv_min = torch.min(valid_adv).detach().item()
    reduce_tensor = torch.as_tensor(
        [-adv_min, adv_max],
        device=Worker.torch_platform.current_device(),
        dtype=torch.float32,
    )
    torch.distributed.all_reduce(
        reduce_tensor,
        torch.distributed.ReduceOp.MAX,
        group=data_parallel_group,
    )
    adv_min, adv_max = reduce_tensor.tolist()

    values_metrics = None
    if "values" in rollout_batch:
        values = rollout_batch["values"].float().to(device=device)
        values = torch.masked_select(values, mask)
        mean_value = torch.mean(values)
        torch.distributed.all_reduce(mean_value, op=torch.distributed.ReduceOp.AVG)
        max_value = torch.max(values).detach().item()
        min_value = torch.min(values).detach().item()
        reduce_value_tensor = torch.as_tensor(
            [-min_value, max_value],
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            reduce_value_tensor, op=torch.distributed.ReduceOp.MAX
        )
        min_value, max_value = reduce_value_tensor.tolist()

        values_metrics = {
            "values_mean": mean_value.item(),
            "values_max": max_value,
            "values_min": -min_value,
        }

    rollout_metrics = {
        "total_num_sequence": num_seq,
        "prompt_length": sum_plen / num_seq,
        "response_length": sum_rlen / num_seq,
        "average_response_length": mean_rlen,
        "variance_of_response_length": var_rlen,
        "max_of_response_length": max_rlen,
        "min_of_response_length": min_rlen,
        "total_length": (sum_plen + sum_rlen) / num_seq,
        "reward_scores": sum_rewards / num_seq,
        "fraction_of_samples_properly_ended": sum_end / num_seq,
        "advantages_mean": sum_adv / n_valid_token,
        "advantages_max": adv_max,
        "advantages_min": -adv_min,
    }
    if values_metrics is not None:
        rollout_metrics.update(values_metrics)

    return rollout_metrics, total_prompt_lengths, total_decode_lengths


class RolloutDataBalance(UserDict):
    def __init__(
        self,
        dictionary_data: Optional[dict[str, torch.Tensor]] = None,
        ordered_keys_hint: Optional[list[str]] = None,
    ):
        super().__init__(dictionary_data if dictionary_data is not None else {})

        if ordered_keys_hint and self.data:
            self._ordered_keys = [k for k in ordered_keys_hint if k in self.data]
            if len(self._ordered_keys) != len(self.data) or not all(
                k in self.data for k in self._ordered_keys
            ):
                self._ordered_keys = sorted(self.data.keys())
        elif self.data:
            self._ordered_keys = sorted(self.data.keys())
        else:
            self._ordered_keys = []

    def __getitem__(self, key: Any) -> torch.Tensor:
        if isinstance(key, int):
            if not self._ordered_keys:
                raise IndexError(
                    f"RolloutDataBalance is empty or has no ordered keys for integer indexing. Data keys: {list(self.data.keys())}"
                )
            if 0 <= key < len(self._ordered_keys):
                actual_key = self._ordered_keys[key]
                if actual_key not in self.data:
                    raise KeyError(
                        f"Internal error: Key '{actual_key}' (from index {key}) not in data. Ordered: {self._ordered_keys}. Data: {list(self.data.keys())}"
                    )
                return self.data[actual_key]
            else:
                raise IndexError(
                    f"Integer index {key} out of range for {len(self._ordered_keys)} ordered keys. Ordered: {self._ordered_keys}"
                )
        return super().__getitem__(key)

    @classmethod
    def from_rollout_batches(
        cls: Self,
        rollout_batches: dict[str, torch.Tensor],
        dp_world_size: int,
        dp_rank: int,
        dp_group: Optional[ProcessGroup],
        partitioning_tool: Callable,
    ) -> Self:
        current_device = Worker.torch_platform.current_device()

        # 1. Get local sample count
        current_num_samples = 0
        if rollout_batches:
            first_tensor = next(iter(rollout_batches.values()))
            if isinstance(first_tensor, torch.Tensor) and first_tensor.numel() > 0:
                current_num_samples = first_tensor.size(0)

        # 2. Calculate local token counts
        local_token_counts = torch.zeros(
            current_num_samples, dtype=torch.int, device=current_device
        )
        if current_num_samples > 0 and "attention_mask" in rollout_batches:
            attn_mask = rollout_batches["attention_mask"]
            if (
                isinstance(attn_mask, torch.Tensor)
                and attn_mask.size(0) == current_num_samples
            ):
                local_token_counts = attn_mask.sum(dim=1).int()

        # 3. Gather global information: sample counts from each rank
        if dp_world_size > 1 and dp_group is not None:
            # Multi-rank case: use all_gather_object
            all_num_samples = [None] * dp_world_size
            torch.distributed.all_gather_object(
                all_num_samples, current_num_samples, group=dp_group
            )
        else:
            # Single-rank case:
            all_num_samples = [current_num_samples]

        all_num_samples = [
            int(num_samples) if num_samples is not None else 0
            for num_samples in all_num_samples
        ]

        global_total_samples = sum(all_num_samples)
        max_samples_rank = max(all_num_samples) if global_total_samples > 0 else 0

        # 4. Gather global token counts for all samples
        global_token_counts_list: list[int] = []
        all_ranks_local_token_counts_list: list[list[int]] = [
            [] for _ in range(dp_world_size)
        ]

        if global_total_samples > 0:
            padded_local_tokens = torch.zeros(
                max_samples_rank, dtype=torch.int, device=current_device
            )
            if local_token_counts.numel() > 0:
                padded_local_tokens[: local_token_counts.size(0)] = local_token_counts

            all_padded_tokens_t = [
                torch.empty_like(padded_local_tokens) for _ in range(dp_world_size)
            ]
            if dp_group and dp_world_size > 1:
                all_padded_tokens_t = [
                    torch.zeros(
                        max_samples_rank, dtype=torch.int, device=current_device
                    )
                    for _ in range(dp_world_size)
                ]
                torch.distributed.all_gather(
                    all_padded_tokens_t, padded_local_tokens, group=dp_group
                )
            else:
                all_padded_tokens_t = [padded_local_tokens]

            for i_rank in range(dp_world_size):
                num_s_rank = all_num_samples[i_rank]
                if num_s_rank > 0:
                    rank_tokens = all_padded_tokens_t[i_rank][:num_s_rank].tolist()
                    global_token_counts_list.extend(rank_tokens)
                    all_ranks_local_token_counts_list[i_rank] = rank_tokens

        # 5. Calculate global sample indices assigned to current rank
        my_assigned_global_indices: list[int] = []
        all_ranks_assigned_tokens_after_balance: list[int] = [
            0
        ] * dp_world_size  # For rank 0 to print summary

        if global_total_samples > 0:
            if not global_token_counts_list:
                global_token_counts_list = [1] * global_total_samples

            k_partitions = min(global_total_samples, dp_world_size)
            if k_partitions > 0 and len(global_token_counts_list) >= k_partitions:
                partitions_indices_all_ranks = partitioning_tool(
                    seqlen_list=global_token_counts_list,
                    k_partitions=k_partitions,
                    equal_size=True,
                )

                if dp_rank < k_partitions and dp_rank < len(
                    partitions_indices_all_ranks
                ):
                    my_assigned_global_indices = partitions_indices_all_ranks[dp_rank]

                if dp_group and dp_world_size > 1:
                    if dp_rank == 0:
                        for r_idx in range(k_partitions):
                            if r_idx < len(partitions_indices_all_ranks):
                                rank_indices = partitions_indices_all_ranks[r_idx]
                                all_ranks_assigned_tokens_after_balance[r_idx] = sum(
                                    global_token_counts_list[g_idx]
                                    for g_idx in rank_indices
                                )

        # 6. Get superset of all keys that appear on all DP ranks and sort them
        local_keys = set(rollout_batches.keys())
        all_keys_sets: list[Optional[set[str]]] = [None] * dp_world_size
        if dp_group and dp_world_size > 1:
            torch.distributed.all_gather_object(
                all_keys_sets, local_keys, group=dp_group
            )
        else:
            all_keys_sets = [local_keys]

        superset_keys = set().union(*(s for s in all_keys_sets if s is not None))
        final_ordered_keys = sorted(superset_keys)

        # 7. Gather all data from all ranks (CPU)
        payload_cpu = {
            k: v.cpu()
            for k, v in rollout_batches.items()
            if k in final_ordered_keys and isinstance(v, torch.Tensor)
        }
        all_payloads_cpu: list[Optional[dict[str, torch.Tensor]]] = [
            None
        ] * dp_world_size
        if dp_group and dp_world_size > 1:
            torch.distributed.all_gather_object(
                all_payloads_cpu, payload_cpu, group=dp_group
            )
            remove_len = len(all_payloads_cpu) % dp_world_size
            if remove_len > 0:
                all_payloads_cpu = all_payloads_cpu[:-remove_len]
        else:
            all_payloads_cpu = [payload_cpu]

        # 8. Rebuild global batch on CPU and record template specifications
        global_batch_cpu: dict[str, torch.Tensor] = {}
        template_specs: dict[str, dict[str, Any]] = {}
        if global_total_samples > 0:
            for key in final_ordered_keys:
                tensors_for_key = []
                for i_rank, rank_payload in enumerate(all_payloads_cpu):
                    if isinstance(rank_payload, dict) and all_num_samples[i_rank] > 0:
                        tensor = rank_payload.get(key)
                        if (
                            isinstance(tensor, torch.Tensor)
                            and tensor.numel() > 0
                            and tensor.size(0) == all_num_samples[i_rank]
                        ):
                            tensors_for_key.append(tensor)
                            if (
                                key not in template_specs
                            ):  # Store spec from first valid tensor
                                template_specs[key] = {
                                    "dtype": tensor.dtype,
                                    "shape_suffix": list(tensor.shape[1:]),
                                }

                if tensors_for_key:
                    try:
                        cat_tensor = torch.cat(tensors_for_key, dim=0)
                        global_batch_cpu[key] = cat_tensor
                        if (
                            key not in template_specs and cat_tensor.numel() > 0
                        ):  # Update spec if first was empty
                            template_specs[key] = {
                                "dtype": cat_tensor.dtype,
                                "shape_suffix": list(cat_tensor.shape[1:]),
                            }
                    except Exception:
                        pass

        # 9. Select data for current rank
        final_rank_data: dict[str, torch.Tensor] = {}

        def _create_empty_tensor_for_key(
            k: str, specs: dict[str, dict[str, Any]], dev: torch.device
        ) -> torch.Tensor:
            spec = specs.get(k)
            if spec:
                return torch.empty(
                    [0] + spec["shape_suffix"], dtype=spec["dtype"], device=dev
                )
            return torch.empty(0, dtype=torch.float32, device=dev)

        if my_assigned_global_indices:
            indices_cpu = torch.tensor(my_assigned_global_indices, dtype=torch.long)
            for key in final_ordered_keys:
                full_tensor = global_batch_cpu.get(key)
                if (
                    isinstance(full_tensor, torch.Tensor)
                    and full_tensor.numel() > 0
                    and full_tensor.size(0) == global_total_samples
                ):
                    try:
                        final_rank_data[key] = full_tensor.index_select(
                            0, indices_cpu
                        ).to(current_device)
                    except IndexError:
                        final_rank_data[key] = _create_empty_tensor_for_key(
                            key, template_specs, current_device
                        )
        else:
            for key in final_ordered_keys:
                final_rank_data[key] = _create_empty_tensor_for_key(
                    key, template_specs, current_device
                )

        return cls(final_rank_data, ordered_keys_hint=final_ordered_keys)

    @classmethod
    def from_rollout_batches_dynamic(
        cls: Self,
        rollout_batches: dict[str, torch.Tensor],
        dp_world_size: int,
        dp_rank: int,
        dp_group: Optional[ProcessGroup],
        rollout_batch_pad: dict[str, torch.Tensor],
        split_fix_chunk: int,
        partitioning_tool: Callable,
    ) -> Self:
        # 0. Check data
        assert rollout_batches.keys() == rollout_batch_pad.keys(), (
            f"rollout_batches and rollout_batch_pad must have the same keys, but these are [{sorted(rollout_batches.keys())}] and [{sorted(rollout_batch_pad.keys())}]"
        )
        assert (
            "input_ids" in rollout_batches
            and "prompt_lengths" in rollout_batches
            and "response_lengths" in rollout_batches
        )
        batch_size = rollout_batches["input_ids"].size(0)
        assert all(v.size(0) == batch_size for v in rollout_batches.values())
        assert all(v.size(0) == 1 for v in rollout_batch_pad.values())
        for k in rollout_batches.keys():
            assert rollout_batches[k].dtype == rollout_batch_pad[k].dtype, (
                f"batch dtype mismatch: key: {k}, dtype: {rollout_batches[k].dtype}, {rollout_batch_pad[k].dtype}"
            )
            assert rollout_batches[k].shape[1:] == rollout_batch_pad[k].shape[1:], (
                f"batch shape mismatch: key: {k}, shape: {rollout_batches[k].shape}, {rollout_batch_pad[k].shape}"
            )
        rollout_batches = {k: v.cpu() for k, v in rollout_batches.items()}
        rollout_batch_pad = {k: v.cpu() for k, v in rollout_batch_pad.items()}

        # 1. Allgather data
        gathered_rollout_batches = [{} for _ in range(dp_world_size)]
        for k in sorted(rollout_batches.keys()):
            rollout_batch_values = [None for _ in range(dp_world_size)]
            torch.distributed.all_gather_object(
                rollout_batch_values, rollout_batches[k], group=dp_group
            )
            for gathered_batch, value in zip(
                gathered_rollout_batches, rollout_batch_values
            ):
                gathered_batch[k] = value

        global_batch_size = batch_size = sum(
            b["input_ids"].size(0) for b in gathered_rollout_batches
        )

        # 2. Merge data and pad data to fixed length
        if global_batch_size % (dp_world_size * split_fix_chunk) == 0:
            pad_size = 0
        else:
            pad_size = (dp_world_size * split_fix_chunk) - (
                global_batch_size % (dp_world_size * split_fix_chunk)
            )
        # make sure dp_world_size * split_fix_chunk (dp size * self.num_train_steps * self.cfg.actor.micro_batch_size) can be divided by global_batch_size
        merged_rollout_batches = {}
        for key in rollout_batches.keys():
            merged_rollout_batches[key] = torch.cat(
                [rollout_batches[key] for rollout_batches in gathered_rollout_batches]
                + [rollout_batch_pad[key]] * pad_size,
                dim=0,
            )

        # 3. get length of each sample
        sample_lengths = (
            merged_rollout_batches["prompt_lengths"]
            + merged_rollout_batches["response_lengths"]
        ).tolist()

        # 4. Calc partitions
        partitions = partitioning_tool(
            seqlen_list=sample_lengths,
            k_partitions=dp_world_size,
            equal_size=True,
        )
        self_partition = partitions[dp_rank]

        # 5. Get indices of samples for current rank
        selected_rollout_batches = {
            k: v[self_partition] for k, v in merged_rollout_batches.items()
        }

        return selected_rollout_batches

    def chunk(self, rank, split_size):
        chunked_rollout_batch = type(self)()

        batch_set = {tensor.size(0) for tensor in self.data.values()}
        assert len(batch_set) == 1, (
            "batch sizes are not the same across the rollout batch"
        )
        B = batch_set.pop()

        indices = torch.arange(B).tensor_split(split_size)[rank]

        for k in self.data:
            chunked_rollout_batch[k] = self.data[k][indices].clone()

        return chunked_rollout_batch


def all_reduce_int(
    obj: int,
    op: ReduceOp = ReduceOp.MIN,
    group: ProcessGroup = None,
):
    obj_tensor = torch.tensor(
        [obj], dtype=torch.long, device=Worker.torch_platform.current_device()
    )
    torch.distributed.all_reduce(
        obj_tensor,
        op,
        group=group,
    )
    return obj_tensor.item()


def normalize_tensor(tensor, mask, group=None):
    """normalizes a tensor using global mean and std"""
    dtype = torch.float64
    tensor = tensor.to(dtype)
    tensor = tensor.to(Worker.torch_device_type)
    mask = mask.to(Worker.torch_device_type)

    tensor_global_mean, tensor_global_var = masked_global_mean_var(
        tensor, mask, group=group
    )
    tensor = (tensor - tensor_global_mean) * torch.rsqrt(tensor_global_var + 1e-5)
    return tensor.float()


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    dim: Optional[int | tuple[int, ...]] = None,
    inplace: Optional[bool] = False,
    unbiased: Optional[bool] = False,
    eps: Optional[float] = 1e-5,
    high_precision: Optional[bool] = True,
    all_reduce: Optional[bool] = True,
    group: Optional[ProcessGroup] = None,
):
    """Normalize x with a mask. Typically used in advantage normalization.

    Args:
        x (torch.Tensor):
            Tensor to be normalized.
        mask (torch.Tensor, optional):
            A mask with the same shape as x. Defaults to None.
        dim (int or tuple of ints, optional):
            Dimensions to be normalized. Defaults to None.
        inplace (bool, optional):
            Whether to perform in-place operation. Defaults to False.
        eps (torch.Tensor, optional):
            Minimal denominator. Defaults to 1e-5.

    Returns:
        torch.Tensor:
            Normalized x, with the same shape as x.
    """
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype=dtype).cuda()
    if not inplace:
        x = x.clone()
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype=dtype).cuda()
        assert len(mask.shape) == len(x.shape), (mask.shape, x.shape, dim)
        for i in range(len(x.shape)):
            if i in dim:
                assert mask.shape[i] == x.shape[i], (mask.shape, x.shape, dim)
            else:
                assert mask.shape[i] == 1, (mask.shape, x.shape, dim)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)

    if torch.distributed.is_initialized() and all_reduce:
        torch.distributed.all_reduce(
            factor,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
        torch.distributed.all_reduce(
            x_sum,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
        torch.distributed.all_reduce(
            x_sum_sq,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def masked_global_mean_var(values, mask, group=None):
    """computes the global mean and var when there is a mask

    NOTE: the variance here is uncorrected

    mask and values must have same shape, with mask being {0,1} with 1 being the values we want to keep
    """
    assert values.shape == mask.shape, (values.shape, mask.shape)
    values = values.to(Worker.torch_device_type)
    mask = mask.to(Worker.torch_device_type)

    values = values * mask

    # Get global sum and count and calculate the global mean and variance
    sum_and_count = torch.tensor(
        [values.sum(), mask.sum()],
        dtype=torch.float64,
        device=Worker.torch_platform.current_device(),
    )
    torch.distributed.all_reduce(sum_and_count, group=group)
    global_sum, global_count = sum_and_count
    global_mean = global_sum / global_count
    variance_summed = (
        (((values - global_mean) ** 2) * mask)
        .sum()
        .to(device=Worker.torch_platform.current_device(), dtype=torch.float64)
    )

    torch.distributed.all_reduce(variance_summed, group=group)

    return global_mean, variance_summed / global_count


def report_device_info(info_str):
    free_gpu_memory, total_gpu_memory = Worker.torch_platform.mem_get_info()
    free_gpu_memory /= 2**30
    total_gpu_memory /= 2**30

    memory_allocated = Worker.torch_platform.memory_allocated() / 2**30
    memory_reserved = Worker.torch_platform.memory_reserved() / 2**30

    print(
        f"[Rank {torch.distributed.get_rank()}] {info_str}, {free_gpu_memory=:.2f} GiB, {total_gpu_memory=:.2f} GiB, {memory_allocated=:.2f} GiB, {memory_reserved=:.2f} GiB"
    )


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return the
    division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def all_reduce_dict(
    dictionary, dtype=torch.float32, group=None, op=torch.distributed.ReduceOp.SUM
):
    keys = sorted(dictionary)
    tensor = torch.as_tensor(
        [dictionary[k] for k in keys],
        dtype=dtype,
        device=Worker.torch_platform.current_device(),
    )
    torch.distributed.all_reduce(tensor, op=op, group=group)
    return dict(zip(keys, tensor.tolist()))


class ScopedTimer:
    """
    A thin adapter over the NamedTimer class to help time sections of code
    using a context manager.

    This class is useful for tracking timings automatically so you don't need
    to manually collect them. You only need to pass the timer around and can
    collect the durations in one place, instead of returning and mutating
    dictionaries throughout your code.

    The ScopedTimer ensures that durations are logged and consumed properly,
    preventing accidental overwriting of previous measurements.

    Usage:
        timer = ScopedTimer()

        # All durations are logged in the timer
        with timer("step_time"):
            with timer("fwd"):
                model.fwd()
            with timer("bwd"):
                model.bwd()

        # Consume all durations and reset internal store
        durations = timer.consume_durations()

        # Durations that are not consumed will raise a ValueError
        with timer("fwd"):
            model.fwd()
        with timer("fwd"):
            model.fwd()  # <-- This will raise an error as timer.consume_durations()
                         # is not called, meaning the previous measurement is
                         # still stored.

    Methods:
        consume_durations() -> dict[str, float]:
            Returns a dictionary of all logged durations and resets the internal log.

        __call__(name: str):
            Context manager for timing a section of code. Raises a ValueError if
            durations are not consumed before starting a new measurement for the
            same name.

    Raises:
        ValueError: If attempting to start a new timing section for a name that
                    already has a recorded duration without consuming the previous
                    measurement using consume_durations().
    """

    def __init__(self, *args, **kwargs):
        self._timer = NamedTimer(*args, **kwargs)
        self._duration_log = {}

    def consume_durations(self) -> dict[str, float]:
        durations = self._duration_log
        self._duration_log = {}
        self._timer.reset()
        return durations

    @contextmanager
    def __call__(self, name: str):
        try:
            self._timer.start(name=name)
            yield
        finally:
            self._timer.stop(name=name)
            if name in self._duration_log:
                raise ValueError(
                    f"Attempted to store new duration for {name=} before consuming last measurement. Call consume_durations() to consume the last set of measurements."
                )
            self._duration_log[name] = self._timer.get(name=name)
