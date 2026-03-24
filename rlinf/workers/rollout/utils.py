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

from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode


class RankMapper:
    @classmethod
    def get_actor_rank_to_rollout_rank_map(
        cls,
        placement: ModelParallelComponentPlacement,
    ) -> dict[int, list[tuple[int, int]]]:
        return cls._get_rank_mapper(
            placement.placement_mode
        ).get_actor_rank_to_rollout_rank_map(
            placement.actor_tp_size,
            placement.actor_pp_size,
            placement.actor_world_size,
            placement.rollout_tp_size,
            placement.rollout_world_size,
        )

    @classmethod
    def get_rollout_rank_to_actor_rank_map(
        cls, placement: ModelParallelComponentPlacement
    ) -> dict[tuple[int, int], int]:
        return cls._get_rank_mapper(
            placement.placement_mode
        ).get_rollout_rank_to_actor_rank_map(
            placement.actor_tp_size,
            placement.actor_pp_size,
            placement.actor_world_size,
            placement.rollout_tp_size,
            placement.rollout_world_size,
        )

    @staticmethod
    def _get_rank_mapper(placement_mode: PlacementMode):
        if placement_mode == PlacementMode.COLLOCATED:
            return CollocateRankMapper
        elif placement_mode in [PlacementMode.DISAGGREGATED, PlacementMode.AUTO]:
            return DisaggRankMapper
        else:
            raise ValueError(f"Unsupported mode: {placement_mode}.")


class CollocateRankMapper(RankMapper):
    @classmethod
    def get_actor_rank_to_rollout_rank_map(
        cls,
        actor_tp_size: int,
        actor_pp_size: int,
        actor_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> dict[int, tuple[int, int]]:
        if actor_tp_size == 1:
            return {
                rank: (rank // rollout_tp_size, rank % rollout_tp_size)
                for rank in range(actor_world_size)
            }
        rank_map = {}
        for actor_rank in range(actor_world_size):
            rank_map[actor_rank] = cls._get_actor_rank_to_rollout_rank(
                actor_rank,
                actor_tp_size,
                rollout_tp_size,
            )
        return rank_map

    @classmethod
    def get_rollout_rank_to_actor_rank_map(
        cls,
        actor_tp_size: int,
        actor_pp_size: int,
        actor_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ):
        rank_map = cls.get_actor_rank_to_rollout_rank_map(
            actor_tp_size,
            actor_pp_size,
            actor_world_size,
            rollout_tp_size,
            rollout_world_size,
        )
        return {v: k for k, v in rank_map.items()}

    @staticmethod
    def _get_actor_rank_to_rollout_rank(
        actor_rank: int,
        actor_tp_size: int,
        rollout_tp_size: int,
    ):
        num_rollout_dp_ranks_per_actor_tp_group = actor_tp_size // rollout_tp_size
        actor_tp_rank = actor_rank % actor_tp_size
        actor_tp_group_id = actor_rank // actor_tp_size
        rollout_start_dp_rank = (
            actor_tp_group_id * num_rollout_dp_ranks_per_actor_tp_group
        )
        weight_dst_dp_rank_in_rollout = (
            rollout_start_dp_rank
            + actor_tp_rank % num_rollout_dp_ranks_per_actor_tp_group
        )
        weight_dst_tp_rank_in_rollout = (
            actor_tp_rank // num_rollout_dp_ranks_per_actor_tp_group
        )
        return (weight_dst_dp_rank_in_rollout, weight_dst_tp_rank_in_rollout)


class DisaggRankMapper(RankMapper):
    @classmethod
    def get_actor_rank_to_rollout_rank_map(
        cls,
        actor_tp_size: int,
        actor_pp_size: int,
        actor_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> dict[int, list[tuple[int, int]]]:
        actor_model_parallel_size = actor_tp_size
        assert rollout_world_size >= actor_model_parallel_size, (
            f"rollout_world_size ({rollout_world_size}) should more than actor_model_parallel_size ({actor_model_parallel_size})"
        )
        assert rollout_world_size % actor_model_parallel_size == 0, (
            f"rollout_world_size ({rollout_world_size}) should be a multiple of actor_model_parallel_size ({actor_model_parallel_size})"
        )

        actor_dp = actor_world_size // actor_tp_size
        stride = actor_model_parallel_size // rollout_tp_size

        rank_map = {}
        for actor_rank in range(actor_world_size):
            if actor_rank > rollout_world_size:
                rank_map[actor_rank] = []
                continue
            gen_dp, gen_tp = cls._get_actor_rank_to_rollout_rank(
                actor_rank,
                actor_tp_size,
                rollout_tp_size,
            )
            if actor_world_size <= rollout_world_size:
                rank_map[actor_rank] = [
                    (gen_dp + i * stride * actor_dp, gen_tp)
                    for i in range(rollout_world_size // actor_world_size)
                ]
            elif actor_rank < rollout_world_size:
                rank_map[actor_rank] = [(gen_dp, gen_tp)]
            else:
                rank_map[actor_rank] = []

        return rank_map

    @classmethod
    def get_rollout_rank_to_actor_rank_map(
        cls,
        actor_tp_size: int,
        actor_pp_size: int,
        actor_world_size: int,
        rollout_tp_size: int,
        rollout_world_size: int,
    ) -> dict[tuple[int, int], int]:
        rank_map = cls.get_actor_rank_to_rollout_rank_map(
            actor_tp_size,
            actor_pp_size,
            actor_world_size,
            rollout_tp_size,
            rollout_world_size,
        )
        result_map = {}
        for actor_rank, rollout_2d_ranks in rank_map.items():
            for rollout_2d_rank in rollout_2d_ranks:
                result_map[rollout_2d_rank] = actor_rank
        return result_map

    @staticmethod
    def _get_actor_rank_to_rollout_rank(
        actor_rank: int,
        actor_tp_size: int,
        rollout_tp_size: int,
    ) -> tuple[int, int]:
        assert actor_tp_size % rollout_tp_size == 0, (
            "actor_tp_size must be a multiple of rollout_tp_size"
        )

        num_rollout_dp_ranks_per_actor_tp_group = actor_tp_size // rollout_tp_size
        actor_tp_rank = actor_rank % actor_tp_size
        actor_tp_group_id = actor_rank // actor_tp_size
        rollout_start_dp_rank = (
            actor_tp_group_id * num_rollout_dp_ranks_per_actor_tp_group
        )
        weight_dst_dp_rank_in_rollout = (
            rollout_start_dp_rank
            + actor_tp_rank % num_rollout_dp_ranks_per_actor_tp_group
        )
        weight_dst_tp_rank_in_rollout = (
            actor_tp_rank // num_rollout_dp_ranks_per_actor_tp_group
        )
        return (weight_dst_dp_rank_in_rollout, weight_dst_tp_rank_in_rollout)
