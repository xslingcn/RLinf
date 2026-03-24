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

"""Entry point for embodied RL using a marl sidecar for planning and TOPReward."""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.envs.yam.remote.simulated_desktop import (
    launch_simulated_desktop_server,
    stop_process,
)
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.remote_yam_env_worker import RemoteYamEnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name=None,
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    simulated_desktop_server = launch_simulated_desktop_server(cfg)
    try:
        cluster = Cluster(
            cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
        )
        component_placement = HybridComponentPlacement(cfg, cluster)

        actor_placement = component_placement.get_strategy("actor")
        if cfg.algorithm.loss_type == "embodied_sac":
            from rlinf.workers.actor.fsdp_sac_policy_worker import (
                EmbodiedSACFSDPPolicy,
            )

            actor_worker_cls = EmbodiedSACFSDPPolicy
        else:
            from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

            actor_worker_cls = EmbodiedFSDPActor

        actor_group = actor_worker_cls.create_group(cfg).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
        )

        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )

        env_placement = component_placement.get_strategy("env")
        env_group = RemoteYamEnvWorker.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

        runner = EmbodiedRunner(
            cfg=cfg,
            actor=actor_group,
            rollout=rollout_group,
            env=env_group,
        )

        runner.init_workers()
        runner.run()
    finally:
        stop_process(simulated_desktop_server)


if __name__ == "__main__":
    main()
