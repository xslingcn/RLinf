# YAM Training Architecture

This document describes the current YAM PPO training loop on the
`marl/aggressive-cleaning` baseline: single-node Beaker, 3 GPUs, desktop
`RobotServer`, and a local `marl` sidecar for planning + TOPReward.

For setup and first-run instructions, see [quickstart](quickstart.md). For
network details, see [network_infrastructure](network_infrastructure.md). For
the exact fork-based workflow, see [yam_marl_runbook](yam_marl_runbook.md).

## Supported Configs

| Config | Algorithm | Policy | Reward | Subtask Planning | Entry Point | GPUs | Beaker Script |
|---|---|---|---|---|---|---|---|
| `yam_ppo_openpi` | PPO + GAE | π₀.5 (OpenPI, diffusion) | TOPReward | no (`marl.planner.interval: 0`) | `train_embodied_agent_marl.py` | 3 | `submit_yam_training.sh` |
| `yam_ppo_openpi_topreward` | PPO + GAE | π₀.5 (OpenPI, diffusion) | TOPReward | yes (`marl.planner.interval: 3`) | `train_embodied_agent_marl.py` | 3 | `submit_yam_training.sh` |

Both configs run with:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: `marl`
- CPU: `RemoteYamEnvWorker`

The recommended manual entrypoint inside Beaker is now
`scripts/run_yam_marl_training.sh`, which starts the `marl` sidecar if needed
and then launches `train_embodied_agent_marl.py`.

## Current Assumptions

The current stack is built around these assumptions:

- `marl` is a single-process local controller service on the Beaker node.
- `marl` owns image ingress persistence, planner generation, and TOPReward
  absolute scoring.
- RLinf owns reward-delta logic, replanning cadence, episode resets, and task
  updates.
- `RemoteYamEnvWorker` talks to the desktop `RobotServer` over gRPC and to
  `marl` over local HTTP.
- Planner prompt wording is aligned with the old RLinf worker path, but planner
  inputs are intentionally richer now: the current path can pass multi-view
  images plus accumulated `memory_text`.

Prompt-level alignment details and remaining non-alignment are documented in
[marl_alignment_findings](marl_alignment_findings.md).

## Data Flow

The current production path is:

```text
Desktop RobotServer                 Beaker Node
──────────────────                 ─────────────────────────────────────────────
YAMEnv
  ▲
  │ gRPC (:50051 via reverse SSH)
  │
RobotServer  ◄──────────────────── RemoteYamEnvWorker
                                        │
                                        │ POST /image-sets
                                        │ POST /topreward
                                        │ POST /plan
                                        ▼
                                  marl sidecar (GPU 2)
                                        │
                                        ▼
                                shared Qwen3-VL runtime

RolloutWorker (GPU 1)  ── actions ─────► RemoteYamEnvWorker
RolloutWorker (GPU 1)  ◄─ obs/reward ─── RemoteYamEnvWorker
ActorWorker   (GPU 0)  ◄─ trajectories ─ RolloutWorker
ActorWorker   (GPU 0)  ── weights ─────► RolloutWorker
```

What changes compared with the legacy in-process VLM worker path:

- there is no `VLMPlannerWorker` Ray actor in the canonical YAM configs
- planner and TOPReward share the same `marl` runtime
- RLinf does not ask the model for delta reward directly; it calls
  `/topreward`, gets an absolute score, and computes the per-step delta locally

## Per-Epoch Loop

At a high level:

1. `EmbodiedRunner` launches actor, rollout, and `RemoteYamEnvWorker`.
2. `RemoteYamEnvWorker.bootstrap_step()` resets the robot at the start of the
   rollout epoch.
3. `RolloutWorker` generates one chunk of actions.
4. `RemoteYamEnvWorker` sends the action chunk over gRPC to `RobotServer`.
5. The env worker stores the latest observation in `marl` via `/image-sets`.
6. If TOPReward is enabled, the env worker calls `/topreward`, receives an
   absolute score, and converts it to a reward delta.
7. If replanning is enabled and the chunk-step counter reaches
   `marl.planner.interval`, the env worker calls `/plan` and updates the task
   description used for future rollout inputs.
8. Rollout trajectories go to the actor; the actor computes GAE + PPO updates
   and syncs weights back to rollout.

## Important Semantics

### `collect_prev_infos: true` is required for GAE

Both YAM configs use `adv_type: gae` and therefore need
`rollout.collect_prev_infos: true`. If it is disabled, training does not crash,
but it silently degrades to a critic-free fallback because `prev_values` are
missing.

### `bootstrap_step()` resets the robot every training step

For these configs, `auto_reset: false` and `rollout_epoch: 1`. That means the
robot is reset once per training step via `bootstrap_step()`.

### `marl.planner.interval` is measured in chunk steps

The current configs use:

- `max_steps_per_rollout_epoch: 100`
- `num_action_chunks: 10`

So one rollout epoch has `100 // 10 = 10` chunk steps. With
`marl.planner.interval: 3`, replanning happens at chunk steps 3, 6, and 9.

### TOPReward is absolute in `marl`, delta in RLinf

`marl /topreward` returns an absolute score for the selected trajectory prefix.
`RemoteYamEnvWorker` stores the previous score and injects the per-step delta:

```text
reward_t = score_t - score_(t-1)
```

The baseline is reset:

- at epoch start
- when an episode ends
- when the planner changes the task description

This avoids mixing scores from different instructions.

### Planner input contract is richer than the legacy RLinf worker path

Current `marl`-based replanning can use:

- `main` images
- `wrist_*` images
- `extra_*` images
- accumulated `memory_text`

So prompt wording is aligned, but effective planner context is intentionally
broader than the old worker-based path.

### Single-node 3-GPU placement is explicit

The YAM marl configs set:

- actor placement: GPU 0
- rollout placement: GPU 1
- env placement: same node, CPU

The `marl` sidecar is not a Ray worker. The Beaker-side launch scripts pin it
to GPU 2 with `CUDA_VISIBLE_DEVICES=2`.

### YAM action dimension is 14

YAM is bimanual (`2 × 7 DOF`), so both configs set:

```yaml
actor:
  model:
    action_dim: 14
```

Without this override, OpenPI would silently use the single-arm template
default.

### `global_batch_size` must divide rollout size

For the default YAM setup:

```text
n_train_chunk_steps = 100 // 10 = 10
rollout_size = 10 × total_num_envs × rollout_epoch = 10
```

So `actor.global_batch_size` must divide 10. The default configs use:

```yaml
actor:
  micro_batch_size: 10
  global_batch_size: 10
```

## Code Reference

| Term | Code component | Location |
|---|---|---|
| Beaker YAM entrypoint | `train_embodied_agent_marl.py` | `examples/embodiment/train_embodied_agent_marl.py` |
| Manual single-node launcher | `run_yam_marl_training.sh` | `scripts/run_yam_marl_training.sh` |
| Env worker | `RemoteYamEnvWorker` | `rlinf/workers/env/remote_yam_env_worker.py` |
| marl client | `MarlClient` | `rlinf/integrations/marl/client.py` |
| Rollout worker | `MultiStepRolloutWorker` | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| Actor worker | `EmbodiedFSDPActor` | `rlinf/workers/actor/fsdp_actor_worker.py` |
| Simulated desktop helper | `launch_simulated_desktop_server` | `rlinf/envs/yam/remote/simulated_desktop.py` |
| Desktop gRPC server | `RobotServer` | `rlinf/envs/yam/remote/robot_server.py` |

## Operational Notes

- Use `submit_yam_training.sh` for the normal single-node Beaker run.
- Use `submit_yam_beaker_cluster.sh` when you want an idle Beaker node and plan
  to SSH in repeatedly for manual debugging.
- Use `run_yam_marl_training.sh` inside Beaker when you need the exact
  `marl + RLinf` loop with sibling `openpi` and the `chex==0.1.90` pin.
- Use `env.remote_desktop_simulation.enabled=true` only for local dummy
  pipeline validation. Real hardware still requires the desktop
  `start_robot_server.sh` path.
