# YAM PPO + TOPReward (`yam_ppo_openpi`)

This is the AI2-facing Markdown guide for the baseline YAM PPO config:
`examples/embodiment/config/yam_ppo_openpi.yaml`.

This config runs:

- PPO + GAE
- π₀.5 / OpenPI policy
- TOPReward dense reward
- no subtask planning (`marl.planner.interval: 0`)

For the variant that also enables subtask planning, see
[yam_ppo_openpi_topreward](yam_ppo_openpi_topreward.md).

## Topology

Canonical topology:

- Beaker runs all Ray workers
- the desktop runs only `RobotServer`
- `RemoteYamEnvWorker` connects over gRPC through the reverse SSH tunnel

Main component placement:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: `marl` sidecar
- CPU: `RemoteYamEnvWorker`

## marl Assumptions

The current baseline does not use the old in-process `VLMPlannerWorker` path.
Instead:

- `marl` runs as a local sidecar on GPU 2
- planner and TOPReward share one Qwen3-VL runtime inside `marl`
- RLinf owns reward-delta logic and replanning cadence
- planner inputs are richer than the legacy worker path: multi-view images and
  `memory_text` are both intentional

See [marl_alignment_findings](marl_alignment_findings.md) for prompt alignment
details.

## Standard Workflow

Submit training from the repo root:

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --model-path /path/to/RLinf-Pi05-SFT
```

Then start the desktop-side robot server:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers \
    --remote-host <tailscale-ip>
```

For pipeline testing without hardware:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --dummy
```

## Key Config Knobs

Model paths:

```yaml
rollout:
  model:
    model_path: "/path/to/RLinf-Pi05-SFT"

actor:
  model:
    model_path: "/path/to/RLinf-Pi05-SFT"
```

Task description:

```yaml
env:
  train:
    task_description: "pick up the red block"
```

Reward / planner settings:

```yaml
marl:
  topreward:
    enabled: true
  planner:
    interval: 0
```

## Local Simulated Desktop Mode

`train_embodied_agent_marl.py` supports simulating the remote desktop input
path locally. This keeps the normal `RemoteEnv -> gRPC -> RobotServer` flow,
but the training process starts a local dummy `RobotServer` automatically, so
no separate desktop machine or reverse SSH tunnel is needed.

Enable it with:

```bash
bash scripts/run_yam_marl_training.sh \
    --config yam_ppo_openpi \
    --model-path /path/to/RLinf-Pi05-SFT \
    -- \
    env.remote_desktop_simulation.enabled=true
```

Optional overrides:

```bash
bash scripts/run_yam_marl_training.sh \
    --config yam_ppo_openpi \
    --model-path /path/to/RLinf-Pi05-SFT \
    -- \
    env.remote_desktop_simulation.enabled=true \
    env.remote_desktop_simulation.env_config_path=/path/to/yam_pi05_follower.yaml \
    env.train.remote_server_url=localhost:50051 \
    env.eval.remote_server_url=localhost:50051
```

Shared config block:

```yaml
env:
  remote_desktop_simulation:
    enabled: true
    dummy: true
    env_config_path: null
    startup_timeout: 30.0
```

Notes:

- Only local RemoteEnv URLs are supported in this mode, such as
  `localhost:50051` or `127.0.0.1:50051`.
- This mode is for dummy input simulation only.
- For real hardware, keep using `start_robot_server.sh` on the desktop.

## Related Docs

- [yam_desktop_smoke](yam_desktop_smoke.md)
- [quickstart](quickstart.md)
- [training_architecture](training_architecture.md)
- [network_infrastructure](network_infrastructure.md)
- [marl_alignment_findings](marl_alignment_findings.md)
- [openpi_joint_logprob](openpi_joint_logprob.md)
- [openpi_nan_gradients](openpi_nan_gradients.md)
