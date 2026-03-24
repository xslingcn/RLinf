# YAM + marl Runbook

This document describes the exact setup for the current YAM + `marl` stack on
the `marl/aggressive-cleaning` baseline.

Use this setup if you want all of these at once:

- `RLinf` on Beaker
- `marl` as a Beaker sidecar
- desktop `RobotServer` over reverse SSH
- local forked `openpi`
- local forked `sglang`

This runbook assumes:

- RLinf is on the `marl/aggressive-cleaning` baseline
- `marl` is on the current `feat/marl` assumptions
- you want the exact sibling-checkout workflow on Beaker

## 1. Repo Layout

The required sibling layout is:

```text
<root>/
  RLinf/
  marl/
  sglang/
  openpi/   # optional: only if you want an editable local override
```

Clone exactly these repos:

```bash
git clone https://github.com/xslingcn/marl
git clone -b marl https://github.com/xslingcn/sglang
# Optional editable override for the pinned OpenPI source:
git clone https://github.com/RLinf/openpi
```

Use exactly these sources:

- `marl`: `https://github.com/xslingcn/marl`
- `sglang`: `https://github.com/xslingcn/sglang/tree/marl`
- default OpenPI source pinned by RLinf: `https://github.com/RLinf/openpi`
- optional editable OpenPI checkout: any local checkout you want to force with
  `--editable-openpi`

Why the sibling layout matters:

- `marl` expects the patched `sglang` checkout at `../sglang/python`
- `submit_yam_training.sh` defaults `MARL_REPO_DIR` to the sibling repo `../marl`
- `scripts/run_yam_marl_training.sh` can reuse `../openpi` as an editable
  override, but only when you explicitly pass `--editable-openpi`

Note:

- current `RLinf/pyproject.toml` pins `RLinf/openpi@54cbaee...`
- the default training path should now use that pinned official source
- `--editable-openpi` is only for cases where you intentionally want to
  override the pinned source with your own local checkout or worktree

## 2. Models And Secrets

You need:

- an OpenPI checkpoint `thomas0829/folding_towel_pi05`
- a Qwen3-VL checkpoint for `marl`
  - configured in `marl/marl.yaml`
- Beaker secrets:
  - `hf_token_shirui`
  - `tailscale_authkey_shirui`

## 3. Start a Beaker Environment

If you want this exact fork-based stack, prefer either:

- `submit_yam_training.sh --interactive`
- `submit_yam_beaker_cluster.sh` + SSH into the idle container

Example interactive session:

```bash
cd RLinf

bash scripts/submit_yam_training.sh \
  --config yam_ppo_openpi_topreward \
  --interactive \
  --allow-dirty
```

Then attach:

```bash
beaker session attach <session-id>
```

Inside the container, the expected repo layout is still sibling-based, for
example:

```text
/weka/oe-training-default/shiruic/
  RLinf/
  marl/
  openpi/
  sglang/
```

## 4. Start the Desktop RobotServer

On the desktop:

```bash
cd RLinf

bash scripts/start_robot_server.sh \
  --config examples/embodiment/config/env/yam_pi05_follower.yaml \
  --use-follower-servers \
  --remote-host beaker-0
```

Dummy hardware mode:

```bash
bash scripts/start_robot_server.sh \
  --config examples/embodiment/config/env/yam_pi05_follower.yaml \
  --dummy
```

This starts:

- follower servers, if requested
- `RobotServer`
- a persistent reverse SSH tunnel to Beaker

`RLinf` always connects to `localhost:50051` on Beaker. The tunnel maps that
back to the desktop `RobotServer`.

## 5. Start Training on Beaker

Inside the Beaker shell:

```bash
cd /weka/oe-training-default/shiruic/RLinf

bash scripts/run_yam_marl_training.sh \
  --config yam_ppo_openpi_topreward \
  --model-path thomas0829/folding_towel_pi05
```

If you want the reward-only variant, switch the config:

```bash
--config yam_ppo_openpi
```

`run_yam_marl_training.sh` does three important things for this workflow:

- starts `marl` on GPU 2 if it is not already running
- defaults to the pinned official `RLinf/openpi` source from `pyproject.toml`
- can switch to a local checkout only when you explicitly pass
  `--editable-openpi`
- keeps the `chex==0.1.90` pin that matches the current OpenPI/JAX path

If you do want to force a local OpenPI checkout, run:

```bash
bash scripts/run_yam_marl_training.sh \
  --config yam_ppo_openpi_topreward \
  --model-path thomas0829/folding_towel_pi05 \
  --editable-openpi \
  --openpi-repo-dir /weka/oe-training-default/shiruic/openpi
```

If you want to inspect or control `marl` separately, you can still launch it
manually:

```bash
export MARL_REPO_DIR=/weka/oe-training-default/shiruic/marl
export MARL_CONFIG_PATH=${MARL_REPO_DIR}/marl.yaml
export MARL_BASE_URL=http://127.0.0.1:8080

nohup env CUDA_VISIBLE_DEVICES=2 \
  uv run --project ${MARL_REPO_DIR} --python 3.12 \
  python -m marl --config ${MARL_CONFIG_PATH} --log-level info \
  > ${MARL_REPO_DIR}/marl_server.log 2>&1 &
```

Then run RLinf without re-starting the sidecar:

```bash
bash scripts/run_yam_marl_training.sh \
  --config yam_ppo_openpi_topreward \
  --model-path thomas0829/folding_towel_pi05 \
  --no-start-marl
```

## 6. Actual Runtime Topology

This exact setup gives:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: `marl`
- CPU: `RemoteYamEnvWorker` + `RobotServerClient`
- desktop: `RobotServer` + optional follower servers

The runtime path is:

```text
OpenPI rollout -> RemoteYamEnvWorker -> RobotServerClient -> RobotServer -> YAMEnv
                                 |
                                 +-> marl /image-sets
                                 +-> marl /topreward
                                 +-> marl /plan
```

For `yam_ppo_openpi_topreward`:

- `/topreward` is used for dense reward
- `/plan` is used for subtask planning

For `yam_ppo_openpi`:

- `/topreward` is used
- `/plan` is not used

## 7. Current marl / MRL Assumptions

This runbook assumes the current `marl` integration contract:

- `marl` returns absolute TOPReward scores; RLinf converts them to deltas
- planner prompt wording is aligned with the legacy RLinf worker path
- planner inputs are richer than the legacy path: multi-view images and
  accumulated `memory_text` are intentional

That means you should not expect planner behavior to exactly match the older
`VLMPlannerWorker` path even if the prompt strings look aligned.

See [marl_alignment_findings](marl_alignment_findings.md) for the exact prompt
and input-contract differences.

## 8. Local 5-Step Smoke Test

Use this only for pipeline validation, not real training.

Start dummy `marl`:

```bash
cd marl

uv run --project . python -m marl.dummy_server \
  --host 127.0.0.1 \
  --port 18080 \
  --planner-prefix dummy-subtask \
  --reward-mode global_step
```

Then from `RLinf`:

```bash
bash scripts/run_yam_marl_training.sh \
  --config yam_ppo_openpi_topreward \
  --model-path /path/to/folding_towel_pi05 \
  --marl-base-url http://127.0.0.1:18080 \
  --no-start-marl \
  -- \
  runner.max_epochs=1 \
  runner.logger.log_path=/tmp/rlinf-smoke-5step \
  env.remote_desktop_simulation.enabled=true \
  env.remote_desktop_simulation.dummy=true \
  env.remote_desktop_simulation.env_config_path=examples/embodiment/config/env/yam_pi05_follower_dummy.yaml \
  env.train.max_steps_per_rollout_epoch=50 \
  env.eval.max_steps_per_rollout_epoch=50 \
  marl.planner.interval=1 \
  cluster.component_placement.actor.placement=0 \
  cluster.component_placement.rollout.placement=0 \
  cluster.component_placement.env.placement=0 \
  actor.micro_batch_size=5 \
  actor.global_batch_size=5
```

This validates:

- `obs -> marl /image-sets`
- `obs -> marl /topreward`
- `obs -> marl /plan`
- single-node Beaker-style `marl + RLinf` wiring
