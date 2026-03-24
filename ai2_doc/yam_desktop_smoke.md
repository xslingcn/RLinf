# YAM Desktop Smoke Validation

This runbook describes the desktop-local validation path for a machine that is
directly attached to the YAM robot hardware.

The immediate goal is not full Beaker training. The immediate goal is to prove
that the current RLinf stack can make the following path work end to end:

```text
OpenPI rollout -> RemoteYamEnvWorker -> RobotServerClient -> RobotServer -> YAM follower -> robot arms
```

At minimum, this should validate:

- the desktop `RobotServer` can reset and stream observations
- the OpenPI rollout worker can consume those observations
- action chunks can be produced by the current RLinf OpenPI stack
- those action chunks can reach the follower-backed YAM runtime
- the robot arms move through the `RobotServer` path rather than a separate local-only client

This document intentionally does not cover the Beaker flow in detail yet.

## When To Use This Document

Use this scenario when one desktop has:

- the YAM hardware attached locally
- access to the RealSense cameras
- access to the follower arm servers
- at least one GPU for local OpenPI inference

This is the recommended first validation step because it removes Beaker,
reverse SSH tunnels, and Tailscale routing from the problem.

## Step 1: Start the Local Robot Server

From the RLinf worktree:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers \
    --no-tunnel
```

This launches:

- the right follower server on `localhost:1234`
- the left follower server on `localhost:1235`
- `RobotServer` on `localhost:50051`

The recommended config is:

- `examples/embodiment/config/env/yam_pi05_follower.yaml`

## Step 2: Verify gRPC Camera Transport Before Policy Inference

Before blaming OpenPI or RLinf, first confirm that `RobotServer` is actually
serving images:

```bash
python scripts/preview_grpc_cameras.py --url localhost:50051
```

This is the quickest way to catch:

- missing cameras
- wrong serial numbers
- follower startup failures that prevent reset
- broken `RobotServer` startup

If this step fails, stop here and fix the desktop-side robot runtime first.

## Step 3: Run the Current RLinf Smoke Path

Use the desktop smoke helper:

```bash
bash scripts/run_yam_desktop_smoke.sh \
    --config yam_ppo_openpi \
    --model-path /path/to/pi05 \
    --task "fold the towel"
```

What this helper does:

- starts a local dummy `marl` server by default on `http://127.0.0.1:18080`
- launches `examples/embodiment/eval_embodied_agent_remote_yam.py`
- runs rollout-only evaluation with `RemoteYamEnvWorker`
- keeps the current `RobotServer` / gRPC / remote-env path intact
- avoids PPO actor updates so the smoke path stays small and debuggable

Important detail:

- `RemoteYamEnvWorker` currently requires `marl.enabled=true`, even in eval-only mode
- for desktop smoke validation, dummy `marl` is enough because the goal here is
  the OpenPI-to-robot control path, not planner or TOPReward fidelity

## What This Smoke Path Validates

The desktop smoke path validates the current production-style RLinf chain:

- `MultiStepRolloutWorker` can load the OpenPI policy
- observations from `RobotServerClient` reach the rollout worker
- rollout actions are sent back through `RemoteYamEnvWorker`
- `ChunkStep` reaches `RobotServer`
- `RobotServer` forwards actions to the follower-backed YAM runtime
- the robot moves through the same gRPC path that Beaker will use later

This is the right smoke test when you want confidence in:

- `openpi -> roboarm` connection
- observation formatting
- action-shape compatibility
- the current RLinf remote-YAM integration

## What This Smoke Path Does Not Validate

This path does not fully validate:

- PPO training updates
- actor FSDP training
- real `marl` planner / TOPReward behavior
- Beaker networking
- reverse SSH tunnel setup
- Tailscale routing

Those belong to the later Beaker stage.

## Optional Overrides

If `RobotServer` is not running on the default port:

```bash
bash scripts/run_yam_desktop_smoke.sh \
    --model-path /path/to/pi05 \
    --task "fold the towel" \
    --robot-server-url localhost:50052
```

If you want to reuse an already-running `marl` endpoint:

```bash
bash scripts/run_yam_desktop_smoke.sh \
    --model-path /path/to/pi05 \
    --task "fold the towel" \
    --marl-base-url http://127.0.0.1:18080 \
    --no-start-marl
```

If you intentionally want to override the pinned OpenPI source with a sibling
editable checkout:

```bash
bash scripts/run_yam_desktop_smoke.sh \
    --model-path /path/to/pi05 \
    --task "fold the towel" \
    --editable-openpi \
    --openpi-repo-dir ../openpi
```

## Alternative: Direct Local PI0.5 Inference

If you only want a local policy client and do not need the current
`RemoteYamEnvWorker` chain, use:

- `ai2_doc/yam_pi05_local_inference.md`

That path uses `examples/embodiment/infer_embodied_agent.py` and is useful for
policy-only local inference debugging, but it is not the same as the current
RLinf remote-YAM integration used by the Beaker workflow.

## Beaker

Beaker is intentionally out of scope for this document for now.

Once the desktop smoke path is stable, the next step is to switch to the normal
single-node Beaker topology:

- desktop runs only `RobotServer`
- Beaker runs actor, rollout, and `marl`
- `RemoteYamEnvWorker` talks to the desktop over gRPC via the reverse SSH tunnel

When you are ready for that stage, use these docs instead:

- `ai2_doc/yam_marl_runbook.md`
- `ai2_doc/yam_ppo_openpi.md`
- `ai2_doc/yam_ppo_openpi_topreward.md`
- `ai2_doc/quickstart.md`

## Important Files

- `scripts/start_robot_server.sh`
- `scripts/preview_grpc_cameras.py`
- `scripts/run_yam_desktop_smoke.sh`
- `examples/embodiment/eval_embodied_agent_remote_yam.py`
- `examples/embodiment/config/env/yam_pi05_follower.yaml`
