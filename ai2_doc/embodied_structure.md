# Embodied Structure

This repo keeps the embodied stack split across four top-level domains:

- `rlinf/runners/`: training-loop orchestration such as `EmbodiedRunner`.
- `rlinf/workers/`: distributed training roles such as actor, rollout, and env-side workers.
- `rlinf/envs/`: environment implementations and environment runtime/transport code.
- `rlinf/integrations/`: external sidecars such as `marl`.

For the YAM + marl path, the intended mapping is:

- `rlinf/workers/env/remote_yam_env_worker.py`
  This is the env-side training worker. It owns rollout-to-env interaction state such as episode ids, step ids, marl reward windows, and subtask timing.
- `rlinf/envs/yam/yam_env.py`
  This is the concrete environment implementation.
- `rlinf/envs/yam/remote/robot_server.py`
  This exposes `YAMEnv` over gRPC on the desktop side.
- `rlinf/envs/yam/remote/robot_server_client.py`
  This is the Beaker-side gRPC wrapper used by `RemoteYamEnvWorker`.
- `rlinf/integrations/marl/client.py`
  This is the HTTP client for the marl sidecar.

The naming rule is:

- `workers/env/*` means training-side env orchestration.
- `envs/yam/*` means the YAM environment domain.
- `envs/yam/remote/*` means the remote runtime and transport for YAM.

This keeps the distinction explicit:

- `envs/*` answers "what is the environment and how is it exposed?"
- `workers/env/*` answers "which training worker drives the environment loop?"
