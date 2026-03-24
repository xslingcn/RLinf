# Local YAM + LeRobot PI0.5 Runbook

This runbook describes the simplest local setup for running a LeRobot-exported
`pi05` checkpoint against a YAM follower on a desktop machine.

This branch uses the **LeRobot-only inference lane**. The legacy OpenPI lane is
disabled here on purpose.

## Scope

This runbook covers:

- a local desktop machine connected to the YAM follower
- `YAMEnv` running locally on the desktop
- `RobotServer` exposing the robot over local gRPC
- `infer_embodied_agent.py` running a `lerobot_pi05` policy against that server

This runbook does **not** cover:

- Beaker / remote training
- the old OpenPI configs
- direct desktop PPO training via `yam_ppo_openpi_desktop.yaml`

## Preconditions

You need:

- this branch checked out
- the project environment synced
- a local LeRobot `pi05` checkpoint, for example:
  - `/home/xsling/Model/folding_towel_pi05`
- local YAM follower servers or the checked-in follower launcher
- camera configs if you want live images

Recommended setup:

```bash
cd /home/xsling/Code/mma2rl/RLinf-lerobot
uv sync --python 3.12.3 --extra embodied
source .venv/bin/activate
```

## Camera Assumptions

The local `folding_towel_pi05` bundle expects three image streams:

- `observation.images.top`
- `observation.images.left`
- `observation.images.right`

The RLinf LeRobot wrapper maps them by default as:

- `observation.images.top <- main_images`
- `observation.images.left <- wrist_images[0]`
- `observation.images.right <- wrist_images[1]`

So the safest local YAM setup is:

- one main camera
- two wrist cameras

If your camera layout is different, override it with `--camera-binding` when
starting inference.

## Step 1: Start from the checked-in follower env config

This branch now includes a concrete local follower config:

- [yam_pi05_follower.yaml](/home/xsling/Code/mma2rl/RLinf-lerobot/examples/embodiment/config/env/yam_pi05_follower.yaml)
- [yam_follower_left.yaml](/home/xsling/Code/mma2rl/RLinf-lerobot/examples/embodiment/config/env/robot_clients/yam_follower_left.yaml)
- [yam_follower_right.yaml](/home/xsling/Code/mma2rl/RLinf-lerobot/examples/embodiment/config/env/robot_clients/yam_follower_right.yaml)

It is wired for the standard local follower setup:

- right follower server on `127.0.0.1:1234`
- left follower server on `127.0.0.1:1235`
- top / left / right RealSense cameras at `640x360`

Before you start, edit this file if your hardware differs:

- `task_description`
- camera serial numbers under `camera_cfgs`
- CAN channel / port settings if your follower setup is non-default

The current checked-in shape is:

```yaml
env_type: yam
is_dummy: false
control_rate_hz: 30.0
max_episode_steps: 10000

img_height: 360
img_width: 640

main_camera: cam_top
wrist_cameras:
  - cam_left
  - cam_right

task_description: "fold the towel"

robot_cfgs:
  left:
    - examples/embodiment/config/env/robot_clients/yam_follower_left.yaml
  right:
    - examples/embodiment/config/env/robot_clients/yam_follower_right.yaml

camera_cfgs:
  cam_top:
    _target_: yam_realtime.sensors.cameras.camera.CameraNode
    camera:
      _target_: rlinf.envs.yam.realsense_camera.RealsenseCamera
      serial_number: "215222073684"
      resolution: [640, 360]
      fps: 30
  cam_left:
    _target_: yam_realtime.sensors.cameras.camera.CameraNode
    camera:
      _target_: rlinf.envs.yam.realsense_camera.RealsenseCamera
      serial_number: "218622275075"
      resolution: [640, 360]
      fps: 30
  cam_right:
    _target_: yam_realtime.sensors.cameras.camera.CameraNode
    camera:
      _target_: rlinf.envs.yam.realsense_camera.RealsenseCamera
      serial_number: "128422272697"
      resolution: [640, 360]
      fps: 30
```

Notes:

- `robot_cfgs.left/right` point to the follower-client configs on this branch.
- `camera_cfgs` is still machine-specific. Update serial numbers for your desktop.
- `img_height` and `img_width` should match what your server returns. The
  current local bundle was trained with raw image feature shapes
  `(3, 360, 640)`.

## Step 2: Start follower servers and the local RobotServer

Use the checked-in helper:

```bash
cd /home/xsling/Code/mma2rl/RLinf-lerobot
source .venv/bin/activate

bash scripts/start_robot_server.sh \
  --config examples/embodiment/config/env/yam_pi05_follower.yaml \
  --use-follower-servers \
  --no-tunnel
```

Notes:

- This resets CAN, starts both follower servers, and then starts the local
  `RobotServer` on `localhost:50051`.
- If you already run follower servers yourself, omit `--use-follower-servers`.
- If you need the old low-level path, `python -m rlinf.envs.yam.remote.robot_server`
  still works, but the helper above is the standard path on this branch.

## Step 3: Preview the camera streams

Open a second terminal:

```bash
cd /home/xsling/Code/mma2rl/RLinf-lerobot
source .venv/bin/activate

python scripts/preview_grpc_cameras.py --url localhost:50051
```

Check that you see:

- the main camera feed
- both wrist camera feeds
- stable refresh without gRPC errors

If this step is wrong, fix the env YAML before trying policy inference.

## Step 4: Run local policy inference

Open a third terminal:

```bash
cd /home/xsling/Code/mma2rl/RLinf-lerobot
source .venv/bin/activate

python examples/embodiment/infer_embodied_agent.py \
  --model-type lerobot_pi05 \
  --model-path /home/xsling/Model/folding_towel_pi05 \
  --server-url localhost:50051 \
  --task-description "fold the towel" \
  --action-chunk 2 \
  --num-steps 1
```

This uses:

- local RobotServer over `localhost:50051`
- the LeRobot `pi05` wrapper inside RLinf
- native LeRobot preprocessing and postprocessing

## Optional: Explicit camera bindings

If your local server does not expose:

- `main_images` as top view
- `wrist_images[0]` as left
- `wrist_images[1]` as right

then pass explicit bindings:

```bash
python examples/embodiment/infer_embodied_agent.py \
  --model-type lerobot_pi05 \
  --model-path /home/xsling/Model/folding_towel_pi05 \
  --server-url localhost:50051 \
  --task-description "fold the towel" \
  --action-chunk 2 \
  --num-steps 1 \
  --camera-binding observation.images.top=main_images \
  --camera-binding observation.images.left=wrist_images[0] \
  --camera-binding observation.images.right=wrist_images[1]
```

## Optional: Dummy smoke test

If you only want to validate the transport path without real hardware:

```bash
bash scripts/start_robot_server.sh \
  --config examples/embodiment/config/env/yam_pi05_follower.yaml \
  --no-tunnel \
  --dummy
```

Then run the same preview and inference commands against `localhost:50051`.

## Troubleshooting

### `ModuleNotFoundError: openpi`

This is expected if you try to use an old OpenPI config or code path on this
branch. Use:

- `--model-type lerobot_pi05`

Do not use:

- `model_type: openpi`
- the old `yam_ppo_openpi*.yaml` training configs

### The policy complains about missing cameras

The local bundle expects three image inputs. Verify:

- your server returns a main image
- your server returns two wrist images
- your bindings match the actual stream ordering

### Camera preview works, but actions are poor

Check:

- the camera semantic mapping is correct
- the robot state ordering matches the checkpoint expectation
- the task string matches the policy’s intended task

### I want direct desktop training with local YAMEnv

That path is separate from this runbook. On this branch, the old
`yam_ppo_openpi_desktop.yaml` config is not yet migrated to `lerobot_pi05`.

## Verified Path on This Branch

The following path is verified:

`YAMEnv -> RobotServer -> infer_embodied_agent.py -> lerobot_pi05 wrapper -> native LeRobot PI05Policy`

The RLinf wrapper has been parity-checked against native LeRobot on the local
`folding_towel_pi05` checkpoint for the same input and seed.
