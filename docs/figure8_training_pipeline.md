# TurboPi Figure-8 Training Pipeline

This document is the running reference for the figure-8 map work. Keep it updated whenever the data collection, training, inference, or video export flow changes.

## Current Scope

The current target is instruction-conditioned driving on the `figure8` map:

- `go_left`: drive the shared center arm, take the left loop, return to the shared start, and repeat.
- `go_right`: drive the shared center arm, take the right loop, return to the shared start, and repeat.
- One episode defaults to `3` complete loops for a single intent.
- The model input image is the robot-mounted forward camera.
- The language intent is saved as the instruction string, currently `go left` or `go right`.
- The action is the autonomous teacher command normalized for ACT/CVAE training.

The original cloned mountain route is preserved as the `original` map. The figure-8 route is an alternate map named `figure8`.

## Files

- `scripts/mountain_cliff_scene.py`
  - Defines the `original` and `figure8` maps.
  - `MountainCliffSceneCfg(map_name="figure8")` selects the new map.
  - `route_waypoints(scene_cfg, task_name)` returns the left or right route for collection.
  - Guard rails and end caps are disabled for `figure8`, so the car has an unobstructed path.

- `scripts/record_turbopi_mountain_act.py`
  - Records ACT episodes from the robot-facing camera.
  - Supports `--map figure8`, `--task go_left|go_right|mix`, and `--laps 3`.
  - Adds per-episode diversity through start jitter, yaw jitter, speed jitter, action noise, and camera-pose jitter.

- `scripts/collect_figure8_act_parallel.sh`
  - Launches separate headless Isaac workers for left and right data collection.
  - This is the fast path for generating balanced left/right data without manual interaction.

- `scripts/act_mountain_dataset.py`
  - Writes each episode as `video.mp4`, `data.parquet`, and `episode_info.json`.
  - Session metadata records the map name and laps per episode.

## What One Episode Contains

For each episode:

1. The robot is reset near the start of the shared straight segment.
2. A single intent is chosen: `go_left` or `go_right`.
3. The waypoint route is repeated `--laps` times.
4. The autonomous teacher drives the car along the route.
5. At each control step, the recorder saves:
   - RGB image from the robot-mounted forward camera.
   - Previous action as state.
   - Current normalized action.
   - Raw velocity command `[vx, vy, wz, stop]`.
   - Track error.
   - Route progress.
   - Task name, task index, and language instruction.
6. The episode is accepted only if the route is completed successfully.

With the default `--laps 3`, a left episode contains three left loops, and a right episode contains three right loops.

## Diversity Sources

The collector is intentionally deterministic enough to stay on the track, but varied enough to avoid a brittle dataset.

- `--speed_jitter`
  - Randomizes teacher speed per episode.
  - Default: `0.15`, meaning plus or minus 15 percent around `--target_speed`.

- `--start_xy_jitter`
  - Randomizes the initial x/y position near the route start.
  - Default: `0.025` meters.

- `--start_yaw_jitter`
  - Randomizes initial heading.
  - Default: `0.08` radians.

- `--camera_xyz_jitter`
  - Randomizes the forward camera eye and target offsets per episode.
  - Default: `0.015` meters.

- `--action_noise_std`
  - Optional command noise.
  - Default: `0.0`; increase carefully if the teacher remains stable.

## Fast Collection

The fastest practical setup right now is process-level parallelism:

```bash
cd /workspace/turbopi_isaac
EPISODES_PER_WORKER=100 WORKERS_PER_INTENT=1 ./scripts/collect_figure8_act_parallel.sh
```

This launches two headless Isaac jobs:

- one worker for `go_left`
- one worker for `go_right`

Each worker writes a separate session under:

```text
data/act_figure8/
```

For more parallelism on a large GPU:

```bash
cd /workspace/turbopi_isaac
EPISODES_PER_WORKER=100 WORKERS_PER_INTENT=2 ./scripts/collect_figure8_act_parallel.sh
```

That launches four jobs total: two left workers and two right workers. Watch GPU memory before increasing worker count.

## Single-Worker Commands

Left-only:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p /workspace/turbopi_isaac/scripts/record_turbopi_mountain_act.py \
  --headless \
  --map figure8 \
  --task go_left \
  --laps 3 \
  --num_episodes 50 \
  --output_dir /workspace/turbopi_isaac/data/act_figure8 \
  --session_name figure8_left_debug \
  --no_rollers
```

Right-only:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p /workspace/turbopi_isaac/scripts/record_turbopi_mountain_act.py \
  --headless \
  --map figure8 \
  --task go_right \
  --laps 3 \
  --num_episodes 50 \
  --output_dir /workspace/turbopi_isaac/data/act_figure8 \
  --session_name figure8_right_debug \
  --no_rollers
```

Balanced mixed collection in one process:

```bash
cd /workspace/isaaclab
./isaaclab.sh -p /workspace/turbopi_isaac/scripts/record_turbopi_mountain_act.py \
  --headless \
  --map figure8 \
  --task mix \
  --laps 3 \
  --num_episodes 100 \
  --output_dir /workspace/turbopi_isaac/data/act_figure8 \
  --session_name figure8_mix_debug \
  --no_rollers
```

## Why This Is Fast

- Headless Isaac avoids viewer overhead.
- `--no_rollers` skips procedural mecanum roller generation; the policy camera does not need those details.
- The route is kinematically followed by the teacher, so collection is stable and does not waste many failed attempts.
- Left and right intents can be collected in parallel workers.
- Each episode contains three loops, so simulator startup cost is amortized over more useful frames.
- Small domain randomization is applied per episode instead of restarting or rebuilding the scene for every variation.

## Dataset Layout

Each session contains:

```text
task_mapping.json
tasks.json
session_info.json
episode_00000/
  video.mp4
  data.parquet
  episode_info.json
episode_00001/
  ...
```

`data.parquet` contains one row per control step. The main columns are:

- `state`
- `action`
- `command`
- `track_error`
- `route_progress`
- `task`
- `task_index`
- `instruction`

The RGB frames are stored in `video.mp4`.

## Update Log

- Added the `figure8` alternate map while preserving `original`.
- Removed guard rails and end caps from `figure8`.
- Added `--laps`, defaulting to three loops per episode.
- Added per-episode speed, start pose, and camera jitter.
- Added `scripts/collect_figure8_act_parallel.sh` for balanced parallel left/right collection.

## Next Sections To Add

When implemented, extend this document with:

- Training commands and dataset selection.
- Model checkpoint layout.
- Inference commands on the figure-8 map.
- How to export and download inference videos.
- Recommended figures for reports or slides.
