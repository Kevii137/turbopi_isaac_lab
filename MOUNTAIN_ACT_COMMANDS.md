# Mountain ACT Commands

Reusable commands for the mountain cliff ACT + language + CVAE pipeline.

## Train

Use the latest recorded mountain ACT session path. Keep `--num-workers 0` in this container because `/dev/shm` is only 64 MB and PyTorch worker processes can crash with a shared-memory bus error.

```bash
cd /workspace/turbopi_standalone
/workspace/isaaclab/_isaac_sim/python.sh -m train_turbopi_mountain_act \
  --episodes-dir /workspace/turbopi_standalone/data/act_mountain_cliff/session_mountain_act_20260506_121828 \
  --run-dir runs/mountain_act_cvae \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 0 \
  --device cuda
```

## Livestream Inference

Left task, isometric view:

```bash
cd /workspace/turbopi_standalone
/workspace/isaaclab/isaaclab.sh -p scripts/drive_turbopi_mountain_act.py \
  --checkpoint act_best.pt \
  --task go_left \
  --view isometric \
  --duration 40 \
  --control_mode kinematic \
  --livestream 2
```

Right task, isometric view:

```bash
cd /workspace/turbopi_standalone
/workspace/isaaclab/isaaclab.sh -p scripts/drive_turbopi_mountain_act.py \
  --checkpoint act_best.pt \
  --task go_right \
  --view isometric \
  --duration 40 \
  --control_mode kinematic \
  --livestream 2
```

## Multi-View Inference Videos

This writes three MP4s per task: `robot`, `chase`, and `isometric`.

```bash
cd /workspace/turbopi_standalone
/workspace/isaaclab/isaaclab.sh -p scripts/drive_turbopi_mountain_act.py \
  --headless \
  --checkpoint act_best.pt \
  --task go_left \
  --view isometric \
  --duration 25 \
  --control_mode kinematic \
  --video_output_dir /workspace/turbopi_standalone/inference_videos/mountain_act_multiview \
  --video_width 1920 \
  --video_height 1080 \
  --video_fps 30
```

```bash
cd /workspace/turbopi_standalone
/workspace/isaaclab/isaaclab.sh -p scripts/drive_turbopi_mountain_act.py \
  --headless \
  --checkpoint act_best.pt \
  --task go_right \
  --view isometric \
  --duration 25 \
  --control_mode kinematic \
  --video_output_dir /workspace/turbopi_standalone/inference_videos/mountain_act_multiview \
  --video_width 1920 \
  --video_height 1080 \
  --video_fps 30
```
