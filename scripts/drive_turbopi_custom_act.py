"""Drive the TurboPi on the custom ACT layout using a trained ACT checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import signal
import math
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Run a trained TurboPi ACT policy on the custom layout.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained act_policy checkpoint.")
# UPDATED: Task choices updated to reflect the 4 new topological routes
parser.add_argument("--task", choices=("left_oval", "teardrop", "horseshoe", "small_circle"), default="left_oval")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument(
    "--view",
    type=str,
    choices=("overview", "chase", "robot"),
    default="overview",
    help="Initial viewport mode when a GUI or livestream is available.",
)
parser.add_argument("--duration", type=float, default=0.0, help="Optional simulation duration in seconds.")
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0, help="Physics step in seconds.")
parser.add_argument("--control_hz", type=float, default=10.0, help="Control/update frequency in Hz.")
parser.add_argument("--camera_warmup_steps", type=int, default=12, help="Zero-command steps used to warm up the robot camera.")
parser.add_argument("--settle_steps", type=int, default=12, help="Zero-command steps after reset before policy control begins.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
parser.add_argument("--policy_device", default="auto", help="Torch device for the ACT checkpoint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if os.environ.get("DISPLAY") is None and not args_cli.headless:
    print("[INFO] DISPLAY is not set. Enabling headless rendering for ACT driving.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import quat_from_euler_xyz

from act_policy.runtime import ACTPolicyRuntime
from common import (
    CAMERA_LINK_TO_SENSOR_POS,
    CAMERA_LINK_TO_SENSOR_ROT,
    PERSPECTIVE_CAMERA_PATH,
    activate_view_mode,
    get_viewport,
    get_wheel_joint_ids,
    resolve_asset_usd,
    reset_robot_pose,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)
from custom_turbopi_act_scene import CustomACTSceneCfg, design_custom_act_scene, start_pose


def rgb_frame_from_camera(camera) -> np.ndarray:
    image = camera.data.output["rgb"]
    if image is None or image.numel() == 0:
        raise RuntimeError("Camera sensor has no RGB data yet.")
    rgb = image[0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb, 0.0, 255.0)
            if rgb.max() <= 1.0:
                rgb = rgb * 255.0
        else:
            rgb = np.clip(rgb, 0, 255)
        rgb = rgb.astype(np.uint8)
    return rgb


def build_camera(width: int, height: int) -> Camera:
    return Camera(
        CameraCfg(
            prim_path="/World/TurboPi/camera_link/RobotCamera",
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=None,
        )
    )


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Finishing cleanup.", flush=True)


def main() -> None:
    scene_cfg = CustomACTSceneCfg()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args_cli.physics_dt, render_interval=1, device=args_cli.device)
    )
    design_custom_act_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    reset_robot_pose(robot, position=start_pose(scene_cfg)[0], yaw=start_pose(scene_cfg)[1])

    sim.reset()
    sim.play()

    runtime = ACTPolicyRuntime(args_cli.checkpoint, task=args_cli.task, device=args_cli.policy_device)
    camera = build_camera(runtime.image_width, runtime.image_height)
    wheel_joint_ids = get_wheel_joint_ids(robot)
    viewport = get_viewport()
    active_view = activate_view_mode(args_cli.view, sim, robot, viewport)

    for _ in range(args_cli.settle_steps):
        robot.set_joint_velocity_target(torch.zeros(3, dtype=torch.float32, device=robot.device), joint_ids=wheel_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(args_cli.physics_dt)
    camera.update(0.0)
    for _ in range(args_cli.camera_warmup_steps):
        sim.step()
        robot.update(args_cli.physics_dt)
        camera.update(args_cli.physics_dt)

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)

    print(f"[drive] checkpoint={resolve_asset_usd(args_cli.asset_usd) if args_cli.asset_usd else 'default asset'} task={args_cli.task}")
    print(f"[drive] view={active_view} duration={args_cli.duration}")

    elapsed = 0.0
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = max(1, int(round(control_dt / args_cli.physics_dt)))

    while simulation_app.is_running() and not stop_flag.requested:
        raw_rgb = rgb_frame_from_camera(camera)
        _, command = runtime.predict(raw_rgb)
        action = torch.tensor(command[:3], dtype=torch.float32, device=robot.device).unsqueeze(0)
        robot.set_joint_velocity_target(twist_to_wheel_targets(action, robot.device), joint_ids=wheel_joint_ids)
        robot.write_data_to_sim()

        for _ in range(substeps):
            sim.step()
            robot.update(args_cli.physics_dt)
        camera.update(substeps * args_cli.physics_dt)

        if active_view == "chase":
            update_chase_camera(robot, viewport)

        elapsed += control_dt
        if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()