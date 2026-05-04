"""Standalone TurboPi viewer for the high-cliff road scene."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch TurboPi on a high cliff road with fall-off edges.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument(
    "--view",
    type=str,
    choices=("isometric", "overview", "chase", "robot"),
    default="isometric",
    help="Initial viewport mode.",
)
parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds. 0 runs until closed.")
parser.add_argument("--road_length", type=float, default=2.20, help="Outer length of the elevated track in meters.")
parser.add_argument("--road_width", type=float, default=0.28, help="Width of each elevated track strip in meters.")
parser.add_argument("--rectangle_half_width", type=float, default=0.78, help="Half-width of the rectangular track layout.")
parser.add_argument("--cliff_height", type=float, default=1.35, help="Height of the road above the lower scenery.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils

from cliff_scene import CliffRoadSceneCfg, design_cliff_road_scene, start_pose
from common import (
    PERSPECTIVE_CAMERA_PATH,
    TURBOPI_URDF,
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    resolve_asset_usd,
    reset_robot_pose,
    set_robot_camera_mount,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)

CLIFF_CAMERA_POS = (0.080, 0.0, 0.030)
CLIFF_CAMERA_ROT = (0.996195, 0.0, -0.087156, 0.0)


def set_isometric_camera(sim: sim_utils.SimulationContext, viewport, scene_cfg: CliffRoadSceneCfg) -> str:
    """Set a static 3/4 camera that shows the road height and lower valley."""
    if viewport is not None:
        viewport.set_active_camera(PERSPECTIVE_CAMERA_PATH)
    z = scene_cfg.cliff_height
    sim.set_camera_view(
        eye=[1.85, -2.25, z + 1.10],
        target=[0.0, -0.15, z - 0.10],
    )
    return "isometric"


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = CliffRoadSceneCfg(
        road_length=args_cli.road_length,
        road_width=args_cli.road_width,
        rectangle_half_width=args_cli.rectangle_half_width,
        cliff_height=args_cli.cliff_height,
    )
    design_cliff_road_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    set_robot_camera_mount(CLIFF_CAMERA_POS, CLIFF_CAMERA_ROT)

    sim.reset()
    start_position, start_yaw = start_pose(scene_cfg)
    reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    idle_targets = twist_to_wheel_targets(torch.zeros((robot.num_instances, 3), device=robot.device), robot.device)

    viewport = get_viewport()
    if args_cli.view == "isometric":
        active_view = set_isometric_camera(sim, viewport, scene_cfg)
    else:
        active_view = activate_view_mode(args_cli.view, sim, robot, viewport)

    print(f"[INFO] TurboPi USD  : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"[INFO] TurboPi URDF : {TURBOPI_URDF}")
    print(f"[INFO] Scene        : high cliff road")
    print(f"[INFO] Track        : {scene_cfg.road_length:.2f} m long, strip width {scene_cfg.road_width:.2f} m")
    print(f"[INFO] Cliff height : {scene_cfg.cliff_height:.2f} m")
    print(f"[INFO] Initial view : {active_view}")
    if viewport is None and args_cli.view not in ("overview", "isometric"):
        print("[INFO] No interactive viewport is available in this launch mode, so camera switching is limited.")

    sim_dt = float(sim_cfg.dt)
    elapsed = 0.0

    while simulation_app.is_running():
        if not sim.is_playing():
            sim.play()

        robot.set_joint_velocity_target(idle_targets, joint_ids=wheel_joint_ids)
        hold_arm_posture(robot, arm_joint_ids)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)

        if active_view == "chase":
            update_chase_camera(robot, viewport)

        elapsed += sim_dt
        if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
