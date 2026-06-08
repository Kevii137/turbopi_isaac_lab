"""View the custom TurboPi ACT layout in Isaac Lab."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="View the custom TurboPi ACT layout.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument(
    "--view",
    type=str,
    choices=("overview", "chase", "robot"),
    default="overview",
    help="Initial viewport mode when a GUI or livestream is available.",
)
parser.add_argument("--duration", type=float, default=0.0, help="Optional simulation duration in seconds.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if os.environ.get("DISPLAY") is None and not args_cli.headless:
    print("[INFO] DISPLAY is not set. Enabling headless rendering for view mode.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from common import (
    activate_view_mode,
    get_viewport,
    reset_robot_pose,
    resolve_asset_usd,
    set_robot_camera_mount,
    spawn_turbopi,
    update_chase_camera,
)
from custom_turbopi_act_scene import CustomACTSceneCfg, design_custom_act_scene, start_pose

VIEW_CAMERA_POS = (0.140, 0.0, 0.115)
VIEW_CAMERA_ROT = (0.987688, 0.0, -0.156434, 0.0)


def main() -> None:
    scene_cfg = CustomACTSceneCfg()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=args_cli.device))

    design_custom_act_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    set_robot_camera_mount(VIEW_CAMERA_POS, VIEW_CAMERA_ROT)
    sim.reset()
    reset_robot_pose(robot, position=start_pose(scene_cfg)[0], yaw=start_pose(scene_cfg)[1])
    sim.play()

    viewport = get_viewport()
    view_mode = activate_view_mode(args_cli.view, sim, robot, viewport)
    print(f"[view] asset={resolve_asset_usd(args_cli.asset_usd) if args_cli.asset_usd else 'default asset'} mode={view_mode}")

    elapsed = 0.0
    dt = sim.get_physics_dt()
    while simulation_app.is_running():
        if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
            break
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.play()
            continue

        sim.step()
        robot.update(dt)
        if view_mode == "chase":
            update_chase_camera(robot, viewport)
        elapsed += dt

    simulation_app.close()


if __name__ == "__main__":
    main()
