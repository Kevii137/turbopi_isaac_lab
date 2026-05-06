"""Autonomously record ACT language episodes on the mountain cliff fork scene."""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASKS = ("go_left", "go_right")
TASK_INSTRUCTIONS = {"go_left": "go left", "go_right": "go right"}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "act_mountain_cliff"

parser = argparse.ArgumentParser(description="Record mountain cliff ACT language episodes.")
parser.add_argument("--asset_usd", type=str, default=None)
parser.add_argument("--view", choices=("isometric", "overview", "chase", "robot"), default="chase")
parser.add_argument(
    "--record_camera",
    choices=("view", "robot", "isometric"),
    default="view",
    help="Camera saved into video.mp4. 'view' follows --view when possible.",
)
parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
parser.add_argument("--session_name", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default="turbopi_mountain_act_cvae")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--task", choices=("go_left", "go_right", "mix"), default="mix")
parser.add_argument("--physics_dt", type=float, default=1.0 / 30.0)
parser.add_argument("--control_hz", type=float, default=10.0)
parser.add_argument("--image_width", type=int, default=128)
parser.add_argument("--image_height", type=int, default=128)
parser.add_argument("--target_speed", type=float, default=0.34)
parser.add_argument("--min_forward_speed", type=float, default=0.07)
parser.add_argument("--max_wz", type=float, default=1.35)
parser.add_argument("--position_tolerance", type=float, default=0.075)
parser.add_argument("--switch_distance", type=float, default=0.095)
parser.add_argument("--lookahead_distance", type=float, default=0.18)
parser.add_argument("--approach_distance", type=float, default=0.22)
parser.add_argument("--heading_gain", type=float, default=2.4)
parser.add_argument("--lookahead_heading_gain", type=float, default=1.1)
parser.add_argument("--heading_slowdown_angle", type=float, default=1.25)
parser.add_argument("--turn_in_place_angle", type=float, default=1.15)
parser.add_argument("--off_track_abort_distance", type=float, default=0.32)
parser.add_argument("--stuck_timeout", type=float, default=8.0)
parser.add_argument("--progress_epsilon", type=float, default=0.025)
parser.add_argument("--max_episode_time", type=float, default=40.0)
parser.add_argument("--settle_steps", type=int, default=4)
parser.add_argument("--camera_warmup_steps", type=int, default=8)
parser.add_argument("--min_image_std", type=float, default=5.0)
parser.add_argument("--action_noise_std", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_rollers", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if os.environ.get("DISPLAY") is None and not args_cli.headless:
    print("[INFO] DISPLAY is not set. Enabling headless rendering.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from act_mountain_dataset import ACTEpisodeFrame, ACTEpisodeResult, ACTMountainSessionWriter
from common import (
    PERSPECTIVE_CAMERA_PATH,
    ROBOT_CAMERA_PATH,
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    reset_robot_pose,
    resolve_asset_usd,
    set_robot_camera_mount,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)
from mountain_cliff_scene import (
    RIGHT_BRANCH_CENTERLINE,
    ROAD_CENTERLINE,
    MountainCliffSceneCfg,
    design_mountain_cliff_scene,
    start_pose,
)

CAMERA_POS = (0.080, 0.0, 0.030)
CAMERA_ROT = (0.996195, 0.0, -0.087156, 0.0)
MAX_COMMAND = np.array([0.45, 0.35, 2.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class Segment:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    length: float
    yaw: float


@dataclass(frozen=True)
class Route:
    task_name: str
    task_index: int
    instruction: str
    waypoints: tuple[tuple[float, float], ...]
    segments: tuple[Segment, ...]
    length: float
    start_xy: tuple[float, float]
    start_yaw: float


class StopFlag:
    requested = False

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Finishing cleanup.", flush=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_session_name() -> str:
    return args_cli.session_name or datetime.utcnow().strftime("session_mountain_act_%Y%m%d_%H%M%S")


def resolve_record_camera_mode() -> str:
    if args_cli.record_camera != "view":
        return args_cli.record_camera
    return "isometric" if args_cli.view == "isometric" else "robot"


def build_camera(width: int, height: int, *, mode: str) -> Camera:
    if mode == "robot":
        return Camera(
            CameraCfg(
                prim_path=ROBOT_CAMERA_PATH,
                update_period=0.0,
                height=height,
                width=width,
                data_types=["rgb"],
                spawn=None,
            )
        )
    return Camera(
        CameraCfg(
            prim_path="/World/MountainACTRecordCamera",
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=20.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 100.0),
            ),
        )
    )


def position_record_camera(camera: Camera, *, mode: str, scene_cfg: MountainCliffSceneCfg, device: str) -> None:
    if mode != "isometric":
        return
    camera.set_world_poses_from_view(
        torch.tensor([[3.10, -3.30, scene_cfg.road_z + 1.80]], dtype=torch.float32, device=device),
        torch.tensor([[0.35, 1.15, scene_cfg.road_z - 0.10]], dtype=torch.float32, device=device),
    )


def build_segment(start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> Segment:
    dx = goal_xy[0] - start_xy[0]
    dy = goal_xy[1] - start_xy[1]
    length = math.hypot(dx, dy)
    return Segment(start_xy=start_xy, goal_xy=goal_xy, length=length, yaw=math.atan2(dy, dx))


def build_route(scene_cfg: MountainCliffSceneCfg, task_name: str) -> Route:
    start_position, start_yaw = start_pose(scene_cfg)
    start_xy = (float(start_position[0]), float(start_position[1]))
    if task_name == "go_left":
        waypoints = (start_xy, *ROAD_CENTERLINE[1:])
    else:
        waypoints = (start_xy, *ROAD_CENTERLINE[1:7], *RIGHT_BRANCH_CENTERLINE[1:])
    segments = tuple(build_segment(waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1))
    return Route(
        task_name=task_name,
        task_index=TASKS.index(task_name),
        instruction=TASK_INSTRUCTIONS[task_name],
        waypoints=waypoints,
        segments=segments,
        length=sum(segment.length for segment in segments),
        start_xy=start_xy,
        start_yaw=start_yaw,
    )


def project(point_xy: tuple[float, float], segment: Segment) -> tuple[float, float, float]:
    px, py = point_xy
    sx, sy = segment.start_xy
    gx, gy = segment.goal_xy
    dx = gx - sx
    dy = gy - sy
    denom = max(dx * dx + dy * dy, 1e-9)
    t = clamp(((px - sx) * dx + (py - sy) * dy) / denom, 0.0, 1.0)
    return sx + t * dx, sy + t * dy, t


def point_at_distance(segment: Segment, distance_m: float) -> tuple[float, float]:
    t = clamp(distance_m / max(segment.length, 1e-6), 0.0, 1.0)
    return (
        segment.start_xy[0] + t * (segment.goal_xy[0] - segment.start_xy[0]),
        segment.start_xy[1] + t * (segment.goal_xy[1] - segment.start_xy[1]),
    )


def distance_to_segment(point_xy: tuple[float, float], segment: Segment) -> float:
    nx, ny, _ = project(point_xy, segment)
    return math.hypot(point_xy[0] - nx, point_xy[1] - ny)


def route_progress(point_xy: tuple[float, float], route: Route, segment_index: int) -> float:
    complete = sum(segment.length for segment in route.segments[:segment_index])
    _, _, t = project(point_xy, route.segments[segment_index])
    return clamp(complete + t * route.segments[segment_index].length, 0.0, route.length)


def rgb_frame(camera: Camera) -> np.ndarray:
    image = camera.data.output["rgb"]
    if image is None or image.numel() == 0:
        raise RuntimeError("Camera has no RGB data yet.")
    rgb = image[0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def get_pose(robot) -> tuple[float, float, float]:
    x = float(robot.data.root_pos_w[0, 0].item())
    y = float(robot.data.root_pos_w[0, 1].item())
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    return x, y, wrap_to_pi(float(yaw_t[0].item()))


def integrate(pose: tuple[float, float, float], command: tuple[float, float, float], dt: float) -> tuple[float, float, float]:
    x, y, yaw = pose
    vx, vy, wz = command
    yaw_mid = yaw + 0.5 * wz * dt
    return (
        x + (vx * math.cos(yaw_mid) - vy * math.sin(yaw_mid)) * dt,
        y + (vx * math.sin(yaw_mid) + vy * math.cos(yaw_mid)) * dt,
        wrap_to_pi(yaw + wz * dt),
    )


def write_kinematic(robot, wheel_joint_ids, arm_joint_ids, pose, command, root_z: float) -> None:
    x, y, yaw = pose
    vx, vy, wz = command
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, 0] = x
    root_pose[:, 1] = y
    root_pose[:, 2] = root_z
    yaw_t = torch.full((robot.num_instances,), yaw, dtype=torch.float32, device=robot.device)
    zeros = torch.zeros_like(yaw_t)
    root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_t)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((robot.num_instances, 6), dtype=torch.float32, device=robot.device))
    command_t = torch.tensor([[vx, vy, wz]], dtype=torch.float32, device=robot.device)
    robot.set_joint_velocity_target(twist_to_wheel_targets(command_t, robot.device), joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def step_n(sim, robot, camera, wheel_joint_ids, arm_joint_ids, pose, command, substeps, physics_dt, viewport, view, root_z):
    current = pose
    for _ in range(substeps):
        if not simulation_app.is_running():
            return False, current
        if not sim.is_playing():
            sim.play()
        current = integrate(current, command, physics_dt)
        write_kinematic(robot, wheel_joint_ids, arm_joint_ids, current, command, root_z)
        sim.step()
        robot.update(physics_dt)
        if view == "chase":
            update_chase_camera(robot, viewport)
    camera.update(dt=substeps * physics_dt)
    return True, current


def compute_command(pose: tuple[float, float, float], segment: Segment) -> tuple[tuple[float, float, float], float, float]:
    x, y, yaw = pose
    dist = math.hypot(segment.goal_xy[0] - x, segment.goal_xy[1] - y)
    if dist <= args_cli.position_tolerance:
        return (0.0, 0.0, 0.0), 0.0, dist
    _, _, t = project((x, y), segment)
    target = point_at_distance(segment, min(segment.length, t * segment.length + args_cli.lookahead_distance))
    dx = target[0] - x
    dy = target[1] - y
    target_bx = math.cos(yaw) * dx + math.sin(yaw) * dy
    target_by = -math.sin(yaw) * dx + math.cos(yaw) * dy
    point_error = math.atan2(target_by, max(target_bx, 0.04))
    yaw_error = wrap_to_pi(segment.yaw - yaw)
    approach = clamp(dist / max(args_cli.approach_distance, 1e-6), 0.35, 1.0)
    heading = clamp(1.0 - abs(yaw_error) / max(args_cli.heading_slowdown_angle, 1e-6), 0.10, 1.0)
    vx = clamp(args_cli.target_speed * min(approach, heading), args_cli.min_forward_speed, args_cli.target_speed)
    if abs(yaw_error) >= args_cli.turn_in_place_angle:
        vx = 0.0
    wz = clamp(args_cli.heading_gain * yaw_error + args_cli.lookahead_heading_gain * point_error, -args_cli.max_wz, args_cli.max_wz)
    return (vx, 0.0, wz), yaw_error, dist


def run_episode(sim, robot, camera, wheel_joint_ids, arm_joint_ids, viewport, view, scene_cfg, route, stop_flag, rng):
    root_z = scene_cfg.road_z + scene_cfg.start_height
    pose = (route.start_xy[0], route.start_xy[1], route.start_yaw)
    reset_robot_pose(robot, position=(pose[0], pose[1], root_z), yaw=pose[2])
    write_kinematic(robot, wheel_joint_ids, arm_joint_ids, pose, (0.0, 0.0, 0.0), root_z)
    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = max(1, int(round(control_dt / physics_dt)))
    for _ in range(max(1, args_cli.settle_steps + args_cli.camera_warmup_steps)):
        ok, pose = step_n(sim, robot, camera, wheel_joint_ids, arm_joint_ids, pose, (0.0, 0.0, 0.0), 1, physics_dt, viewport, view, root_z)
        if not ok:
            raise RuntimeError("Simulation closed during warmup.")

    frames: list[ACTEpisodeFrame] = []
    errors: list[float] = []
    previous_action = np.zeros(4, dtype=np.float32)
    segment_index = 0
    best_progress = 0.0
    last_progress_time = 0.0
    terminal_reason = "timeout"
    success = False
    max_steps = max(1, int(math.ceil(args_cli.max_episode_time / control_dt)))
    for step_index in range(max_steps):
        if stop_flag.requested:
            terminal_reason = "interrupted"
            break
        pose = get_pose(robot)
        segment = route.segments[segment_index]
        progress_m = route_progress((pose[0], pose[1]), route, segment_index)
        progress_ratio = progress_m / max(route.length, 1e-6)
        error = distance_to_segment((pose[0], pose[1]), segment)
        if error > args_cli.off_track_abort_distance:
            terminal_reason = "off_track"
            break
        command, _yaw_error, dist = compute_command(pose, segment)
        if dist <= args_cli.switch_distance:
            if segment_index >= len(route.segments) - 1:
                terminal_reason = "goal_reached"
                success = True
                break
            segment_index += 1
            continue
        if progress_m >= best_progress + args_cli.progress_epsilon:
            best_progress = progress_m
            last_progress_time = step_index * control_dt
        if step_index * control_dt - last_progress_time >= args_cli.stuck_timeout:
            terminal_reason = "stuck"
            break
        if args_cli.action_noise_std > 0.0:
            noise = rng.normal(0.0, args_cli.action_noise_std, size=3).astype(np.float32) * MAX_COMMAND[:3]
            command = (
                clamp(command[0] + float(noise[0]), 0.0, args_cli.target_speed),
                clamp(command[1] + float(noise[1]), -0.18, 0.18),
                clamp(command[2] + float(noise[2]), -args_cli.max_wz, args_cli.max_wz),
            )
        stop_value = 1.0 if progress_ratio >= 0.97 else 0.0
        command4 = np.asarray([command[0], command[1], command[2], stop_value], dtype=np.float32)
        action = np.clip(command4 / MAX_COMMAND, -1.0, 1.0)
        image = rgb_frame(camera)
        frames.append(
            ACTEpisodeFrame(
                image_rgb=image,
                timestamp=float(step_index * control_dt),
                state=previous_action.copy(),
                action=action.copy(),
                command=command4.copy(),
                track_error=float(error),
                route_progress=float(progress_ratio),
            )
        )
        previous_action = action
        errors.append(error)
        ok, pose = step_n(sim, robot, camera, wheel_joint_ids, arm_joint_ids, pose, command, substeps, physics_dt, viewport, view, root_z)
        if not ok:
            terminal_reason = "app_closed"
            break
    duration = len(frames) * control_dt
    return ACTEpisodeResult(
        task_name=route.task_name,
        task_index=route.task_index,
        instruction=route.instruction,
        frames=frames,
        success=success,
        terminal_reason=terminal_reason,
        final_route_progress=1.0 if success else best_progress / max(route.length, 1e-6),
        mean_track_error=float(np.mean(errors)) if errors else float("inf"),
        duration_s=duration,
    )


def main() -> None:
    scene_cfg = MountainCliffSceneCfg()
    routes = {task: build_route(scene_cfg, task) for task in TASKS}
    physics_dt = max(float(args_cli.physics_dt), 1.0 / 60.0)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = max(1, int(round(control_dt / physics_dt)))
    render_interval = substeps if args_cli.headless and not bool(getattr(args_cli, "livestream", 0)) else 1
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, render_interval=render_interval, device=args_cli.device))
    design_mountain_cliff_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    set_robot_camera_mount(CAMERA_POS, CAMERA_ROT)
    record_camera_mode = resolve_record_camera_mode()
    camera = build_camera(args_cli.image_width, args_cli.image_height, mode=record_camera_mode)
    sim.reset()
    position_record_camera(camera, mode=record_camera_mode, scene_cfg=scene_cfg, device=robot.device)
    camera.update(dt=0.0)
    sim.play()
    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    viewport = get_viewport()
    view = activate_view_mode(args_cli.view, sim, robot, viewport)
    if args_cli.view == "isometric" and viewport is not None:
        viewport.set_active_camera(PERSPECTIVE_CAMERA_PATH)
    writer = ACTMountainSessionWriter(
        output_root=args_cli.output_dir,
        session_name=build_session_name(),
        dataset_name=args_cli.dataset_name,
        fps=args_cli.control_hz,
        image_width=args_cli.image_width,
        image_height=args_cli.image_height,
        control_hz=args_cli.control_hz,
        physics_dt=physics_dt,
        tasks=TASKS,
        task_instructions=TASK_INSTRUCTIONS,
        record_camera=record_camera_mode,
    )
    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)
    rng = np.random.default_rng(args_cli.seed)
    print(f"[record] Output session: {writer.session_dir}")
    print(f"[record] Saved video camera: {record_camera_mode}")
    saved = 0
    attempts = 0
    try:
        while saved < args_cli.num_episodes and simulation_app.is_running() and not stop_flag.requested:
            attempts += 1
            task_name = TASKS[saved % len(TASKS)] if args_cli.task == "mix" else args_cli.task
            result = run_episode(sim, robot, camera, wheel_joint_ids, arm_joint_ids, viewport, view, scene_cfg, routes[task_name], stop_flag, rng)
            if result.success and result.frames:
                episode_dir = writer.save_episode(saved, result)
                print(f"[record] saved episode_{saved:05d} {task_name} frames={len(result.frames)} -> {episode_dir}", flush=True)
                saved += 1
            else:
                writer.record_failure()
                print(f"[record] failed attempt={attempts} task={task_name} reason={result.terminal_reason} progress={result.final_route_progress:.2f}", flush=True)
    finally:
        print(f"[record] complete: saved={saved} session={writer.session_dir}", flush=True)


def close_app_and_exit(code: int = 0) -> None:
    timer = threading.Timer(5.0, lambda: os._exit(code))
    timer.daemon = True
    timer.start()
    try:
        simulation_app.close()
    finally:
        timer.cancel()
        os._exit(code)


if __name__ == "__main__":
    main()
    close_app_and_exit(0)
