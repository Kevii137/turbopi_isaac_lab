"""Record task-conditioned TurboPi episodes in the cliff split-rectangle scene.

Each accepted episode starts on the center strip, drives straight toward the
far end, takes the requested branch, and returns to the start:

- ``go_left``: center straight -> top-left -> left side -> bottom center.
- ``go_right``: center straight -> top-right -> right side -> bottom center.
"""

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

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "cnn_cliff_intent"
TASKS = ("go_left", "go_right")
TASK_INSTRUCTIONS = {
    "go_left": "go left",
    "go_right": "go right",
}

parser = argparse.ArgumentParser(description="Record TurboPi cliff intent episodes.")
parser.add_argument("--asset_usd", type=str, default=None)
parser.add_argument("--view", type=str, choices=("isometric", "overview", "chase", "robot"), default="chase")
parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
parser.add_argument("--session_name", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default="turbopi_cliff_intent_cnn")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--task", type=str, choices=("go_left", "go_right", "mix"), default="mix")
parser.add_argument("--physics_dt", type=float, default=1.0 / 30.0)
parser.add_argument("--control_hz", type=float, default=10.0)
parser.add_argument("--image_width", type=int, default=160)
parser.add_argument("--image_height", type=int, default=120)
parser.add_argument("--road_length", type=float, default=2.20)
parser.add_argument("--road_width", type=float, default=0.28)
parser.add_argument("--rectangle_half_width", type=float, default=0.78)
parser.add_argument("--cliff_height", type=float, default=1.35)
parser.add_argument("--target_speed", type=float, default=0.26)
parser.add_argument("--min_forward_speed", type=float, default=0.06)
parser.add_argument("--position_tolerance", type=float, default=0.055)
parser.add_argument("--switch_distance", type=float, default=0.070)
parser.add_argument("--lookahead_distance", type=float, default=0.13)
parser.add_argument("--approach_distance", type=float, default=0.18)
parser.add_argument("--heading_gain", type=float, default=2.5)
parser.add_argument("--lookahead_heading_gain", type=float, default=1.0)
parser.add_argument("--max_wz", type=float, default=1.15)
parser.add_argument("--turn_in_place_angle", type=float, default=1.05)
parser.add_argument("--heading_slowdown_angle", type=float, default=1.20)
parser.add_argument("--off_track_abort_distance", type=float, default=0.18)
parser.add_argument("--stuck_timeout", type=float, default=8.0)
parser.add_argument("--progress_epsilon", type=float, default=0.02)
parser.add_argument("--settle_steps", type=int, default=4)
parser.add_argument("--cooldown_steps", type=int, default=2)
parser.add_argument("--camera_warmup_steps", type=int, default=6)
parser.add_argument("--min_image_std", type=float, default=8.0)
parser.add_argument("--max_episode_time", type=float, default=35.0)
parser.add_argument("--randomize_start", action="store_true")
parser.add_argument("--start_lateral_jitter", type=float, default=0.025)
parser.add_argument("--start_yaw_jitter_deg", type=float, default=7.0)
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

from cliff_intent_dataset import CliffIntentSessionWriter, IntentEpisodeFrame, IntentEpisodeResult
from cliff_scene import CliffRoadSceneCfg, design_cliff_road_scene, start_pose
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

CLIFF_CAMERA_POS = (0.080, 0.0, 0.030)
CLIFF_CAMERA_ROT = (0.996195, 0.0, -0.087156, 0.0)


@dataclass(frozen=True)
class RouteSegment:
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    length: float
    yaw: float


@dataclass(frozen=True)
class CliffIntentRoute:
    task_name: str
    task_index: int
    instruction: str
    waypoints: tuple[tuple[float, float], ...]
    segments: tuple[RouteSegment, ...]
    length: float
    start_xy: tuple[float, float]
    start_yaw: float


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Finishing cleanup.", flush=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_session_name() -> str:
    return args_cli.session_name or datetime.utcnow().strftime("session_cliff_intent_%Y%m%d_%H%M%S")


def build_robot_camera_sensor(*, width: int, height: int) -> Camera:
    camera_cfg = CameraCfg(
        prim_path=ROBOT_CAMERA_PATH,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=None,
    )
    return Camera(camera_cfg)


def build_segment(start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> RouteSegment:
    dx = goal_xy[0] - start_xy[0]
    dy = goal_xy[1] - start_xy[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        raise ValueError("Route waypoints must be distinct.")
    return RouteSegment(start_xy=start_xy, goal_xy=goal_xy, length=length, yaw=math.atan2(dy, dx))


def build_route(scene_cfg: CliffRoadSceneCfg, task_name: str) -> CliffIntentRoute:
    if task_name not in TASKS:
        raise ValueError(f"Unsupported task: {task_name}")

    start_position, start_yaw = start_pose(scene_cfg)
    start_xy = (float(start_position[0]), float(start_position[1]))
    half_y = 0.5 * scene_cfg.road_length
    half_x = scene_cfg.rectangle_half_width
    top_center = (0.0, half_y)

    if task_name == "go_left":
        waypoints = (
            start_xy,
            top_center,
            (-half_x, half_y),
            (-half_x, -half_y),
            start_xy,
        )
    else:
        waypoints = (
            start_xy,
            top_center,
            (half_x, half_y),
            (half_x, -half_y),
            start_xy,
        )

    segments = tuple(build_segment(waypoints[i], waypoints[i + 1]) for i in range(len(waypoints) - 1))
    return CliffIntentRoute(
        task_name=task_name,
        task_index=TASKS.index(task_name),
        instruction=TASK_INSTRUCTIONS[task_name],
        waypoints=waypoints,
        segments=segments,
        length=sum(segment.length for segment in segments),
        start_xy=start_xy,
        start_yaw=start_yaw,
    )


def project_point_to_segment(
    point_xy: tuple[float, float],
    segment: RouteSegment,
) -> tuple[float, float, float]:
    px, py = point_xy
    sx, sy = segment.start_xy
    gx, gy = segment.goal_xy
    dx = gx - sx
    dy = gy - sy
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return sx, sy, 0.0
    t = ((px - sx) * dx + (py - sy) * dy) / length_sq
    t = clamp(t, 0.0, 1.0)
    return sx + t * dx, sy + t * dy, t


def point_at_distance(segment: RouteSegment, distance_m: float) -> tuple[float, float]:
    t = clamp(distance_m / max(segment.length, 1e-6), 0.0, 1.0)
    sx, sy = segment.start_xy
    gx, gy = segment.goal_xy
    return sx + t * (gx - sx), sy + t * (gy - sy)


def distance_to_segment(point_xy: tuple[float, float], segment: RouteSegment) -> float:
    nx, ny, _ = project_point_to_segment(point_xy, segment)
    return math.hypot(point_xy[0] - nx, point_xy[1] - ny)


def route_progress_m(point_xy: tuple[float, float], route: CliffIntentRoute, segment_index: int) -> float:
    completed = sum(segment.length for segment in route.segments[:segment_index])
    _, _, t = project_point_to_segment(point_xy, route.segments[segment_index])
    return clamp(completed + t * route.segments[segment_index].length, 0.0, route.length)


def ensure_sim_playing(sim: sim_utils.SimulationContext) -> None:
    if not sim.is_playing():
        sim.play()


def rgb_frame(camera: Camera) -> np.ndarray:
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


def steps_per_control(physics_dt: float, control_hz: float) -> int:
    control_dt = 1.0 / max(control_hz, 1e-6)
    return max(1, int(round(control_dt / physics_dt)))


def get_pose(robot) -> tuple[float, float, float]:
    x = float(robot.data.root_pos_w[0, 0].item())
    y = float(robot.data.root_pos_w[0, 1].item())
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    return x, y, wrap_to_pi(float(yaw_t[0].item()))


def write_kinematic_state(
    robot,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    pose: tuple[float, float, float],
    command_vec: tuple[float, float, float],
    root_z: float,
) -> None:
    x, y, yaw = pose
    vx, vy, wz = command_vec
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, 0] = float(x)
    root_pose[:, 1] = float(y)
    root_pose[:, 2] = float(root_z)

    yaw_t = torch.full((robot.num_instances,), float(yaw), dtype=torch.float32, device=robot.device)
    zeros = torch.zeros_like(yaw_t)
    root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_t)

    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((robot.num_instances, 6), dtype=torch.float32, device=robot.device))

    command_t = torch.tensor([[vx, vy, wz]], dtype=torch.float32, device=robot.device)
    wheel_targets = twist_to_wheel_targets(command_t, robot.device)
    robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def integrate_body_pose(
    pose: tuple[float, float, float],
    command_vec: tuple[float, float, float],
    dt: float,
) -> tuple[float, float, float]:
    x, y, yaw = pose
    vx, vy, wz = command_vec
    yaw_mid = yaw + 0.5 * wz * dt
    x += (vx * math.cos(yaw_mid) - vy * math.sin(yaw_mid)) * dt
    y += (vx * math.sin(yaw_mid) + vy * math.cos(yaw_mid)) * dt
    yaw = wrap_to_pi(yaw + wz * dt)
    return x, y, yaw


def step_kinematic_n(
    sim,
    robot,
    camera,
    wheel_joint_ids,
    arm_joint_ids,
    pose,
    command_vec,
    substeps,
    physics_dt,
    viewport,
    active_view,
    root_z,
):
    current_pose = pose
    for _ in range(substeps):
        if not simulation_app.is_running():
            return False, current_pose
        ensure_sim_playing(sim)
        current_pose = integrate_body_pose(current_pose, command_vec, physics_dt)
        write_kinematic_state(robot, wheel_joint_ids, arm_joint_ids, current_pose, command_vec, root_z)
        sim.step()
        robot.update(physics_dt)
        if active_view == "chase":
            update_chase_camera(robot, viewport)
    camera.update(dt=substeps * physics_dt)
    return True, current_pose


def compute_command(
    pose: tuple[float, float, float],
    segment: RouteSegment,
) -> tuple[tuple[float, float, float], float, float]:
    x, y, yaw = pose
    dist_to_goal = math.hypot(segment.goal_xy[0] - x, segment.goal_xy[1] - y)
    if dist_to_goal <= args_cli.position_tolerance:
        return (0.0, 0.0, 0.0), 0.0, dist_to_goal

    _nearest_x, _nearest_y, t = project_point_to_segment((x, y), segment)
    current_progress = t * segment.length
    target_xy = point_at_distance(segment, min(segment.length, current_progress + args_cli.lookahead_distance))

    dx = target_xy[0] - x
    dy = target_xy[1] - y
    target_bx = math.cos(yaw) * dx + math.sin(yaw) * dy
    target_by = -math.sin(yaw) * dx + math.cos(yaw) * dy
    point_heading_error = math.atan2(target_by, max(target_bx, 0.04))
    yaw_error = wrap_to_pi(segment.yaw - yaw)

    approach_scale = clamp(dist_to_goal / max(args_cli.approach_distance, 1e-6), 0.35, 1.0)
    heading_scale = clamp(1.0 - abs(yaw_error) / max(args_cli.heading_slowdown_angle, 1e-6), 0.10, 1.0)
    vx = clamp(
        args_cli.target_speed * min(approach_scale, heading_scale),
        args_cli.min_forward_speed,
        args_cli.target_speed,
    )
    if abs(yaw_error) >= args_cli.turn_in_place_angle:
        vx = 0.0

    wz = clamp(
        args_cli.heading_gain * yaw_error + args_cli.lookahead_heading_gain * point_heading_error,
        -args_cli.max_wz,
        args_cli.max_wz,
    )
    return (vx, 0.0, wz), yaw_error, dist_to_goal


def set_isometric_camera(sim: sim_utils.SimulationContext, viewport, scene_cfg: CliffRoadSceneCfg) -> str:
    if viewport is not None:
        viewport.set_active_camera(PERSPECTIVE_CAMERA_PATH)
    z = scene_cfg.cliff_height
    sim.set_camera_view(eye=[1.85, -2.25, z + 1.10], target=[0.0, -0.15, z - 0.10])
    return "isometric"


def warm_camera(sim, robot, camera, wheel_joint_ids, arm_joint_ids, pose, viewport, active_view, root_z):
    physics_dt = float(args_cli.physics_dt)
    last_std = 0.0
    current_pose = pose
    for _ in range(max(1, args_cli.camera_warmup_steps)):
        ok, current_pose = step_kinematic_n(
            sim,
            robot,
            camera,
            wheel_joint_ids,
            arm_joint_ids,
            current_pose,
            (0.0, 0.0, 0.0),
            1,
            physics_dt,
            viewport,
            active_view,
            root_z,
        )
        if not ok:
            return False, current_pose, last_std
        try:
            last_std = float(np.asarray(rgb_frame(camera), dtype=np.float32).std())
        except Exception:
            last_std = 0.0
        if last_std >= args_cli.min_image_std:
            return True, current_pose, last_std
    return False, current_pose, last_std


def run_episode(
    *,
    sim,
    robot,
    camera,
    wheel_joint_ids,
    arm_joint_ids,
    viewport,
    active_view,
    scene_cfg: CliffRoadSceneCfg,
    route: CliffIntentRoute,
    stop_flag: StopFlag,
    rng: np.random.Generator,
):
    root_z = scene_cfg.cliff_height + scene_cfg.start_height
    start_x, start_y = route.start_xy
    start_yaw = route.start_yaw
    if args_cli.randomize_start:
        start_x += float(rng.uniform(-args_cli.start_lateral_jitter, args_cli.start_lateral_jitter))
        start_yaw = wrap_to_pi(
            start_yaw + math.radians(float(rng.uniform(-args_cli.start_yaw_jitter_deg, args_cli.start_yaw_jitter_deg)))
        )

    pose = (start_x, start_y, start_yaw)
    reset_robot_pose(robot, position=(start_x, start_y, root_z), yaw=start_yaw)
    write_kinematic_state(robot, wheel_joint_ids, arm_joint_ids, pose, (0.0, 0.0, 0.0), root_z)

    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = steps_per_control(physics_dt, args_cli.control_hz)
    max_steps = max(1, int(math.ceil(args_cli.max_episode_time / control_dt)))

    ok, pose = step_kinematic_n(
        sim,
        robot,
        camera,
        wheel_joint_ids,
        arm_joint_ids,
        pose,
        (0.0, 0.0, 0.0),
        max(1, args_cli.settle_steps),
        physics_dt,
        viewport,
        active_view,
        root_z,
    )
    if not ok:
        raise RuntimeError("Simulation app closed during reset settle.")

    camera_ready, pose, warm_std = warm_camera(
        sim, robot, camera, wheel_joint_ids, arm_joint_ids, pose, viewport, active_view, root_z
    )
    if not camera_ready:
        print(f"[WARN] Camera warmup std stayed at {warm_std:.2f}; continuing anyway.", flush=True)

    frames: list[IntentEpisodeFrame] = []
    track_errors: list[float] = []
    body_speeds: list[float] = []
    image_stds: list[float] = []
    actions: list[np.ndarray] = []
    prev_action = np.zeros(3, dtype=np.float32)
    max_command = np.array([0.45, 0.35, 2.0], dtype=np.float32)
    segment_index = 0
    best_progress_m = 0.0
    last_progress_time = 0.0
    terminal_reason = "timeout"
    success = False
    last_print_at = -1.0

    print(
        f"[INFO] Starting {route.task_name} episode: "
        f"{' -> '.join(f'({x:+.2f},{y:+.2f})' for x, y in route.waypoints)}",
        flush=True,
    )

    for step_index in range(max_steps):
        if stop_flag.requested:
            terminal_reason = "interrupted"
            break
        if not simulation_app.is_running():
            terminal_reason = "app_closed"
            break

        pose = get_pose(robot)
        segment = route.segments[segment_index]
        progress_m = route_progress_m((pose[0], pose[1]), route, segment_index)
        progress_ratio = progress_m / max(route.length, 1e-6)
        track_error = distance_to_segment((pose[0], pose[1]), segment)
        if track_error > args_cli.off_track_abort_distance:
            terminal_reason = "off_track"
            break

        command_vec, yaw_error, dist_to_goal = compute_command(pose, segment)
        if dist_to_goal <= args_cli.switch_distance:
            best_progress_m = max(best_progress_m, sum(item.length for item in route.segments[: segment_index + 1]))
            last_progress_time = step_index * control_dt
            if segment_index >= len(route.segments) - 1:
                terminal_reason = "goal_reached"
                success = True
                break
            segment_index += 1
            continue

        if progress_m >= best_progress_m + args_cli.progress_epsilon:
            best_progress_m = progress_m
            last_progress_time = step_index * control_dt
        if (
            step_index * control_dt >= args_cli.stuck_timeout
            and step_index * control_dt - last_progress_time >= args_cli.stuck_timeout
        ):
            terminal_reason = "stuck"
            break

        if args_cli.action_noise_std > 0.0:
            noise = rng.normal(0.0, args_cli.action_noise_std, size=3).astype(np.float32) * max_command
            command_vec = (
                clamp(float(command_vec[0] + noise[0]), 0.0, args_cli.target_speed),
                clamp(float(command_vec[1] + noise[1]), -0.18, 0.18),
                clamp(float(command_vec[2] + noise[2]), -args_cli.max_wz, args_cli.max_wz),
            )

        image = rgb_frame(camera)
        image_stds.append(float(np.asarray(image, dtype=np.float32).std()))
        track_errors.append(track_error)
        body_speeds.append(abs(float(command_vec[0])))

        command_np = np.asarray(command_vec, dtype=np.float32)
        action_np = np.clip(command_np / max_command, -1.0, 1.0)
        actions.append(action_np)
        frames.append(
            IntentEpisodeFrame(
                image_rgb=image,
                timestamp=float(step_index * control_dt),
                state=prev_action.copy(),
                action=action_np.copy(),
                command=command_np.copy(),
                body_velocity=command_np.copy(),
                track_error=float(track_error),
                route_progress=float(progress_ratio),
            )
        )
        prev_action = action_np

        ok, pose = step_kinematic_n(
            sim,
            robot,
            camera,
            wheel_joint_ids,
            arm_joint_ids,
            pose,
            command_vec,
            substeps,
            physics_dt,
            viewport,
            active_view,
            root_z,
        )
        if not ok:
            terminal_reason = "app_closed"
            break

        episode_time = step_index * control_dt
        if episode_time - last_print_at >= 1.0:
            last_print_at = episode_time
            print(
                f"[INFO] {route.task_name:8s} t={episode_time:5.1f}s "
                f"seg={segment_index + 1}/{len(route.segments)} progress={progress_ratio:.3f} "
                f"dist={dist_to_goal:.3f} track={track_error:.3f} yaw_err={yaw_error:+.2f} "
                f"pose=({pose[0]:+.2f},{pose[1]:+.2f},{pose[2]:+.2f}) "
                f"cmd=[{command_vec[0]:+.2f},{command_vec[1]:+.2f},{command_vec[2]:+.2f}]",
                flush=True,
            )

    step_kinematic_n(
        sim,
        robot,
        camera,
        wheel_joint_ids,
        arm_joint_ids,
        pose,
        (0.0, 0.0, 0.0),
        max(1, args_cli.cooldown_steps),
        physics_dt,
        viewport,
        active_view,
        root_z,
    )

    duration_s = len(frames) * control_dt
    mean_track_error = float(np.mean(track_errors)) if track_errors else float("inf")
    p90_track_error = float(np.quantile(track_errors, 0.9)) if track_errors else float("inf")
    max_track_error = float(np.max(track_errors)) if track_errors else float("inf")
    over_010 = float(np.mean(np.asarray(track_errors) > 0.10)) if track_errors else 1.0
    over_015 = float(np.mean(np.asarray(track_errors) > 0.15)) if track_errors else 1.0
    mean_image_std = float(np.mean(image_stds)) if image_stds else 0.0
    min_image_std = float(np.min(image_stds)) if image_stds else 0.0
    mean_speed = float(np.mean(body_speeds)) if body_speeds else 0.0
    action_arr = np.asarray(actions, dtype=np.float32) if actions else np.zeros((0, 3), dtype=np.float32)
    mean_abs = np.mean(np.abs(action_arr), axis=0) if len(action_arr) > 0 else np.zeros(3, dtype=np.float32)
    final_progress = 1.0 if success else (float(best_progress_m / max(route.length, 1e-6)) if frames else 0.0)

    return IntentEpisodeResult(
        task_name=route.task_name,
        task_index=route.task_index,
        instruction=route.instruction,
        frames=frames,
        success=success,
        terminal_reason=terminal_reason,
        final_route_progress=final_progress,
        mean_track_error=mean_track_error,
        p90_track_error=p90_track_error,
        max_track_error=max_track_error,
        frames_over_010_ratio=over_010,
        frames_over_015_ratio=over_015,
        mean_image_std=mean_image_std,
        min_image_std=min_image_std,
        mean_abs_action_vx=float(mean_abs[0]),
        mean_abs_action_vy=float(mean_abs[1]),
        mean_abs_action_wz=float(mean_abs[2]),
        mean_action_vy_vx_ratio=float(mean_abs[1] / max(float(mean_abs[0]), 1e-6)),
        mean_speed=mean_speed,
        duration_s=duration_s,
    )


def main() -> None:
    scene_cfg = CliffRoadSceneCfg(
        road_length=args_cli.road_length,
        road_width=args_cli.road_width,
        rectangle_half_width=args_cli.rectangle_half_width,
        cliff_height=args_cli.cliff_height,
    )
    routes = {task: build_route(scene_cfg, task) for task in TASKS}

    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    physics_dt = min(control_dt, max(physics_dt, 1.0 / 30.0))
    substeps = steps_per_control(physics_dt, args_cli.control_hz)
    livestream_enabled = bool(getattr(args_cli, "livestream", 0))
    render_interval = substeps if args_cli.headless and not livestream_enabled else 1

    sim_cfg = sim_utils.SimulationCfg(dt=physics_dt, render_interval=render_interval, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    design_cliff_road_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    set_robot_camera_mount(CLIFF_CAMERA_POS, CLIFF_CAMERA_ROT)
    camera = build_robot_camera_sensor(width=args_cli.image_width, height=args_cli.image_height)

    sim.reset()
    camera.update(dt=0.0)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    viewport = get_viewport()
    if args_cli.view == "isometric":
        active_view = set_isometric_camera(sim, viewport, scene_cfg)
    else:
        active_view = activate_view_mode(args_cli.view, sim, robot, viewport)

    writer = CliffIntentSessionWriter(
        output_root=Path(args_cli.output_dir),
        session_name=build_session_name(),
        dataset_name=args_cli.dataset_name,
        fps=args_cli.control_hz,
        image_width=args_cli.image_width,
        image_height=args_cli.image_height,
        episode_time_s=args_cli.max_episode_time,
        control_hz=args_cli.control_hz,
        physics_dt=physics_dt,
        tasks=TASKS,
        task_instructions=TASK_INSTRUCTIONS,
    )

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)
    rng = np.random.default_rng(int(args_cli.seed))

    print()
    print("=" * 60)
    print("  TurboPi Cliff Intent Recorder")
    print("=" * 60)
    print(f"  TurboPi USD    : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"  Output session : {writer.session_dir}")
    print(f"  Task mode      : {args_cli.task}")
    print(f"  Episodes       : {args_cli.num_episodes}")
    print(f"  Control rate   : {args_cli.control_hz:.1f} Hz")
    print(f"  Sim dt         : {physics_dt:.4f} s ({substeps} substeps/control)")
    print(f"  Image          : {args_cli.image_width}x{args_cli.image_height}")
    print(f"  Cliff track    : length={scene_cfg.road_length:.2f}, width={scene_cfg.road_width:.2f}, half_x={scene_cfg.rectangle_half_width:.2f}")
    print(f"  Start view     : {active_view}")
    print()

    saved = 0
    attempts = 0
    try:
        while saved < args_cli.num_episodes and simulation_app.is_running() and not stop_flag.requested:
            attempts += 1
            if args_cli.task == "mix":
                task_name = TASKS[saved % len(TASKS)]
            else:
                task_name = args_cli.task
            result = run_episode(
                sim=sim,
                robot=robot,
                camera=camera,
                wheel_joint_ids=wheel_joint_ids,
                arm_joint_ids=arm_joint_ids,
                viewport=viewport,
                active_view=active_view,
                scene_cfg=scene_cfg,
                route=routes[task_name],
                stop_flag=stop_flag,
                rng=rng,
            )
            if result.success and result.frames:
                episode_dir = writer.save_episode(saved, result)
                print(
                    f"[INFO] Saved episode_{saved:05d} [{result.task_name}] "
                    f"frames={len(result.frames)} progress={result.final_route_progress:.2f} "
                    f"mean_err={result.mean_track_error:.3f} -> {episode_dir}",
                    flush=True,
                )
                saved += 1
            else:
                writer.record_failure()
                print(
                    f"[WARN] Attempt {attempts} failed [{result.task_name}] "
                    f"reason={result.terminal_reason} frames={len(result.frames)} "
                    f"progress={result.final_route_progress:.2f}",
                    flush=True,
                )
    finally:
        print()
        print(f"[INFO] Session complete : {writer.session_dir}", flush=True)
        print(f"[INFO] Saved episodes   : {saved}", flush=True)


def close_app_and_exit(exit_code: int = 0) -> None:
    def force_exit() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(exit_code)

    timer = threading.Timer(5.0, force_exit)
    timer.daemon = True
    timer.start()
    try:
        simulation_app.close()
    except Exception:
        pass
    finally:
        timer.cancel()
        force_exit()


if __name__ == "__main__":
    main()
    close_app_and_exit(0)
