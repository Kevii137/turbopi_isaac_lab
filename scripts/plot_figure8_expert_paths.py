"""Plot reconstructed figure-8 ACT expert paths from saved episodes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE8_COMMON_ARM_CENTERLINE: tuple[tuple[float, float], ...] = (
    (0.00, -1.45),
    (0.00, -0.68),
    (0.00, 0.10),
    (0.00, 0.88),
)

FIGURE8_LEFT_LOOP_CENTERLINE: tuple[tuple[float, float], ...] = (
    (0.00, 0.88),
    (-0.91, 0.72),
    (-1.58, 0.30),
    (-1.82, -0.28),
    (-1.58, -0.87),
    (-0.91, -1.29),
    (0.00, -1.45),
)

FIGURE8_RIGHT_LOOP_CENTERLINE: tuple[tuple[float, float], ...] = (
    (0.00, 0.88),
    (0.91, 0.72),
    (1.58, 0.30),
    (1.82, -0.28),
    (1.58, -0.87),
    (0.91, -1.29),
    (0.00, -1.45),
)

FIGURE8_LEFT_ROUTE: tuple[tuple[float, float], ...] = (
    *FIGURE8_COMMON_ARM_CENTERLINE,
    *FIGURE8_LEFT_LOOP_CENTERLINE[1:],
)

FIGURE8_RIGHT_ROUTE: tuple[tuple[float, float], ...] = (
    *FIGURE8_COMMON_ARM_CENTERLINE,
    *FIGURE8_RIGHT_LOOP_CENTERLINE[1:],
)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def repeated_waypoints(task_name: str, laps: int) -> np.ndarray:
    base = list(FIGURE8_LEFT_ROUTE if task_name == "go_left" else FIGURE8_RIGHT_ROUTE)
    points = list(base)
    for _ in range(max(1, laps) - 1):
        points.extend(base[1:])
    return np.asarray(points, dtype=np.float32)


def points_from_progress(task_name: str, progress: np.ndarray, laps: int) -> np.ndarray:
    route = repeated_waypoints(task_name, laps)
    seg = route[1:] - route[:-1]
    lengths = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    dist = np.clip(progress.astype(np.float32), 0.0, 1.0) * total
    idx = np.searchsorted(cumulative[1:], dist, side="right")
    idx = np.clip(idx, 0, len(lengths) - 1)
    local = np.clip((dist - cumulative[idx]) / np.maximum(lengths[idx], 1e-6), 0.0, 1.0)
    return route[idx] + local[:, None] * seg[idx]


def integrate(commands: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    start = FIGURE8_LEFT_ROUTE[0]
    next_point = FIGURE8_LEFT_ROUTE[1]
    yaw = math.atan2(next_point[1] - start[1], next_point[0] - start[0])
    xy = np.empty((len(commands) + 1, 2), dtype=np.float32)
    x = float(start[0])
    y = float(start[1])
    xy[0] = (x, y)
    default_dt = 0.1
    if len(timestamps) > 1:
        diffs = np.diff(timestamps)
        positive = diffs[diffs > 0.0]
        if len(positive):
            default_dt = float(np.median(positive))
    for idx, command in enumerate(commands):
        dt = float(timestamps[idx + 1] - timestamps[idx]) if idx + 1 < len(timestamps) else default_dt
        vx, vy, wz = (float(command[0]), float(command[1]), float(command[2]))
        yaw_mid = yaw + 0.5 * wz * dt
        x += (vx * math.cos(yaw_mid) - vy * math.sin(yaw_mid)) * dt
        y += (vx * math.sin(yaw_mid) + vy * math.cos(yaw_mid)) * dt
        yaw = wrap_to_pi(yaw + wz * dt)
        xy[idx + 1] = (x, y)
    return xy


def load_episode(path: Path, laps: int) -> dict[str, object]:
    df = pd.read_parquet(path / "data.parquet")
    info = json.loads((path / "episode_info.json").read_text(encoding="utf-8"))
    commands = np.asarray(df["command"].to_list(), dtype=np.float32)
    timestamps = df["timestamp"].to_numpy(dtype=np.float32)
    route_progress = df["route_progress"].to_numpy(dtype=np.float32)
    xy = integrate(commands, timestamps)
    if {"pose_x", "pose_y"}.issubset(df.columns) and df["pose_x"].notna().all() and df["pose_y"].notna().all():
        expert_xy = df[["pose_x", "pose_y"]].to_numpy(dtype=np.float32)
        expert_source = "saved_pose"
    else:
        expert_xy = points_from_progress(str(info["task_name"]), route_progress, laps)
        expert_source = "route_progress_projection"
    return {
        "path": path,
        "task": str(info["task_name"]),
        "frames": int(info["num_frames"]),
        "duration_s": float(info["duration_s"]),
        "final_route_progress": float(info["final_route_progress"]),
        "mean_track_error": float(info["mean_track_error"]),
        "max_track_error": float(df["track_error"].max()),
        "xy": xy,
        "progress_xy": points_from_progress(str(info["task_name"]), route_progress, laps),
        "expert_xy": expert_xy,
        "expert_source": expert_source,
    }


def plot_task(ax, task_name: str, episodes: list[dict[str, object]], laps: int, color: str) -> None:
    ref = repeated_waypoints(task_name, laps)
    ax.plot(ref[:, 0], ref[:, 1], color="black", lw=2.2, linestyle="--", label="reference route")
    for episode in episodes:
        xy = episode["expert_xy"]
        assert isinstance(xy, np.ndarray)
        ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=0.28, lw=1.0)
    ax.scatter(ref[0, 0], ref[0, 1], c="limegreen", s=70, zorder=5, label="start")
    ax.scatter(ref[3, 0], ref[3, 1], c="gold", edgecolors="black", s=70, zorder=5, label="decision point")
    ax.set_title(f"{task_name}: {len(episodes)} expert episodes")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)


def plot_progress_task(ax, task_name: str, episodes: list[dict[str, object]], laps: int, color: str) -> None:
    ref = repeated_waypoints(task_name, laps)
    ax.plot(ref[:, 0], ref[:, 1], color="0.72", lw=5.0, solid_capstyle="round", label="reference route")
    for episode in episodes:
        xy = episode["progress_xy"]
        assert isinstance(xy, np.ndarray)
        ax.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.18, s=8, edgecolors="none")
    ax.scatter(ref[0, 0], ref[0, 1], c="limegreen", s=70, zorder=5, label="start")
    ax.scatter(ref[3, 0], ref[3, 1], c="gold", edgecolors="black", s=70, zorder=5, label="decision point")
    ax.set_title(f"{task_name}: stored route-progress samples")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)


def plot_map_overlay(episodes: list[dict[str, object]], laps: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    left_ref = repeated_waypoints("go_left", 1)
    right_ref = repeated_waypoints("go_right", 1)

    # Draw the drivable figure-8 map first. Line widths are visual, but keep
    # the road/shoulder relationship close to the scene configuration.
    for ref in (left_ref, right_ref):
        ax.plot(ref[:, 0], ref[:, 1], color="#8a6a42", lw=46, solid_capstyle="round", solid_joinstyle="round", zorder=1)
        ax.plot(ref[:, 0], ref[:, 1], color="#3a2a1c", lw=28, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.plot(ref[:, 0], ref[:, 1], color="#d7c27a", lw=2.0, alpha=0.75, linestyle=(0, (5, 8)), zorder=3)

    colors = {"go_left": "tab:blue", "go_right": "tab:red"}
    labels_seen: set[str] = set()
    for episode in episodes:
        task = str(episode["task"])
        xy = episode["expert_xy"]
        assert isinstance(xy, np.ndarray)
        label = f"{task} expert samples" if task not in labels_seen else None
        labels_seen.add(task)
        ax.plot(xy[:, 0], xy[:, 1], color=colors.get(task, "white"), alpha=0.28, lw=1.0, zorder=5, label=label)
        ax.scatter(xy[::4, 0], xy[::4, 1], color=colors.get(task, "white"), alpha=0.18, s=5, edgecolors="none", zorder=6)

    start = left_ref[0]
    decision = left_ref[3]
    ax.scatter(start[0], start[1], c="limegreen", s=120, edgecolors="black", linewidths=0.8, zorder=8, label="start")
    ax.scatter(decision[0], decision[1], c="gold", s=120, edgecolors="black", linewidths=0.8, zorder=8, label="decision point")

    ax.set_title("Ground-Truth Expert Paths Overlaid On Figure-8 Map", fontsize=15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.18)
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-1.95, 1.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reconstructed figure-8 expert paths from ACT dataset episodes.")
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/figure8_expert_path_audit")
    parser.add_argument("--laps", type=int, default=3)
    args = parser.parse_args()

    episodes_root = Path(args.episodes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_dirs = sorted(path for path in episodes_root.glob("**/episode_*") if (path / "data.parquet").exists())
    if not episode_dirs:
        raise SystemExit(f"No episodes found under {episodes_root}")

    episodes = [load_episode(path, args.laps) for path in episode_dirs]
    left = [episode for episode in episodes if episode["task"] == "go_left"]
    right = [episode for episode in episodes if episode["task"] == "go_right"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharex=True, sharey=True)
    plot_task(axes[0], "go_left", left, args.laps, "tab:blue")
    plot_task(axes[1], "go_right", right, args.laps, "tab:red")
    fig.suptitle("Reconstructed Ground-Truth Expert Paths From Training Dataset", fontsize=14)
    fig.tight_layout()
    overview_path = out_dir / "figure8_expert_paths_overlay.png"
    fig.savefig(overview_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharex=True, sharey=True)
    plot_progress_task(axes[0], "go_left", left, args.laps, "tab:blue")
    plot_progress_task(axes[1], "go_right", right, args.laps, "tab:red")
    fig.suptitle("Stored Ground-Truth Route-Progress Coverage From Training Dataset", fontsize=14)
    fig.tight_layout()
    progress_path = out_dir / "figure8_expert_progress_projection.png"
    fig.savefig(progress_path, dpi=160)
    plt.close(fig)

    map_overlay_path = out_dir / "figure8_expert_paths_on_topdown_map.png"
    plot_map_overlay(episodes, args.laps, map_overlay_path)

    rows = []
    for episode in episodes:
        rows.append(
            {
                "episode": episode["path"].name,
                "task": episode["task"],
                "frames": episode["frames"],
                "duration_s": episode["duration_s"],
                "final_route_progress": episode["final_route_progress"],
                "mean_track_error": episode["mean_track_error"],
                "max_track_error": episode["max_track_error"],
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "figure8_expert_paths_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[OK] wrote {overview_path}")
    print(f"[OK] wrote {progress_path}")
    print(f"[OK] wrote {map_overlay_path}")
    print(f"[OK] wrote {summary_path}")
    print()
    print(summary.groupby("task").agg(
        episodes=("episode", "count"),
        frames=("frames", "sum"),
        mean_duration_s=("duration_s", "mean"),
        mean_track_error=("mean_track_error", "mean"),
        max_track_error=("max_track_error", "max"),
        min_final_progress=("final_route_progress", "min"),
    ).to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
