from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_parquet_engine() -> str:
    for engine_name in ("pyarrow", "fastparquet"):
        try:
            __import__(engine_name)
            return engine_name
        except Exception:
            continue
    raise RuntimeError(
        "Saving `data.parquet` requires `pyarrow` or `fastparquet` in the Isaac Lab Python environment.\n"
        "Install one with:\n"
        "  /workspace/isaaclab/_isaac_sim/python.sh -m pip install pyarrow"
    )


@dataclass(frozen=True)
class IntentEpisodeFrame:
    image_rgb: np.ndarray
    timestamp: float
    state: np.ndarray
    action: np.ndarray
    command: np.ndarray
    body_velocity: np.ndarray
    track_error: float
    route_progress: float


@dataclass(frozen=True)
class IntentEpisodeResult:
    task_name: str
    task_index: int
    instruction: str
    frames: list[IntentEpisodeFrame]
    success: bool
    terminal_reason: str
    final_route_progress: float
    mean_track_error: float
    p90_track_error: float
    max_track_error: float
    frames_over_010_ratio: float
    frames_over_015_ratio: float
    mean_image_std: float
    min_image_std: float
    mean_abs_action_vx: float
    mean_abs_action_vy: float
    mean_abs_action_wz: float
    mean_action_vy_vx_ratio: float
    mean_speed: float
    duration_s: float


class CliffIntentSessionWriter:
    """Write cliff intent rollouts in the same episode-folder style as the CNN pipeline."""

    def __init__(
        self,
        *,
        output_root: Path | str,
        session_name: str,
        dataset_name: str,
        fps: float,
        image_width: int,
        image_height: int,
        episode_time_s: float,
        control_hz: float,
        physics_dt: float,
        tasks: tuple[str, ...] = ("go_left", "go_right"),
        task_instructions: dict[str, str] | None = None,
        track_layout: str = "cliff_center_split_rectangle",
        episode_definition: str = "start_center_straight_turn_intent_return_to_start",
    ):
        self.output_root = Path(output_root)
        self.session_name = session_name
        self.dataset_name = dataset_name
        self.fps = float(fps)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.episode_time_s = float(episode_time_s)
        self.control_hz = float(control_hz)
        self.physics_dt = float(physics_dt)
        self.tasks = tuple(tasks)
        self.task_instructions = dict(task_instructions or {})
        self.track_layout = track_layout
        self.episode_definition = episode_definition
        self.parquet_engine = resolve_parquet_engine()

        self.session_dir = self.output_root / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.collection_summary_path = self.session_dir / "collection_summary.json"
        self.task_mapping_path = self.session_dir / "task_mapping.json"
        self.tasks_path = self.session_dir / "tasks.json"
        self.session_info_path = self.session_dir / "session_info.json"

        self.accepted_episodes = 0
        self.failed_attempts = 0
        self.total_frames = 0
        self._write_task_mapping()
        self._write_session_info(created=True)
        self._write_collection_summary()

    def record_failure(self) -> None:
        self.failed_attempts += 1
        self._write_session_info(created=False)
        self._write_collection_summary()

    def save_episode(self, episode_index: int, result: IntentEpisodeResult) -> Path:
        episode_dir = self.session_dir / f"episode_{episode_index:05d}"
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=False)

        try:
            self._write_video(episode_dir / "video.mp4", result.frames)
            self._write_parquet(episode_dir / "data.parquet", result)
            self._write_episode_info(episode_dir / "episode_info.json", episode_index, result)
        except Exception:
            shutil.rmtree(episode_dir, ignore_errors=True)
            raise

        self.accepted_episodes += 1
        self.total_frames += len(result.frames)
        self._write_session_info(created=False)
        self._write_collection_summary()
        return episode_dir

    def _write_task_mapping(self) -> None:
        payload = {
            "mode_family": "cnn",
            "intent_mode": "language",
            "conditioning": "task_id",
            "tasks": list(self.tasks),
            "task_to_index": {name: idx for idx, name in enumerate(self.tasks)},
            "task_instructions": {
                task: self.task_instructions.get(task, task.replace("_", " "))
                for task in self.tasks
            },
        }
        self.task_mapping_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.tasks_path.write_text(json.dumps(list(self.tasks), indent=2), encoding="utf-8")

    def _write_session_info(self, *, created: bool) -> None:
        created_at = utc_now()
        resume_count = 0
        if self.session_info_path.exists():
            try:
                existing = json.loads(self.session_info_path.read_text(encoding="utf-8"))
                created_at = existing.get("created_at", created_at)
                resume_count = int(existing.get("resume_count", 0))
            except Exception:
                resume_count = 0

        payload = {
            "created_at": created_at,
            "last_updated_at": utc_now(),
            "resume_count": resume_count,
            "session_name": self.session_name,
            "dataset_name": self.dataset_name,
            "simulator": "Isaac Lab",
            "robot_type": "Hiwonder TurboPi",
            "fps": self.fps,
            "control_hz": self.control_hz,
            "physics_dt": self.physics_dt,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "episode_time_s": self.episode_time_s,
            "mode_family": "cnn",
            "intent_mode": "language",
            "conditioning": "task_id",
            "task_type": "instruction_conditioned_path_following",
            "track_layout": self.track_layout,
            "allowed_tasks": list(self.tasks),
            "task_instructions": {
                task: self.task_instructions.get(task, task.replace("_", " "))
                for task in self.tasks
            },
            "episode_definition": self.episode_definition,
            "collection_style": "autonomous_teacher",
            "observation_state_semantics": "previous_action_normalized",
            "action_semantics": "current_action_normalized",
            "task_mapping_file": self.task_mapping_path.name,
            "tasks_file": self.tasks_path.name,
            "accepted_episodes": self.accepted_episodes,
            "failed_attempts": self.failed_attempts,
            "total_frames": self.total_frames,
        }
        self.session_info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_collection_summary(self) -> None:
        payload = {
            "updated_at": utc_now(),
            "session_name": self.session_name,
            "accepted_episodes": self.accepted_episodes,
            "failed_attempts": self.failed_attempts,
            "total_frames": self.total_frames,
        }
        self.collection_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_video(self, path: Path, frames: list[IntentEpisodeFrame]) -> None:
        if not frames:
            raise RuntimeError("Refusing to save an empty episode video.")

        height, width = frames[0].image_rgb.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if writer.isOpened():
            try:
                for frame in frames:
                    image_bgr = cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2BGR)
                    writer.write(image_bgr)
            finally:
                writer.release()
            return

        writer.release()
        imageio.mimwrite(
            str(path),
            [frame.image_rgb for frame in frames],
            fps=self.fps,
            codec="libx264",
            macro_block_size=None,
        )

    def _write_parquet(self, path: Path, result: IntentEpisodeResult) -> None:
        rows = []
        for frame_index, frame in enumerate(result.frames):
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp": float(frame.timestamp),
                    "state": np.asarray(frame.state, dtype=np.float32).tolist(),
                    "action": np.asarray(frame.action, dtype=np.float32).tolist(),
                    "command": np.asarray(frame.command, dtype=np.float32).tolist(),
                    "body_velocity": np.asarray(frame.body_velocity, dtype=np.float32).tolist(),
                    "track_error": float(frame.track_error),
                    "lap_progress": float(frame.route_progress),
                    "route_progress": float(frame.route_progress),
                    "task": result.task_name,
                    "task_index": int(result.task_index),
                    "intent": result.task_name,
                    "instruction": result.instruction,
                }
            )
        pd.DataFrame(rows).to_parquet(path, engine=self.parquet_engine, index=False)

    def _write_episode_info(self, path: Path, episode_index: int, result: IntentEpisodeResult) -> None:
        payload = {
            "episode_index": episode_index,
            "direction": result.task_name,
            "mode_family": "cnn",
            "intent_mode": "language",
            "conditioning": "task_id",
            "task_type": "instruction_conditioned_path_following",
            "track_layout": self.track_layout,
            "episode_definition": self.episode_definition,
            "collection_style": "autonomous_teacher",
            "task_name": result.task_name,
            "task_index": int(result.task_index),
            "instruction": result.instruction,
            "num_frames": len(result.frames),
            "duration_s": float(result.duration_s),
            "success": bool(result.success),
            "terminal_reason": result.terminal_reason,
            "final_lap_progress": float(result.final_route_progress),
            "final_route_progress": float(result.final_route_progress),
            "mean_track_error": float(result.mean_track_error),
            "p90_track_error": float(result.p90_track_error),
            "max_track_error": float(result.max_track_error),
            "frames_over_010_ratio": float(result.frames_over_010_ratio),
            "frames_over_015_ratio": float(result.frames_over_015_ratio),
            "mean_image_std": float(result.mean_image_std),
            "min_image_std": float(result.min_image_std),
            "mean_abs_action_vx": float(result.mean_abs_action_vx),
            "mean_abs_action_vy": float(result.mean_abs_action_vy),
            "mean_abs_action_wz": float(result.mean_abs_action_wz),
            "mean_action_vy_vx_ratio": float(result.mean_action_vy_vx_ratio),
            "mean_speed": float(result.mean_speed),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
