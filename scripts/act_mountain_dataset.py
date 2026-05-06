"""Episode writer for mountain ACT + language + CVAE data."""

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
    raise RuntimeError("Saving parquet requires pyarrow or fastparquet.")


@dataclass(frozen=True)
class ACTEpisodeFrame:
    image_rgb: np.ndarray
    timestamp: float
    state: np.ndarray
    action: np.ndarray
    command: np.ndarray
    track_error: float
    route_progress: float


@dataclass(frozen=True)
class ACTEpisodeResult:
    task_name: str
    task_index: int
    instruction: str
    frames: list[ACTEpisodeFrame]
    success: bool
    terminal_reason: str
    final_route_progress: float
    mean_track_error: float
    duration_s: float


class ACTMountainSessionWriter:
    def __init__(
        self,
        *,
        output_root: Path | str,
        session_name: str,
        dataset_name: str,
        fps: float,
        image_width: int,
        image_height: int,
        control_hz: float,
        physics_dt: float,
        tasks: tuple[str, ...],
        task_instructions: dict[str, str],
        record_camera: str = "robot",
    ):
        self.output_root = Path(output_root)
        self.session_name = session_name
        self.dataset_name = dataset_name
        self.fps = float(fps)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.control_hz = float(control_hz)
        self.physics_dt = float(physics_dt)
        self.tasks = tuple(tasks)
        self.task_instructions = dict(task_instructions)
        self.record_camera = record_camera
        self.parquet_engine = resolve_parquet_engine()
        self.session_dir = self.output_root / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.accepted_episodes = 0
        self.failed_attempts = 0
        self.total_frames = 0
        self._write_static_metadata()
        self._write_session_info()

    def record_failure(self) -> None:
        self.failed_attempts += 1
        self._write_session_info()

    def save_episode(self, episode_index: int, result: ACTEpisodeResult) -> Path:
        episode_dir = self.session_dir / f"episode_{episode_index:05d}"
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True)
        try:
            self._write_video(episode_dir / "video.mp4", result.frames)
            self._write_parquet(episode_dir / "data.parquet", result)
            self._write_episode_info(episode_dir / "episode_info.json", episode_index, result)
        except Exception:
            shutil.rmtree(episode_dir, ignore_errors=True)
            raise
        self.accepted_episodes += 1
        self.total_frames += len(result.frames)
        self._write_session_info()
        return episode_dir

    def _write_static_metadata(self) -> None:
        mapping = {
            "mode_family": "act",
            "policy_family": "act_cvae",
            "intent_mode": "language",
            "conditioning": "task_id",
            "tasks": list(self.tasks),
            "task_to_index": {task: idx for idx, task in enumerate(self.tasks)},
            "task_instructions": self.task_instructions,
        }
        (self.session_dir / "task_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        (self.session_dir / "tasks.json").write_text(json.dumps(list(self.tasks), indent=2), encoding="utf-8")

    def _write_session_info(self) -> None:
        payload = {
            "last_updated_at": utc_now(),
            "session_name": self.session_name,
            "dataset_name": self.dataset_name,
            "simulator": "Isaac Lab",
            "robot_type": "Hiwonder TurboPi",
            "mode_family": "act",
            "policy_family": "act_cvae",
            "intent_mode": "language",
            "conditioning": "task_id",
            "task_type": "instruction_conditioned_mountain_fork_following",
            "track_layout": "mountain_cliff_fork",
            "collection_style": "autonomous_teacher",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "control_hz": self.control_hz,
            "physics_dt": self.physics_dt,
            "fps": self.fps,
            "record_camera": self.record_camera,
            "accepted_episodes": self.accepted_episodes,
            "failed_attempts": self.failed_attempts,
            "total_frames": self.total_frames,
        }
        (self.session_dir / "session_info.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_video(self, path: Path, frames: list[ACTEpisodeFrame]) -> None:
        if not frames:
            raise RuntimeError("Cannot save empty episode.")
        height, width = frames[0].image_rgb.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if writer.isOpened():
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            return
        writer.release()
        imageio.mimwrite(str(path), [frame.image_rgb for frame in frames], fps=self.fps, codec="libx264", macro_block_size=None)

    def _write_parquet(self, path: Path, result: ACTEpisodeResult) -> None:
        rows = []
        for frame_index, frame in enumerate(result.frames):
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp": float(frame.timestamp),
                    "state": np.asarray(frame.state, dtype=np.float32).tolist(),
                    "action": np.asarray(frame.action, dtype=np.float32).tolist(),
                    "command": np.asarray(frame.command, dtype=np.float32).tolist(),
                    "track_error": float(frame.track_error),
                    "route_progress": float(frame.route_progress),
                    "task": result.task_name,
                    "task_index": int(result.task_index),
                    "instruction": result.instruction,
                }
            )
        pd.DataFrame(rows).to_parquet(path, engine=self.parquet_engine, index=False)

    def _write_episode_info(self, path: Path, episode_index: int, result: ACTEpisodeResult) -> None:
        payload = {
            "episode_index": episode_index,
            "mode_family": "act",
            "policy_family": "act_cvae",
            "intent_mode": "language",
            "conditioning": "task_id",
            "task_type": "instruction_conditioned_mountain_fork_following",
            "track_layout": "mountain_cliff_fork",
            "task_name": result.task_name,
            "task_index": int(result.task_index),
            "instruction": result.instruction,
            "num_frames": len(result.frames),
            "duration_s": float(result.duration_s),
            "success": bool(result.success),
            "terminal_reason": result.terminal_reason,
            "final_route_progress": float(result.final_route_progress),
            "mean_track_error": float(result.mean_track_error),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
