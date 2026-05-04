"""Inference helpers for the task-conditioned TurboPi CNN policy."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .intent_dataset import frame_to_tensor
from .intent_model import load_checkpoint


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("mps")
    return torch.device("cpu")


def denormalize_action(action: np.ndarray, vx_cap: float, vy_cap: float, omega_cap: float) -> np.ndarray:
    clipped = np.clip(action, -1.0, 1.0)
    caps = np.asarray([vx_cap, vy_cap, omega_cap], dtype=np.float32)
    return clipped * caps


def apply_minimum_command_floor(
    command: np.ndarray,
    *,
    min_vx: float,
    min_vy: float,
    min_omega: float,
    zero_eps: float = 1e-4,
) -> np.ndarray:
    result = np.asarray(command, dtype=np.float32).copy()
    minimums = np.asarray([min_vx, min_vy, min_omega], dtype=np.float32)
    for index, floor in enumerate(minimums):
        if floor <= 0:
            continue
        value = float(result[index])
        if abs(value) <= zero_eps:
            result[index] = 0.0
            continue
        if abs(value) < floor:
            result[index] = np.sign(value) * floor
    return result


@dataclass(frozen=True)
class IntentPolicyRuntimeConfig:
    smoothing: float = 0.65
    vx_cap: float = 0.45
    vy_cap: float = 0.35
    omega_cap: float = 2.0
    min_vx: float = 0.0
    min_vy: float = 0.0
    min_omega: float = 0.0


class IntentPolicyRuntime:
    """Load a task-conditioned checkpoint and turn RGB frames into body commands."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        task: str | int = "go_left",
        device: str | torch.device = "auto",
        runtime_cfg: IntentPolicyRuntimeConfig | None = None,
    ):
        self.device = resolve_device(str(device))
        self.runtime_cfg = runtime_cfg or IntentPolicyRuntimeConfig()
        self.model, self.payload = load_checkpoint(Path(checkpoint), map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        extra = self.payload.get("extra", {})
        self.task_names = list(extra.get("task_names", [])) if isinstance(extra, dict) else []
        raw_task_to_index = extra.get("task_to_index", {}) if isinstance(extra, dict) else {}
        if isinstance(raw_task_to_index, dict):
            self.task_to_index = {str(key): int(value) for key, value in raw_task_to_index.items()}
        else:
            self.task_to_index = {task_name: index for index, task_name in enumerate(self.task_names)}
        if not self.task_names and self.task_to_index:
            self.task_names = [task for task, _idx in sorted(self.task_to_index.items(), key=lambda item: item[1])]
        if not self.task_to_index and self.task_names:
            self.task_to_index = {task_name: index for index, task_name in enumerate(self.task_names)}
        if not self.task_to_index:
            self.task_to_index = {"go_left": 0, "go_right": 1}
            self.task_names = ["go_left", "go_right"]

        self.image_width = int(self.model.config.image_width)
        self.image_height = int(self.model.config.image_height)
        self.frame_history = int(self.model.config.frame_history)
        self.buffer: collections.deque[np.ndarray] = collections.deque(maxlen=self.frame_history)
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.task_id = self.resolve_task_id(task)

    def resolve_task_id(self, task: str | int) -> int:
        if isinstance(task, int):
            task_id = int(task)
        else:
            key = str(task)
            if key not in self.task_to_index:
                valid = ", ".join(sorted(self.task_to_index))
                raise ValueError(f"Unknown task '{key}'. Available tasks: {valid}")
            task_id = int(self.task_to_index[key])
        if task_id < 0 or task_id >= int(self.model.config.task_vocab_size):
            raise ValueError(f"Task id {task_id} is outside checkpoint vocab size {self.model.config.task_vocab_size}")
        return task_id

    def set_task(self, task: str | int) -> None:
        self.task_id = self.resolve_task_id(task)

    def reset(self, initial_frame_rgb: np.ndarray | None = None, *, task: str | int | None = None) -> None:
        if task is not None:
            self.set_task(task)
        self.buffer.clear()
        self.previous_action.fill(0.0)
        if initial_frame_rgb is not None:
            for _ in range(self.frame_history):
                self.buffer.append(np.asarray(initial_frame_rgb, dtype=np.uint8).copy())

    def is_primed(self) -> bool:
        return len(self.buffer) >= self.frame_history

    def append_frame(self, frame_rgb: np.ndarray) -> None:
        self.buffer.append(np.asarray(frame_rgb, dtype=np.uint8).copy())

    def predict(
        self,
        frame_rgb: np.ndarray,
        *,
        task: str | int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if task is not None:
            self.set_task(task)
        self.append_frame(frame_rgb)
        if len(self.buffer) < self.frame_history:
            raise RuntimeError("Policy buffer is not primed yet.")

        stacked = torch.stack(
            [
                frame_to_tensor(img, image_width=self.image_width, image_height=self.image_height)
                for img in list(self.buffer)
            ],
            dim=0,
        ).reshape(1, self.frame_history * 3, self.image_height, self.image_width)
        task_ids = torch.tensor([self.task_id], dtype=torch.long, device=self.device)

        with torch.no_grad():
            pred = self.model(stacked.to(self.device), task_ids).squeeze(0).detach().cpu().numpy().astype(np.float32)

        alpha = float(np.clip(self.runtime_cfg.smoothing, 0.0, 0.99))
        smoothed = alpha * self.previous_action + (1.0 - alpha) * np.clip(pred, -1.0, 1.0)
        raw_command = denormalize_action(
            smoothed,
            vx_cap=self.runtime_cfg.vx_cap,
            vy_cap=self.runtime_cfg.vy_cap,
            omega_cap=self.runtime_cfg.omega_cap,
        )
        command = apply_minimum_command_floor(
            raw_command,
            min_vx=self.runtime_cfg.min_vx,
            min_vy=self.runtime_cfg.min_vy,
            min_omega=self.runtime_cfg.min_omega,
        )
        self.previous_action = smoothed.astype(np.float32, copy=True)
        return pred, smoothed.astype(np.float32), command.astype(np.float32)
