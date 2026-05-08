"""Build predecoded tensor shards for fast ACT training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from act_policy import DEFAULT_ACTION_CHUNK_SIZE, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH
from act_policy.dataset import (
    discover_episodes,
    discover_task_names,
    load_episode_actions,
    load_episode_frames,
    split_sessions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predecode ACT MP4/parquet episodes into fast tensor shards.")
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_ACTION_CHUNK_SIZE)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def make_action_chunks(actions: np.ndarray, chunk_size: int) -> torch.Tensor:
    chunks = np.empty((len(actions), chunk_size, actions.shape[1]), dtype=np.float32)
    for frame_idx in range(len(actions)):
        for offset in range(chunk_size):
            source_idx = min(frame_idx + offset, len(actions) - 1)
            chunks[frame_idx, offset] = actions[source_idx]
    return torch.from_numpy(chunks)


def build_split(records, task_to_index: dict[str, int], image_size: tuple[int, int], chunk_size: int, desc: str) -> dict[str, torch.Tensor | list[str]]:
    image_tensors: list[torch.Tensor] = []
    action_tensors: list[torch.Tensor] = []
    task_tensors: list[torch.Tensor] = []
    episode_dirs: list[str] = []
    for record in tqdm(records, desc=desc, dynamic_ncols=True):
        frames = load_episode_frames(record.episode_dir / "video.mp4", image_size)
        actions = load_episode_actions(record.episode_dir / "data.parquet")
        if len(frames) != len(actions):
            raise ValueError(f"Frame/action count mismatch in {record.episode_dir}: {len(frames)} vs {len(actions)}")
        images = torch.from_numpy(np.stack(frames, axis=0).astype(np.uint8, copy=False))
        chunks = make_action_chunks(actions, chunk_size)
        tasks = torch.full((len(frames),), task_to_index[record.task], dtype=torch.long)
        image_tensors.append(images)
        action_tensors.append(chunks)
        task_tensors.append(tasks)
        episode_dirs.append(str(record.episode_dir))
    if not image_tensors:
        raise RuntimeError(f"No records for {desc}")
    return {
        "images": torch.cat(image_tensors, dim=0),
        "action_chunks": torch.cat(action_tensors, dim=0),
        "task_indices": torch.cat(task_tensors, dim=0),
        "episode_dirs": episode_dirs,
    }


def main() -> None:
    args = build_parser().parse_args()
    episodes_dir = Path(args.episodes_dir)
    cache_dir = Path(args.cache_dir)
    train_path = cache_dir / "train.pt"
    val_path = cache_dir / "val.pt"
    metadata_path = cache_dir / "metadata.json"
    if cache_dir.exists() and any(cache_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Cache directory is not empty. Pass --overwrite to replace: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = discover_episodes(episodes_dir)
    if not records:
        raise SystemExit(f"No ACT episodes found under {episodes_dir}")
    task_names = discover_task_names(episodes_dir, records)
    task_to_index = {task: index for index, task in enumerate(task_names)}
    train_records = split_sessions(records, "train", args.val_ratio, args.seed)
    val_records = split_sessions(records, "val", args.val_ratio, args.seed)
    if not train_records or not val_records:
        raise SystemExit("Fast cache requires non-empty train and validation splits.")

    image_size = (args.image_width, args.image_height)
    train_payload = build_split(train_records, task_to_index, image_size, args.chunk_size, "cache train")
    val_payload = build_split(val_records, task_to_index, image_size, args.chunk_size, "cache val")
    for payload in (train_payload, val_payload):
        payload["task_names"] = task_names
        payload["image_size"] = [args.image_width, args.image_height]
        payload["chunk_size"] = args.chunk_size

    torch.save(train_payload, train_path)
    torch.save(val_payload, val_path)
    metadata = {
        "episodes_dir": str(episodes_dir),
        "task_names": task_names,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "chunk_size": args.chunk_size,
        "train_episodes": len(train_records),
        "val_episodes": len(val_records),
        "train_frames": int(train_payload["images"].shape[0]),
        "val_frames": int(val_payload["images"].shape[0]),
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
