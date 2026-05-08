#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/isaaclab}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/act_figure8}"
EPISODES_PER_WORKER="${EPISODES_PER_WORKER:-50}"
WORKERS_PER_INTENT="${WORKERS_PER_INTENT:-1}"
LAPS="${LAPS:-3}"
CONTROL_HZ="${CONTROL_HZ:-10}"
PHYSICS_DT="${PHYSICS_DT:-0.03333333333333333}"
IMAGE_WIDTH="${IMAGE_WIDTH:-128}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-128}"
TARGET_SPEED="${TARGET_SPEED:-0.42}"
MAX_EPISODE_TIME="${MAX_EPISODE_TIME:-120}"
DATASET_NAME="${DATASET_NAME:-turbopi_figure8_act_cvae}"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
EXTRA_ARGS=("$@")

mkdir -p "${OUTPUT_DIR}/logs"

launch_worker() {
  local task_name="$1"
  local worker_index="$2"
  local seed="$3"
  local session_name="figure8_${task_name}_w${worker_index}_${RUN_STAMP}"
  local log_path="${OUTPUT_DIR}/logs/${session_name}.log"

  echo "[collect] launching ${session_name} -> ${log_path}"
  (
    cd "${ISAACLAB_ROOT}"
    ./isaaclab.sh -p "${REPO_ROOT}/scripts/record_turbopi_mountain_act.py" \
      --headless \
      --map figure8 \
      --task "${task_name}" \
      --laps "${LAPS}" \
      --num_episodes "${EPISODES_PER_WORKER}" \
      --output_dir "${OUTPUT_DIR}" \
      --session_name "${session_name}" \
      --dataset_name "${DATASET_NAME}" \
      --control_hz "${CONTROL_HZ}" \
      --physics_dt "${PHYSICS_DT}" \
      --image_width "${IMAGE_WIDTH}" \
      --image_height "${IMAGE_HEIGHT}" \
      --target_speed "${TARGET_SPEED}" \
      --max_episode_time "${MAX_EPISODE_TIME}" \
      --seed "${seed}" \
      --no_rollers \
      "${EXTRA_ARGS[@]}"
  ) >"${log_path}" 2>&1 &
}

seed_base="${SEED:-1000}"
for worker_index in $(seq 0 "$((WORKERS_PER_INTENT - 1))"); do
  launch_worker "go_left" "${worker_index}" "$((seed_base + worker_index))"
  launch_worker "go_right" "${worker_index}" "$((seed_base + 1000 + worker_index))"
done

wait
echo "[collect] complete. Output: ${OUTPUT_DIR}"
