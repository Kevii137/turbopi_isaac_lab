from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import isaaclab.sim as sim_utils


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: tuple[str, ...] = ("go_left_oval", "go_right_horseshoe", "go_small_circle", "go_upper_right")
TASK_INSTRUCTIONS: dict[str, str] = {
    "go_left_oval": "go left oval",
    "go_right_horseshoe": "go right horseshoe",
    "go_small_circle": "go small circle",
    "go_upper_right": "go upper right teardrop",
}


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CustomACTSceneCfg:
    road_width: float = 0.48
    road_thickness: float = 0.035
    road_z: float = 0.02
    start_height: float = 0.055
    shoulder_width: float = 0.10
    wall_height: float = 0.04
    wall_thickness: float = 0.06
    floor_half_extent: float = 2.80
    floor_z: float = 0.001
    floor_color: tuple[float, float, float] = (0.16, 0.16, 0.16)
    road_color: tuple[float, float, float] = (0.07, 0.07, 0.08)
    shoulder_color: tuple[float, float, float] = (0.22, 0.22, 0.22)
    wall_color: tuple[float, float, float] = (0.00, 0.00, 0.00)
    sky_color: tuple[float, float, float] = (0.95, 0.95, 0.94)
    start_offset: float = 0.0


# ---------------------------------------------------------------------------
# Track centerline waypoints (Image-Accurate Topology)
# ---------------------------------------------------------------------------

# Shared entry trunk - Spawns on the middle-left horizontal, driving East to center
COMMON_ARM_CENTERLINE: tuple[tuple[float, float], ...] = (
    (-1.20,  0.00),
    (-0.60,  0.00),
    ( 0.00,  0.00),
)

# 1. Left Oval (Traces the top-left rectangular hole)
LEFT_OVAL_CENTERLINE: tuple[tuple[float, float], ...] = (
    ( 0.00,  0.00),
    ( 0.00,  1.60),
    (-0.40,  2.00),
    (-1.20,  2.00),
    (-1.60,  1.60),
    (-1.60,  0.40),
    (-1.20,  0.00),
    (-0.60,  0.00),
    ( 0.00,  0.00),
)

# 2. Upper Right Teardrop (Traces the top-right chamfered hole)
UPPER_RIGHT_TEARDROP_CENTERLINE: tuple[tuple[float, float], ...] = (
    ( 0.00,  0.00),
    ( 0.00,  1.60),
    ( 0.40,  2.00),
    ( 1.20,  2.00),
    ( 2.40,  0.80),   # 45-deg chamfer cut
    ( 2.40,  0.40),
    ( 2.00,  0.00),
    ( 1.00,  0.00),
    ( 0.00,  0.00),
)

# 3. Right Horseshoe (Traces the large bottom-right black parking block)
RIGHT_HORSESHOE_CENTERLINE: tuple[tuple[float, float], ...] = (
    ( 0.00,  0.00),
    ( 1.00,  0.00),
    ( 2.00,  0.00),
    ( 2.40, -0.40),
    ( 2.40, -2.00),
    ( 2.00, -2.40),
    (-1.20, -2.40),
    (-1.60, -2.00),   # Junction at bottom-left
    (-0.80, -1.00),   # Diagonal return
    ( 0.00,  0.00),
)

# 4. Small Circle (Drives down diagonal to the bottom-left loop)
SMALL_CIRCLE_CENTERLINE: tuple[tuple[float, float], ...] = (
    ( 0.00,  0.00),
    (-0.80, -1.00),   # Diagonal down
    (-1.60, -2.00),   # Arrive at circle junction
    (-2.20, -1.60),
    (-2.80, -2.20),
    (-2.20, -2.80),
    (-1.60, -2.40),   # Exit circle
    (-1.60, -1.20),   # Up left straight
    (-1.60, -0.40),
    (-1.20,  0.00),   # Right turn back to start
    (-0.60,  0.00),
    ( 0.00,  0.00),
)


def route_waypoints(scene_cfg: CustomACTSceneCfg, task_name: str) -> tuple[tuple[float, float], ...]:
    routes = {
        "go_left_oval": LEFT_OVAL_CENTERLINE,
        "go_right_horseshoe": RIGHT_HORSESHOE_CENTERLINE,
        "go_small_circle": SMALL_CIRCLE_CENTERLINE,
        "go_upper_right": UPPER_RIGHT_TEARDROP_CENTERLINE,
    }
    if task_name not in routes:
        raise ValueError(f"Unknown task name {task_name!r}")
    return routes[task_name]


def start_pose(scene_cfg: CustomACTSceneCfg) -> tuple[tuple[float, float, float], float]:
    start = COMMON_ARM_CENTERLINE[0]
    next_point = COMMON_ARM_CENTERLINE[1]
    yaw = math.atan2(next_point[1] - start[1], next_point[0] - start[0])
    return (start[0], start[1], scene_cfg.road_z + scene_cfg.start_height), yaw


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _preview(color: tuple[float, float, float], roughness: float = 0.90) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)

def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))

def _cuboid(prim_path: str, *, size: tuple[float, float, float], translation: tuple[float, float, float], color: tuple[float, float, float], collision: bool = False, roughness: float = 0.90, yaw: float = 0.0) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        physics_material=(
            sim_utils.RigidBodyMaterialCfg(friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=0.8, restitution=0.0) if collision else None
        ),
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation, orientation=_yaw_to_quat(yaw))

def segment_geometry(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float, float, float]:
    cx = 0.5 * (start[0] + end[0])
    cy = 0.5 * (start[1] + end[1])
    length = math.dist(start, end)
    yaw = math.atan2(end[1] - start[1], end[0] - start[0]) - 0.5 * math.pi
    return cx, cy, length, yaw


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def design_custom_act_scene(scene_cfg: CustomACTSceneCfg) -> None:
    ground_cfg = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8, restitution=0.0))
    ground_cfg.func("/World/ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=950.0, color=scene_cfg.sky_color)
    light_cfg.func("/World/Light", light_cfg)

    floor_size = 2.0 * scene_cfg.floor_half_extent
    _cuboid("/World/CustomACT/Floor", size=(floor_size, floor_size, 0.006), translation=(0.0, 0.0, scene_cfg.floor_z), color=scene_cfg.floor_color, collision=False)

    # ── Non-Overlapping Mesh Edge Generation ──────────────────────────────
    # To prevent Z-fighting in the simulation, we deconstruct the full routes 
    # into strictly unique topological edges so meshes are only drawn once.
    unique_generation_routes: dict[str, tuple[tuple[float, float], ...]] = {
        "MidVertical":   (( 0.00,  0.00), ( 0.00,  1.60)),
        "LeftOvalTop":   (( 0.00,  1.60), (-0.40,  2.00), (-1.20,  2.00), (-1.60,  1.60), (-1.60,  0.40), (-1.20,  0.00)),
        "MidHorizLeft":  ((-1.20,  0.00), ( 0.00,  0.00)),
        "MidHorizRight": (( 0.00,  0.00), ( 2.00,  0.00)),
        "TeardropTop":   (( 0.00,  1.60), ( 0.40,  2.00), ( 1.20,  2.00), ( 2.40,  0.80), ( 2.40,  0.40), ( 2.00,  0.00)),
        "HorseshoeBot":  (( 2.00,  0.00), ( 2.40, -0.40), ( 2.40, -2.00), ( 2.00, -2.40), (-1.20, -2.40), (-1.60, -2.00)),
        "Diagonal":      ((-1.60, -2.00), ( 0.00,  0.00)),
        "SmallCircle":   ((-1.60, -2.00), (-2.20, -1.60), (-2.80, -2.20), (-2.20, -2.80), (-1.60, -2.40)),
        "LeftVertLower": ((-1.60, -2.40), (-1.60, -0.40), (-1.20,  0.00)),
    }

    total_width = scene_cfg.road_width + 2.0 * scene_cfg.shoulder_width

    for route_name, points in unique_generation_routes.items():
        for idx, (start, end) in enumerate(zip(points[:-1], points[1:])):
            cx, cy, length, yaw = segment_geometry(start, end)
            _cuboid(f"/World/CustomACT/{route_name}RoadDeck{idx:02d}", size=(total_width, length + 0.04, scene_cfg.road_thickness), translation=(cx, cy, scene_cfg.road_z - 0.5 * scene_cfg.road_thickness), color=scene_cfg.shoulder_color, collision=False, yaw=yaw)
            _cuboid(f"/World/CustomACT/{route_name}RoadSurface{idx:02d}", size=(scene_cfg.road_width, length + 0.055, 0.005), translation=(cx, cy, scene_cfg.road_z + 0.004), color=scene_cfg.road_color, collision=False, yaw=yaw)

    # ── Boundary walls ─────────────────────────────────────────────────────
    wall_extent = scene_cfg.floor_half_extent + 0.5 * scene_cfg.wall_thickness
    wall_z = scene_cfg.floor_z + 0.5 * scene_cfg.wall_height

    _cuboid("/World/CustomACT/WallTop", size=(2.0 * scene_cfg.floor_half_extent + scene_cfg.wall_thickness, scene_cfg.wall_thickness, scene_cfg.wall_height), translation=(0.0, wall_extent, wall_z), color=scene_cfg.wall_color, collision=True)
    _cuboid("/World/CustomACT/WallBottom", size=(2.0 * scene_cfg.floor_half_extent + scene_cfg.wall_thickness, scene_cfg.wall_thickness, scene_cfg.wall_height), translation=(0.0, -wall_extent, wall_z), color=scene_cfg.wall_color, collision=True)
    _cuboid("/World/CustomACT/WallLeft", size=(scene_cfg.wall_thickness, 2.0 * scene_cfg.floor_half_extent + scene_cfg.wall_thickness, scene_cfg.wall_height), translation=(-wall_extent, 0.0, wall_z), color=scene_cfg.wall_color, collision=True)
    _cuboid("/World/CustomACT/WallRight", size=(scene_cfg.wall_thickness, 2.0 * scene_cfg.floor_half_extent + scene_cfg.wall_thickness, scene_cfg.wall_height), translation=(wall_extent, 0.0, wall_z), color=scene_cfg.wall_color, collision=True)