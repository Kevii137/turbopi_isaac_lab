from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils


@dataclass(frozen=True)
class CliffRoadSceneCfg:
    """High cliff-side road scene for TurboPi visual inspection."""

    road_length: float = 2.20
    road_width: float = 0.36
    rectangle_half_width: float = 0.78
    deck_thickness: float = 0.08
    cliff_height: float = 1.35
    start_height: float = 0.04
    marker_width: float = 0.018
    lower_terrain_z: float = -0.32


def _preview(color: tuple[float, float, float], roughness: float = 0.85) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)


def _cuboid(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _cylinder(
    prim_path: str,
    *,
    radius: float,
    height: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _cone(
    prim_path: str,
    *,
    radius: float,
    height: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.ConeCfg(
        radius=radius,
        height=height,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _sphere(
    prim_path: str,
    *,
    radius: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool = False,
    roughness: float = 0.85,
) -> None:
    cfg = sim_utils.SphereCfg(
        radius=radius,
        collision_props=sim_utils.CollisionPropertiesCfg() if collision else None,
        visual_material=_preview(color, roughness),
    )
    cfg.func(prim_path, cfg, translation=translation)


def design_cliff_road_scene(scene_cfg: CliffRoadSceneCfg) -> None:
    """Spawn an elevated road where steering mistakes can fall off the cliff."""

    sky_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.72, 0.82, 1.0))
    sky_cfg.func("/World/SkyLight", sky_cfg)
    sun_cfg = sim_utils.DistantLightCfg(intensity=1200.0, color=(1.0, 0.92, 0.78), angle=0.35)
    sun_cfg.func("/World/SunLight", sun_cfg)

    deck_top_z = scene_cfg.cliff_height
    deck_center_z = deck_top_z - 0.5 * scene_cfg.deck_thickness
    cliff_top_z = deck_top_z - scene_cfg.deck_thickness - 0.025
    cliff_face_height = cliff_top_z - scene_cfg.lower_terrain_z
    cliff_face_center_z = scene_cfg.lower_terrain_z + 0.5 * cliff_face_height
    marker_z = deck_top_z + 0.004
    half_y = 0.5 * scene_cfg.road_length
    half_x = scene_cfg.rectangle_half_width
    path_w = scene_cfg.road_width
    road_color = (0.48, 0.55, 0.53)

    # Four blue backdrop panels keep the scene from falling back to the gray
    # viewport background from any viewer angle.
    sky_color = (0.42, 0.72, 0.92)
    band_color = (0.80, 0.70, 0.55)
    panel_extent = 4.80
    panel_half = 2.05
    panel_height = 3.20
    panel_center_z = 0.78
    _cuboid(
        "/World/CliffRoad/SkyBackPanel",
        size=(panel_extent, 0.018, panel_height),
        translation=(0.0, panel_half, panel_center_z),
        color=sky_color,
        collision=False,
        roughness=1.0,
    )
    _cuboid(
        "/World/CliffRoad/SkyLeftPanel",
        size=(0.018, panel_extent, panel_height),
        translation=(-panel_half, 0.0, panel_center_z),
        color=sky_color,
        collision=False,
        roughness=1.0,
    )
    _cuboid(
        "/World/CliffRoad/SkyRightPanel",
        size=(0.018, panel_extent, panel_height),
        translation=(panel_half, 0.0, panel_center_z),
        color=sky_color,
        collision=False,
        roughness=1.0,
    )
    for name, size, translation in (
        ("Back", (panel_extent, 0.020, 0.22), (0.0, panel_half - 0.012, -0.10)),
        ("Left", (0.020, panel_extent, 0.22), (-panel_half + 0.012, 0.0, -0.10)),
        ("Right", (0.020, panel_extent, 0.22), (panel_half - 0.012, 0.0, -0.10)),
    ):
        _cuboid(
            f"/World/CliffRoad/{name}LowerBand",
            size=size,
            translation=translation,
            color=band_color,
            collision=False,
            roughness=1.0,
        )
    track_pieces = (
        ("BottomTrack", (2.0 * half_x + path_w, path_w, scene_cfg.deck_thickness), (0.0, -half_y, deck_center_z)),
        ("TopTrack", (2.0 * half_x + path_w, path_w, scene_cfg.deck_thickness), (0.0, half_y, deck_center_z)),
        ("LeftTrack", (path_w, 2.0 * half_y + path_w, scene_cfg.deck_thickness), (-half_x, 0.0, deck_center_z)),
        ("RightTrack", (path_w, 2.0 * half_y + path_w, scene_cfg.deck_thickness), (half_x, 0.0, deck_center_z)),
        ("CenterStraightTrack", (path_w, 2.0 * half_y + path_w, scene_cfg.deck_thickness), (0.0, 0.0, deck_center_z)),
    )
    for name, size, translation in track_pieces:
        _cuboid(
            f"/World/CliffRoad/{name}",
            size=size,
            translation=translation,
            color=road_color,
            collision=True,
            roughness=0.95,
        )

    _cuboid(
        "/World/CliffRoad/StartLine",
        size=(path_w, 0.050, 0.007),
        translation=(0.0, -half_y - 0.5 * path_w + 0.05, marker_z + 0.002),
        color=(0.92, 0.92, 0.86),
        collision=False,
    )

    # Visual cliff supports sit below the track pieces. They do not provide
    # collision, so falling off the track is still a real physics failure.
    support_pieces = (
        ("BottomCliffFace", (2.0 * half_x + path_w, 0.14, cliff_face_height), (0.0, -half_y, cliff_face_center_z)),
        ("TopCliffFace", (2.0 * half_x + path_w, 0.14, cliff_face_height), (0.0, half_y, cliff_face_center_z)),
        ("LeftCliffFace", (0.14, 2.0 * half_y + path_w, cliff_face_height), (-half_x, 0.0, cliff_face_center_z)),
        ("RightCliffFace", (0.14, 2.0 * half_y + path_w, cliff_face_height), (half_x, 0.0, cliff_face_center_z)),
        ("CenterCliffFace", (0.12, 2.0 * half_y + path_w, cliff_face_height), (0.0, 0.0, cliff_face_center_z)),
    )
    for name, size, translation in support_pieces:
        _cuboid(
            f"/World/CliffRoad/{name}",
            size=size,
            translation=translation,
            color=(0.42, 0.31, 0.25),
            collision=False,
        )
    for idx, z in enumerate((0.12, 0.42, 0.74, 1.02)):
        for side_name, x_pos in (("Left", -half_x), ("Right", half_x), ("Center", 0.0)):
            _cuboid(
                f"/World/CliffRoad/{side_name}RockLayer{idx:02d}",
                size=(0.150 if side_name != "Center" else 0.130, 2.0 * half_y + path_w + 0.015, 0.030),
                translation=(x_pos, 0.0, z),
                color=(0.58, 0.43, 0.29) if idx % 2 == 0 else (0.26, 0.22, 0.20),
                collision=False,
            )
    _cuboid(
        "/World/CliffRoad/RiverBelow",
        size=(2.80, scene_cfg.road_length + 0.80, 0.004),
        translation=(0.0, 0.0, scene_cfg.lower_terrain_z),
        color=(0.02, 0.32, 0.62),
        collision=False,
        roughness=0.25,
    )
    for idx, x in enumerate((-0.56, -0.18, 0.26, 0.68)):
        _cuboid(
            f"/World/CliffRoad/RiverHighlight{idx:02d}",
            size=(0.22, scene_cfg.road_length + 0.40, 0.003),
            translation=(x, 0.0, scene_cfg.lower_terrain_z + 0.004),
            color=(0.32, 0.68, 0.88),
            collision=False,
            roughness=0.18,
        )
    _cuboid(
        "/World/CliffRoad/LowerSandbar",
        size=(0.56, scene_cfg.road_length + 0.20, 0.006),
        translation=(-0.92, 0.0, scene_cfg.lower_terrain_z + 0.006),
        color=(0.69, 0.56, 0.34),
        collision=False,
    )
    _cuboid(
        "/World/CliffRoad/LowerMeadow",
        size=(0.62, scene_cfg.road_length + 0.20, 0.006),
        translation=(0.92, 0.0, scene_cfg.lower_terrain_z + 0.006),
        color=(0.10, 0.36, 0.20),
        collision=False,
    )

    # Background color bands are intentionally outside the driving corridor.
    _cuboid(
        "/World/CliffRoad/FarHillA",
        size=(0.24, 0.72, 0.52),
        translation=(-1.18, -0.74, scene_cfg.lower_terrain_z + 0.26),
        color=(0.22, 0.43, 0.24),
        collision=False,
    )
    _cuboid(
        "/World/CliffRoad/FarHillB",
        size=(0.24, 0.92, 0.68),
        translation=(-1.22, 0.32, scene_cfg.lower_terrain_z + 0.34),
        color=(0.27, 0.50, 0.28),
        collision=False,
    )
    _cuboid(
        "/World/CliffRoad/FarRock",
        size=(0.28, 0.36, 0.36),
        translation=(1.08, 1.04, scene_cfg.lower_terrain_z + 0.18),
        color=(0.46, 0.42, 0.36),
        collision=False,
    )

    tree_specs = [
        (-0.84, -0.96, 0.18, 0.09),
        (-0.88, -0.30, 0.22, 0.11),
        (-0.86, 0.50, 0.20, 0.10),
        (0.86, -0.72, 0.16, 0.08),
        (0.90, 0.62, 0.18, 0.09),
    ]
    for idx, (x, y, trunk_h, leaf_r) in enumerate(tree_specs):
        base_z = scene_cfg.lower_terrain_z + 0.006
        _cylinder(
            f"/World/CliffRoad/TreeTrunk{idx:02d}",
            radius=0.025,
            height=trunk_h,
            translation=(x, y, base_z + 0.5 * trunk_h),
            color=(0.35, 0.20, 0.11),
            collision=False,
        )
        _cone(
            f"/World/CliffRoad/TreeTop{idx:02d}",
            radius=leaf_r,
            height=0.24,
            translation=(x, y, base_z + trunk_h + 0.12),
            color=(0.05, 0.30, 0.13),
            collision=False,
        )

    shrub_specs = [
        (-0.92, -1.06, (0.07, 0.32, 0.13)),
        (-1.06, -0.48, (0.08, 0.38, 0.16)),
        (-0.88, 0.88, (0.06, 0.30, 0.12)),
        (0.92, -0.18, (0.08, 0.36, 0.14)),
        (1.08, 0.72, (0.10, 0.42, 0.17)),
    ]
    for idx, (x, y, color) in enumerate(shrub_specs):
        _cone(
            f"/World/CliffRoad/Shrub{idx:02d}",
            radius=0.055,
            height=0.090,
            translation=(x, y, scene_cfg.lower_terrain_z + 0.055),
            color=color,
            collision=False,
        )


def start_pose(scene_cfg: CliffRoadSceneCfg | None = None) -> tuple[tuple[float, float, float], float]:
    """Start at the bottom of the guide line, facing uphill in +Y."""
    cfg = scene_cfg or CliffRoadSceneCfg()
    return (0.0, -0.5 * cfg.road_length + 0.12, cfg.cliff_height + cfg.start_height), 1.5707963267948966
