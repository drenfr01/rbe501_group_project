"""Record MP4 videos of the pick-and-place results: controllers side by side, a mass sweep, and live joint torque.

HOW TO RUN (from the project folder; needs ffmpeg and an EGL-capable GPU driver):
    uv run python pick_place/scripts/record_videos.py                 # all videos (~1 min)
    uv run python pick_place/scripts/record_videos.py --only compare_1kg torque_1kg

Every clip shows the same start and goal (an evaluation seed), seen from the side so vertical sag is visible, with
a top-down inset of the goal with the arm hidden. Markers drawn into the scene:
    green disk   the 2 cm place tolerance around the goal
    red sphere   where the cube center would be with no sag: the pinch point at the IK target θ* (while carrying)
The gap between the red sphere and the cube is the servo sag K_p^-1 τ in Cartesian space.

OUTPUT (checkpoints/pick-place/videos/):
    compare_1kg.mp4     scripted | feedforward | fixed SAC | random SAC, 1.0 kg, episode 20008
    compare_1500g.mp4   the same at 1.5 kg, episode 20016
    limits_1500g.mp4    1.5 kg, episode 20018: both SAC policies miss, the feedforward still places it
    mass_sweep.mp4      scripted controller at 0.05 / 0.5 / 1.0 / 1.5 kg, episode 20008
    torque_1kg.mp4      scripted and feedforward at 1.0 kg with live shoulder/elbow torque and sag charts
    torque_rl_1kg.mp4   the same charts for the fixed-mass and random-mass SAC policies
    torque_rl_1kg_seed4.mp4  the same episode with training seed 4: fixed places, random misses
Trained policies use seed 0 of each v2 run.
"""

import argparse
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from rbe_501_so101.config import ObjectMassConfig, SimConfig  # noqa: E402
from rbe_501_so101.envs.model_indices import descendant_body_ids  # noqa: E402
from rbe_501_so101.pick_place.env import (  # noqa: E402
    CHECKPOINTS,
    PlaceConfig,
    PlaceRun,
    ScriptedPickPlace,
    load_policy,
    make_env,
    trained_runs,
)
from rbe_501_so101.pick_place.torque import ELBOW, HOLDING_PHASES, LIFT, RATED_NM, STALL_NM, ArmStatics  # noqa: E402
from rbe_501_so101.pick_place.torque import FeedforwardPickPlace  # noqa: E402

OUT = CHECKPOINTS / "videos"
WIDTH, HEIGHT = 640, 480
INSET = (200, 150)
CHART_WIDTH = 480
FPS = 30
HOLD_FRAMES = 45
"""Extra copies of the last frame, so the result banner stays on screen for 1.5 s."""
POLICY_SEED = 0
TOLERANCE_M = PlaceConfig().place_tolerance_m
CAMERA = dict(distance=0.62, elevation=-14.0, along=0.55, height_m=0.06, side_deg=90.0)
"""Side camera: looks at a point `along` of the way from the base to the goal, from 90° off that line."""
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf")
LABELS = {
    "scripted": "Scripted IK",
    "feedforward": "IK + gravity feedforward (known mass)",
    "ff_payload": "IK + payload feedforward",
    "fixed": "Residual SAC, fixed 1.0 kg",
    "random": "Residual SAC, random 0.05-1.0 kg",
}
COMPARE = ("scripted", "feedforward", "fixed", "random")
GREEN, RED, WHITE, AMBER, BLUE = (40, 170, 80), (215, 50, 40), (255, 255, 255), (230, 120, 20), (40, 100, 200)


@dataclass
class Clip:
    """Rendered frames of one episode plus what the overlays and charts need."""

    name: str
    mass_kg: float
    frames: list[np.ndarray] = field(default_factory=list)
    torque_nm: list[np.ndarray] = field(default_factory=list)
    sag_deg: list[np.ndarray] = field(default_factory=list)
    policy_seed: int | None = None
    success: bool = False
    error_m: float = float("nan")


# ================================================================ controllers and rendering


def make_controller(name: str, mass_kg: float, policy_seed: int = POLICY_SEED) -> tuple:
    """(env, policy, scripted controller) for a controller name at a fixed cube mass."""
    mass = ObjectMassConfig(mode="fixed", fixed_kg=mass_kg)
    if name in ("fixed", "random"):
        model_dir = next(d for d in trained_runs("v2") if name in d.name)
        run = PlaceRun.load(model_dir)
        env = make_env(mass, run)
        return env, load_policy(model_dir / f"seed_{policy_seed}", run), env._controller
    env = make_env(mass)
    controller = {"scripted": ScriptedPickPlace.for_env, "feedforward": FeedforwardPickPlace.for_env,
                  "ff_payload": lambda e: FeedforwardPickPlace.for_env(e, payload_only=True)}[name](env)
    return env, controller, controller


class View:
    """Side camera with a top-down goal inset and the tolerance / target markers."""

    def __init__(self, env) -> None:
        u = env.unwrapped
        self._backend, self._p_goal = u.backend, u.p_goal.copy()
        self._statics = ArmStatics(env)
        self._main = mujoco.Renderer(self._backend.model, HEIGHT, WIDTH)
        self._inset = mujoco.Renderer(self._backend.model, INSET[1], INSET[0])
        self._side, self._top = self._side_camera(), self._top_camera()
        self._robot = descendant_body_ids(self._backend.model, SimConfig().robot_root_body)

    def frame(self, controller: ScriptedPickPlace) -> np.ndarray:
        """Side view with markers, and the top view pasted bottom-right."""
        target = self._target(controller)
        main = self._render(self._main, self._side, target)
        inset = self._render(self._inset, self._top, target, hide_robot=True)
        w, h = INSET
        main[-h - 8:-8, -w - 8:-8] = inset
        main[-h - 10:-6, -w - 10:-w - 8] = main[-h - 10:-h - 8, -w - 10:-6] = 255
        return main

    def close(self) -> None:
        """Free the GL contexts."""
        self._main.close()
        self._inset.close()

    def _render(self, renderer: mujoco.Renderer, camera: mujoco.MjvCamera, target: np.ndarray | None,
                hide_robot: bool = False) -> np.ndarray:
        """Render one camera and draw the markers into its scene (optionally without the arm)."""
        renderer.update_scene(self._backend.data, camera=camera)
        if hide_robot:
            self._hide_robot(renderer.scene)
        table_z = self._backend.rest_z - self._backend._object_cfg.side_m / 2 + 0.0005
        _add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [TOLERANCE_M, 0.0005, 0],
                  [*self._p_goal[:2], table_z], [0.2, 0.85, 0.3, 0.45])
        if target is not None:
            _add_geom(renderer.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], target, [0.9, 0.15, 0.1, 0.8])
        return renderer.render().copy()

    def _hide_robot(self, scene: mujoco.MjvScene) -> None:
        """Make every arm geom invisible so the top view sees the cube and the goal."""
        body_of = self._backend.model.geom_bodyid
        for geom in scene.geoms[: scene.ngeom]:
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and body_of[geom.objid] in self._robot:
                geom.rgba[3] = 0.0

    def _target(self, controller: ScriptedPickPlace) -> np.ndarray | None:
        """Pinch point at the IK target θ* while the cube is carried; None otherwise."""
        if controller.phase not in HOLDING_PHASES or controller._target is None:
            return None
        q = np.append(controller._target.q_arm, self._backend.data.qpos[self._backend.ids.qpos_adr[-1]])
        self._statics.loads(q)
        return self._statics.pinch_point()

    def _side_camera(self) -> mujoco.MjvCamera:
        """Camera 90° off the base-to-goal line, looking at the far part of the reach."""
        axis, goal = self._backend.pan_axis_xy, self._p_goal[:2]
        phi = np.degrees(np.arctan2(*(goal - axis)[::-1]))
        look = axis + CAMERA["along"] * (goal - axis)
        return _camera([*look, self._backend.rest_z + CAMERA["height_m"]], CAMERA["distance"],
                       phi + CAMERA["side_deg"], CAMERA["elevation"])

    def _top_camera(self) -> mujoco.MjvCamera:
        """Straight down onto the goal, about 8 cm across."""
        return _camera([*self._p_goal[:2], self._backend.rest_z], 0.13, 90.0, -89.9)


def _camera(lookat: list[float], distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    """Free camera with the given orbit parameters."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, distance, azimuth, elevation
    return cam


def _add_geom(scene: mujoco.MjvScene, kind: int, size: list[float], pos, rgba: list[float]) -> None:
    """Append one decorative geom (no collision) to a rendered scene."""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


# ================================================================ rollouts and overlays


def rollout(name: str, mass_kg: float, seed: int, policy_seed: int = POLICY_SEED) -> Clip:
    """Play one episode, rendering and labelling every control step."""
    env, policy, controller = make_controller(name, mass_kg, policy_seed)
    obs, _ = env.reset(seed=seed)
    getattr(policy, "reset", lambda: None)()
    trained = name in ("fixed", "random")
    view, clip, truncated, info = View(env), Clip(name, mass_kg, policy_seed=policy_seed if trained else None), False, {}
    backend = env.unwrapped.backend
    while not truncated:
        obs, _, _, truncated, info = env.step(policy(obs))
        s = backend.grasp_state().robot
        clip.torque_nm.append(backend.data.qfrc_actuator[backend.ids.dof_adr[[LIFT, ELBOW]]].copy())
        clip.sag_deg.append(np.degrees((s.q_cmd - s.qpos)[[LIFT, ELBOW]]))
        clip.frames.append(label(view.frame(controller), clip, str(controller.phase), info["goal_distance_m"]))
    clip.success, clip.error_m = bool(info["is_success"]), float(info["goal_distance_m"])
    clip.frames += [banner(clip.frames[-1], clip)] * HOLD_FRAMES
    view.close()
    env.close()
    return clip


def _font(size: int) -> ImageFont.FreeTypeFont:
    """DejaVu Sans if present, else PIL's built-in font."""
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def label(frame: np.ndarray, clip: Clip, phase: str, goal_distance_m: float) -> np.ndarray:
    """Controller and mass on top, phase and cube-goal distance at the bottom."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, WIDTH, 30), fill=(20, 24, 30))
    seed = "" if clip.policy_seed in (None, POLICY_SEED) else f"  (training seed {clip.policy_seed})"
    draw.text((10, 5), LABELS[clip.name] + seed, fill=WHITE, font=_font(18))
    draw.text((WIDTH - 10, 5), f"{clip.mass_kg:g} kg", fill=AMBER, font=_font(18), anchor="ra")
    color = GREEN if goal_distance_m <= TOLERANCE_M else WHITE
    draw.rectangle((0, HEIGHT - 28, WIDTH - INSET[0] - 14, HEIGHT), fill=(20, 24, 30))
    draw.text((10, HEIGHT - 24), f"{phase:<8}  cube to goal {100 * goal_distance_m:4.1f} cm", fill=color, font=_font(16))
    return np.asarray(img)


def banner(frame: np.ndarray, clip: Clip) -> np.ndarray:
    """Result banner across the middle of the last frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    text = (f"PLACED  {100 * clip.error_m:.1f} cm" if clip.success
            else f"MISSED  {100 * clip.error_m:.1f} cm (tolerance {100 * TOLERANCE_M:.0f} cm)")
    draw.rectangle((0, 34, WIDTH, 80), fill=GREEN if clip.success else RED)
    draw.text((WIDTH // 2, 57), text, fill=WHITE, font=_font(24), anchor="mm")
    return np.asarray(img)


def tile(frames: list[np.ndarray], columns: int) -> np.ndarray:
    """Frames of equal size in a grid, filled row by row."""
    rows = [np.concatenate(frames[i:i + columns], axis=1) for i in range(0, len(frames), columns)]
    return np.concatenate(rows, axis=0)


# ================================================================ torque chart


def torque_chart(clip: Clip, step: int) -> np.ndarray:
    """Shoulder/elbow torque (top) and sag (bottom) up to `step`, beside the video frame."""
    img = Image.new("RGB", (CHART_WIDTH, HEIGHT), (250, 251, 253))
    draw = ImageDraw.Draw(img)
    n = len(clip.torque_nm)
    k = min(step, n - 1) + 1
    torque, sag = np.asarray(clip.torque_nm[:k]), np.asarray(clip.sag_deg[:k])
    _panel(draw, (50, 30, CHART_WIDTH - 15, 225), n, torque, (-3.2, 3.2), "joint torque (N·m)",
           [(STALL_NM, "stall"), (-STALL_NM, ""), (RATED_NM, "rated"), (-RATED_NM, "")])
    _panel(draw, (50, 265, CHART_WIDTH - 15, HEIGHT - 30), n, sag, (-8.0, 1.0), "sag  θ_cmd − θ (deg)", [(0.0, "")])
    draw.text((60, HEIGHT - 22), "shoulder_lift", fill=BLUE, font=_font(14))
    draw.text((180, HEIGHT - 22), "elbow_flex", fill=AMBER, font=_font(14))
    return np.asarray(img)


def _panel(draw: ImageDraw.ImageDraw, box: tuple, n: int, series: np.ndarray, ylim: tuple, title: str,
           guides: list[tuple[float, str]]) -> None:
    """One chart panel: frame, guide lines, and one polyline per column of `series`."""
    x0, y0, x1, y1 = box
    to_y = lambda v: y1 - (np.clip(v, *ylim) - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)  # noqa: E731
    draw.rectangle(box, outline=(170, 176, 186))
    draw.text((x0, y0 - 20), title, fill=(30, 34, 40), font=_font(15))
    for value, name in guides:
        draw.line((x0, to_y(value), x1, to_y(value)), fill=(160, 160, 160), width=1)
        draw.text((x1 - 4, to_y(value) - 15), name, fill=(120, 120, 120), font=_font(12), anchor="ra")
    for tick in (ylim[0], ylim[1]):
        draw.text((x0 - 6, to_y(tick)), f"{tick:g}", fill=(90, 90, 90), font=_font(12), anchor="rm")
    xs = x0 + np.arange(len(series)) / max(n - 1, 1) * (x1 - x0)
    for col, color in zip(series.T, (BLUE, AMBER)):
        if len(col) > 1:
            draw.line(list(zip(xs, to_y(col))), fill=color, width=2)


# ================================================================ videos


def write_mp4(frames: list[np.ndarray], path: Path) -> None:
    """Encode RGB frames to H.264 with the system ffmpeg."""
    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
           "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames), check=True)
    print("saved", path)


def grid_video(clips: list[Clip], columns: int = 2) -> list[np.ndarray]:
    """Clips played side by side, each holding its last frame until the longest ends."""
    n = max(len(c.frames) for c in clips)
    return [tile([c.frames[min(i, len(c.frames) - 1)] for c in clips], columns) for i in range(n)]


def torque_video(clips: list[Clip]) -> list[np.ndarray]:
    """One row per clip: the video frame and its live torque/sag chart."""
    n = max(len(c.frames) for c in clips)
    rows = lambda i: [np.concatenate([c.frames[min(i, len(c.frames) - 1)], torque_chart(c, i)], axis=1)  # noqa: E731
                      for c in clips]
    return [np.concatenate(rows(i), axis=0) for i in range(n)]


VIDEOS = {
    "compare_1kg": lambda: grid_video([rollout(c, 1.0, 20_008) for c in COMPARE]),
    "compare_1500g": lambda: grid_video([rollout(c, 1.5, 20_016) for c in COMPARE]),
    "limits_1500g": lambda: grid_video([rollout(c, 1.5, 20_018) for c in COMPARE]),
    "mass_sweep": lambda: grid_video([rollout("scripted", m, 20_008) for m in (0.05, 0.5, 1.0, 1.5)]),
    "torque_1kg": lambda: torque_video([rollout(c, 1.0, 20_008) for c in ("scripted", "feedforward")]),
    "torque_rl_1kg": lambda: torque_video([rollout(c, 1.0, 20_008) for c in ("fixed", "random")]),
    "torque_rl_1kg_seed4": lambda: torque_video([rollout(c, 1.0, 20_008, policy_seed=4) for c in ("fixed", "random")]),
}


def main() -> None:
    """Render the requested videos into --out."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(VIDEOS), default=list(VIDEOS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        write_mp4(VIDEOS[name](), args.out / f"{name}.mp4")


if __name__ == "__main__":
    main()
