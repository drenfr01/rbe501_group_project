"""Joint-torque analysis of pick-and-place: where the torque goes, how much the arm sags, what the RL policies learned.

HOW TO RUN (from the project folder):
    uv run python pick_place/scripts/torque_analysis.py                    # full sweep (~10 min on 8 cores), figures + report
    uv run python pick_place/scripts/torque_analysis.py --episodes 5       # quick look
    uv run python pick_place/scripts/torque_analysis.py --plot-only        # redraw from the saved sweep

Controllers, each at every mass with the same start/goal seeds as `compare`:
    scripted        IK waypoints, no dynamics model
    feedforward     scripted + K_p^-1 (g + m b) on every target, true mass known (model-based gravity compensation)
    ff_payload      scripted + K_p^-1 m b only (the script was tuned with the arm's own-weight sag)
    fixed, random   the trained v2 residual SAC policies (all seeds, episodes spread over them)

OUTPUT (checkpoints/pick-place/torque/): sweep.pkl, report.txt, and
    fig1_torque_breakdown.png   τ_act and its terms over one episode, light vs heavy cube
    fig2_model_validation.png   payload model vs measured contact torque, predicted vs measured sag, EoM residual
    fig3_sag_and_drop.png       joint sag and cube drop (arm vs slip) vs mass
    fig4_capacity.png           largest static payload over the workspace, stall and rated torque
    fig5_load_vs_mass.png       peak torque (% of stall), saturation, heat, work vs mass
    fig6_correction.png         correction θ_cmd - θ* at placement vs the sag it should cancel; sideways offset
    fig7_success.png            success rate and placement error vs mass for all five controllers
    fig8_residual_by_phase.png  where in the episode the RL policies act (mean |residual| per phase and joint)
"""

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rbe_501_so101.config import ObjectMassConfig  # noqa: E402
from rbe_501_so101.pick_place.env import (  # noqa: E402
    CHECKPOINTS,
    EVAL_MASSES_KG,
    EVAL_SEED_OFFSET,
    PickPlaceConfig,
    PlaceConfig,
    PlaceRun,
    ScriptedPickPlace,
    load_policy,
    make_env,
    trained_runs,
)
from rbe_501_so101.pick_place.torque import (  # noqa: E402
    ELBOW,
    JOINT_NAMES,
    LIFT,
    RATED_NM,
    STALL_NM,
    WRIST_FLEX,
    FeedforwardPickPlace,
    Trace,
    TraceSummary,
    capacity_map,
    record_episode,
)

OUT = CHECKPOINTS / "torque"
CONTROLLERS = ("scripted", "feedforward", "ff_payload", "fixed", "random")
LOADED = (LIFT, ELBOW, WRIST_FLEX)
EXAMPLE_MASSES = (0.05, 1.0)
"""Masses whose first episode is kept whole for the time-series figure."""
CAPACITY_RADII = tuple(np.round(np.arange(0.10, 0.2801, 0.01), 3))
CAPACITY_HEIGHTS = tuple(np.round(np.arange(0.0, 0.1201, 0.01), 3))
CARRY_HEIGHT_M = 0.07
RESIDUAL_CAP_DEG = np.rad2deg(PlaceRun(ObjectMassConfig()).residual_scale * PickPlaceConfig().task.delta_q_max_rad)
"""Largest steady command offset a v2 residual can hold: the script pulls the command back to θ* every step."""
PHASES = ("approach", "descend", "close", "lift", "carry", "lower", "release", "retreat")
TUNGSTEN_KG = 0.315
"""1-inch tungsten cube, the heaviest real test object."""
PHASE_COLORS = {"approach": "#f2f2f2", "descend": "#e3ecf7", "close": "#fde9d9", "lift": "#e6f2e6",
                "carry": "#fff6cc", "lower": "#efe3f5", "release": "#fde9d9", "retreat": "#f2f2f2"}
TERM_STYLE = {"gravity": ("arm gravity g(θ)", "tab:blue"), "contact": ("contact / payload −JᵀF", "tab:red"),
              "inertial": ("inertial M θ̈", "tab:green"), "coriolis": ("Coriolis c(θ,θ̇)", "tab:purple")}
COLORS = {"scripted": "black", "feedforward": "tab:blue", "ff_payload": "tab:cyan", "fixed": "tab:orange",
          "random": "tab:green"}


@dataclass(frozen=True)
class Job:
    """One controller at one mass."""

    controller: str
    mass_kg: float
    episodes: int


@dataclass(frozen=True)
class JobResult:
    """Episode summaries of a job, plus its first full trace for the example masses."""

    job: Job
    summaries: list[TraceSummary]
    example: Trace | None
    max_residual_nm: float


# ================================================================ sweep


def _policies(job: Job) -> list[tuple]:
    """(env, policy, scripted controller) per seed for a controller at a mass."""
    mass = ObjectMassConfig(mode="fixed", fixed_kg=job.mass_kg)
    if job.controller in ("fixed", "random"):
        model_dir = next(d for d in trained_runs("v2") if job.controller in d.name)
        run = PlaceRun.load(model_dir)
        seeds = sorted(model_dir.glob("seed_*/model.zip"))
        envs = [make_env(mass, run) for _ in seeds]
        return [(env, load_policy(s.parent, run), env._controller) for env, s in zip(envs, seeds)]
    env = make_env(mass)
    controller = {"scripted": lambda: ScriptedPickPlace.for_env(env),
                  "feedforward": lambda: FeedforwardPickPlace.for_env(env),
                  "ff_payload": lambda: FeedforwardPickPlace.for_env(env, payload_only=True)}[job.controller]()
    return [(env, controller, controller)]


def run_job(job: Job) -> JobResult:
    """Record `job.episodes` episodes (seeds rotate across trained models) and summarize each."""
    setups = _policies(job)
    summaries, example, worst = [], None, 0.0
    for i in range(job.episodes):
        env, policy, controller = setups[i % len(setups)]
        trace = record_episode(env, policy, controller, EVAL_SEED_OFFSET + i, job.controller)
        summaries.append(trace.summary())
        worst = max(worst, float(np.abs(trace.terms.residual).max()))
        if i == 0 and job.mass_kg in EXAMPLE_MASSES:
            example = trace
    for env, *_ in setups:
        env.close()
    return JobResult(job, summaries, example, worst)


def sweep(masses: tuple[float, ...], episodes: int, workers: int) -> dict:
    """Run every controller at every mass in parallel; add the capacity maps."""
    jobs = [Job(c, m, episodes) for c in CONTROLLERS for m in masses]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(run_job, jobs))
    env = make_env(ObjectMassConfig(mode="fixed", fixed_kg=0.05))
    env.reset(seed=EVAL_SEED_OFFSET)
    caps = {limit: capacity_map(env, CAPACITY_RADII, CAPACITY_HEIGHTS, limit) for limit in (STALL_NM, RATED_NM)}
    env.close()
    return {"masses": masses, "results": results, "capacity": caps}


# ================================================================ statistics


def field(data: dict, controller: str, name: str, joint: int | None = None) -> np.ndarray:
    """(n_masses, episodes) array of a summary field for one controller (one joint if per-joint)."""
    rows = []
    for m in data["masses"]:
        result = next(r for r in data["results"] if r.job.controller == controller and r.job.mass_kg == m)
        values = [getattr(s, name) for s in result.summaries]
        rows.append([v[joint] for v in values] if joint is not None else values)
    return np.asarray(rows, dtype=float)


def residual_grid(data: dict, controller: str, mass: float) -> np.ndarray:
    """(phase, joint) mean |residual| (rad) over the episodes of a controller at a mass."""
    result = next(r for r in data["results"] if r.job.controller == controller and r.job.mass_kg == mass)
    rows = [[s.residual_by_phase.get(p, np.full(5, np.nan)) for p in PHASES] for s in result.summaries]
    return np.nanmean(np.asarray(rows), axis=0)


def mean(x: np.ndarray) -> np.ndarray:
    """Per-mass mean over episodes, ignoring episodes where the window was empty."""
    return np.nanmean(x, axis=1)


def example(data: dict, controller: str, mass: float) -> Trace:
    """The kept full trace of a controller at an example mass."""
    return next(r.example for r in data["results"] if r.job.controller == controller and r.job.mass_kg == mass)


def saturated(data: dict, controller: str) -> np.ndarray:
    """(n_masses, episodes) True where any loaded joint hit the stall cap during the episode."""
    return np.any([field(data, controller, "saturated_frac", j) > 0 for j in LOADED], axis=0)


def r_squared(model: np.ndarray, measured: np.ndarray) -> float:
    """Coefficient of determination of `model` as a predictor of `measured`."""
    ok = np.isfinite(model) & np.isfinite(measured)
    residual = ((measured[ok] - model[ok]) ** 2).sum()
    return float(1 - residual / ((measured[ok] - measured[ok].mean()) ** 2).sum())


# ================================================================ figures


def _shade_phases(ax: plt.Axes, trace: Trace) -> None:
    """Background color per scripted phase."""
    t, phases = trace.t_s, trace.phase
    starts = np.flatnonzero(np.r_[True, phases[1:] != phases[:-1]])
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(t) - 1
        ax.axvspan(t[s] - trace.dt_s, t[end] - trace.dt_s, color=PHASE_COLORS.get(phases[s], "white"), lw=0)


def fig_breakdown(data: dict, path: Path) -> None:
    """τ_act and its terms for the loaded joints, light vs heavy cube, scripted controller."""
    fig, axes = plt.subplots(len(LOADED), 2, figsize=(14, 9), sharex=True)
    for col, mass in enumerate(EXAMPLE_MASSES):
        trace = example(data, "scripted", mass)
        for row, j in enumerate(LOADED):
            ax = axes[row, col]
            _shade_phases(ax, trace)
            for name, (label, color) in TERM_STYLE.items():
                ax.plot(trace.t_s, getattr(trace.terms, name)[:, j], color=color, lw=1, label=label)
            ax.plot(trace.t_s, trace.terms.actuator[:, j], "k", lw=2, label="actuator τ_act")
            for lim, style in ((STALL_NM, "--"), (RATED_NM, ":")):
                ax.axhline(lim, color="gray", ls=style), ax.axhline(-lim, color="gray", ls=style)
            ax.set(ylabel=f"{JOINT_NAMES[j]} (N·m)", ylim=(-3.2, 3.2))
        axes[0, col].set_title(f"scripted, {mass:g} kg cube (dashed: stall {STALL_NM}, dotted: rated {RATED_NM})")
        axes[-1, col].set_xlabel("time (s); shading = phase: approach, descend, close, lift, carry, lower, release")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    _save(fig, path)


def fig_validation(data: dict, path: Path) -> None:
    """Payload model vs measured contact torque, predicted vs measured sag, and the EoM residual."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 5))
    ok = ~saturated(data, "scripted")
    for j in LOADED:
        model, meas = field(data, "scripted", "payload_model_nm", j), field(data, "scripted", "contact_nm", j)
        a.scatter(model[ok], meas[ok], s=8, label=f"{JOINT_NAMES[j]} R²={r_squared(model[ok], meas[ok]):.4f}")
        a.scatter(model[~ok], meas[~ok], s=14, marker="x", color="gray")
        pred, sag = field(data, "scripted", "sag_predicted", j), field(data, "scripted", "sag_measured", j)
        b.scatter(np.rad2deg(pred[ok]), np.rad2deg(sag[ok]), s=8,
                  label=f"{JOINT_NAMES[j]} R²={r_squared(pred[ok], sag[ok]):.4f}")
        b.scatter(np.rad2deg(pred[~ok]), np.rad2deg(sag[~ok]), s=14, marker="x", color="gray")
    b.scatter([], [], marker="x", color="gray", label="episodes that hit the stall cap")
    for ax, unit in ((a, "N·m"), (b, "deg")):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.legend(fontsize=8), ax.grid(alpha=0.3)
    a.set(xlabel="model m Jᵥᵀ(−g_s) (N·m)", ylabel="measured −Jᵀf_contact (N·m)", title="payload torque, hold window")
    b.set(xlabel="predicted K_p⁻¹(g + m b) (deg)", ylabel="measured θ_cmd − θ (deg)", title="servo sag, hold window")
    worst = [r.max_residual_nm for r in data["results"]]
    c.hist(np.log10(np.maximum(worst, 1e-12)), bins=20, color="gray")
    c.set(xlabel="log10 max |Mθ̈ + c + g + τ_c − τ_act| per job (N·m)", ylabel="jobs",
          title="equation-of-motion residual (all episodes)")
    _save(fig, path)


def fig_sag_drop(data: dict, path: Path) -> None:
    """Joint sag (measured ± std, predicted) and cube drop split into arm sag and slip, scripted controller."""
    masses = np.asarray(data["masses"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
    for j in LOADED:
        sag = np.rad2deg(field(data, "scripted", "sag_measured", j))
        line = a.errorbar(masses, np.nanmean(sag, 1), yerr=np.nanstd(sag, 1), marker="o", capsize=3,
                          label=f"{JOINT_NAMES[j]} measured")
        pred = np.rad2deg(mean(field(data, "scripted", "sag_predicted", j)))
        a.plot(masses, pred, "--", color=line[0].get_color(), label=f"{JOINT_NAMES[j]} K_p⁻¹τ")
    a.set(xlabel="cube mass (kg)", ylabel="θ_cmd − θ (deg)", title="servo sag at the end of carry (scripted)")
    for name, label in (("drop_measured_m", "cube drop (total)"), ("drop_arm_m", "pinch-point sag (arm)"),
                        ("slip_m", "slip in the grip"), ("drop_predicted_m", "predicted Jᵥ,z Δθ")):
        b.plot(masses, 1000 * mean(field(data, "scripted", name)), marker="o", label=label)
    b.axhline(1000 * PlaceConfig().place_tolerance_m, color="red", ls="--", label="place tolerance")
    b.set(xlabel="cube mass (kg)", ylabel="mm below the carry height", title="where the cube hangs (scripted)")
    for ax in (a, b):
        ax.grid(alpha=0.3), ax.legend(fontsize=8)
    _save(fig, path)


def fig_capacity(data: dict, path: Path) -> None:
    """m_max(r) at carry height for stall and rated torque, and the stall map over (r, h) with the limiting joint."""
    caps = data["capacity"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.5))
    h = int(np.argmin(np.abs(caps[STALL_NM].heights_m - CARRY_HEIGHT_M)))
    for limit, cmap in caps.items():
        a.plot(100 * cmap.radii_m, cmap.m_max_kg[h], marker="o", label=f"limit {limit} N·m")
    for m in data["masses"]:
        a.axhline(m, color="gray", lw=0.5)
    a.axhline(TUNGSTEN_KG, color="tab:brown", ls="--", label="1-inch tungsten cube 0.315 kg")
    a.axvspan(10, 24, color="tab:green", alpha=0.08, label="task workspace")
    a.set(xlabel="reach r (cm)", ylabel="largest static payload (kg)", ylim=(0, 2.5),
          title=f"static payload limit at carry height {100 * CARRY_HEIGHT_M:.0f} cm (gray: sim masses)")
    stall = caps[STALL_NM]
    im = b.pcolormesh(100 * stall.radii_m, 100 * stall.heights_m, stall.m_max_kg, shading="nearest", cmap="viridis",
                      vmin=0, vmax=2.5)
    for (i, j), joint in np.ndenumerate(stall.joint):
        b.text(100 * stall.radii_m[j], 100 * stall.heights_m[i], JOINT_NAMES[joint][0].upper(), ha="center",
               va="center", fontsize=6, color="white")
    fig.colorbar(im, ax=b, label="m_max at stall (kg)")
    b.set(xlabel="reach r (cm)", ylabel="pinch height above cube rest (cm)",
          title="stall-limited payload; letter = limiting joint (S: shoulder_lift, E: elbow, W: wrist)")
    a.grid(alpha=0.3), a.legend(fontsize=8)
    _save(fig, path)


def fig_load(data: dict, path: Path) -> None:
    """Peak torque (% of stall), saturated steps, heat, and work per loaded joint vs mass (scripted)."""
    masses = np.asarray(data["masses"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    specs = (("peak_nm", 100 / STALL_NM, "peak |τ_act| (% of stall)"),
             ("saturated_frac", 100, "steps at the stall cap (%)"),
             ("heat_n2m2s", 1, "Σ τ² Δt (N²m²s) ∝ copper loss"), ("work_j", 1, "mechanical work Σ|τ θ̇| Δt (J)"))
    for ax, (name, scale, label) in zip(axes.flat, specs):
        for j in LOADED:
            values = scale * field(data, "scripted", name, j)
            ax.errorbar(masses, values.mean(1), yerr=values.std(1), marker="o", capsize=3, label=JOINT_NAMES[j])
        ax.set(xlabel="cube mass (kg)", ylabel=label), ax.grid(alpha=0.3), ax.legend(fontsize=8)
    axes[0, 0].axhline(100 * RATED_NM / STALL_NM, color="gray", ls=":", label="rated")
    fig.suptitle("Joint load per episode, scripted controller (mean ± std over episodes)")
    _save(fig, path)


def fig_correction(data: dict, path: Path) -> None:
    """θ_cmd − θ* just before release per controller vs the sag K_p⁻¹τ; and the sideways cube offset there."""
    masses = np.asarray(data["masses"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, j in zip(axes[:2], (LIFT, ELBOW)):
        ax.plot(masses, np.rad2deg(mean(field(data, "scripted", "place_sag_predicted", j))), "k--", lw=2,
                label="needed: K_p⁻¹(g + m b)")
        ax.axhspan(-RESIDUAL_CAP_DEG, RESIDUAL_CAP_DEG, color="tab:orange", alpha=0.12,
                   label=f"reach of a v2 residual (±{RESIDUAL_CAP_DEG:.1f}°)")
        for c in CONTROLLERS:
            ax.plot(masses, np.rad2deg(mean(field(data, c, "place_correction", j))), marker="o", color=COLORS[c],
                    label=c)
        ax.set(xlabel="cube mass (kg)", ylabel="θ_cmd − θ* (deg)", title=f"{JOINT_NAMES[j]}: offset at placement")
    for c in CONTROLLERS:
        axes[2].plot(masses, 1000 * mean(field(data, c, "place_offset_m")), marker="o", color=COLORS[c], label=c)
    axes[2].plot(masses, 1000 * mean(field(data, "scripted", "place_shift_predicted_m")), "k--",
                 label="‖Jᵥ,xy Δθ‖ sag shift (model)")
    axes[2].axhline(1000 * PlaceConfig().place_tolerance_m, color="red", ls=":", label="tolerance")
    axes[2].set(xlabel="cube mass (kg)", ylabel="cube − goal, horizontal (mm)", title="offset just before release")
    for ax in axes:
        ax.grid(alpha=0.3), ax.legend(fontsize=7)
    _save(fig, path)


def fig_residual(data: dict, path: Path) -> None:
    """Mean |residual| per phase and joint for both trained policies, light vs heavy cube."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, c in enumerate(("fixed", "random")):
        for col, m in enumerate(EXAMPLE_MASSES):
            grid = np.rad2deg(residual_grid(data, c, m))
            im = axes[row, col].imshow(grid, cmap="magma", vmin=0, vmax=RESIDUAL_CAP_DEG, aspect="auto")
            for (i, j), v in np.ndenumerate(grid):
                axes[row, col].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                    color="white" if v < RESIDUAL_CAP_DEG / 2 else "black")
            axes[row, col].set(xticks=range(5), xticklabels=[n.replace("_", "\n") for n in JOINT_NAMES],
                               yticks=range(len(PHASES)), yticklabels=PHASES, title=f"{c}, {m:g} kg")
    fig.colorbar(im, ax=axes, label=f"mean |residual| per step (deg); max {RESIDUAL_CAP_DEG:.2f}")
    fig.suptitle("Where the learned residual acts")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


def fig_success(data: dict, path: Path) -> None:
    """Success rate and placement error vs mass for every controller."""
    masses = np.asarray(data["masses"])
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for c in CONTROLLERS:
        a.plot(masses, 100 * mean(field(data, c, "success")), marker="o", color=COLORS[c], label=c)
        b.plot(masses, 100 * mean(field(data, c, "error_m")), marker="o", color=COLORS[c], label=c)
    b.axhline(100 * PlaceConfig().place_tolerance_m, color="red", ls="--", label="tolerance")
    a.set(ylabel="success rate (%)", title="Pick-and-place: model-based compensation vs learned residual")
    b.set(xlabel="cube mass (kg)", ylabel="mean placement error (cm)")
    for ax in (a, b):
        ax.grid(alpha=0.3), ax.legend(fontsize=8)
    _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> None:
    """Save and close a figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)


# ================================================================ report


def report(data: dict) -> str:
    """Plain-text tables of everything in the figures."""
    return "\n\n".join([_validation_table(data), _load_table(data), _sag_table(data), _capacity_table(data),
                        _controller_table(data), _residual_table(data)])


def _validation_table(data: dict) -> str:
    """How well the dynamics model explains the simulator."""
    sat = saturated(data, "scripted")
    rows = ["MODEL VALIDATION (scripted, hold window)",
            f"max |EoM residual| over all episodes: {max(r.max_residual_nm for r in data['results']):.1e} N·m",
            f"episodes that hit the stall cap: {sat.sum()} of {sat.size} (" + ", ".join(
                f"{m:g} kg: {n}" for m, n in zip(data["masses"], sat.sum(axis=1)) if n) + ")",
            "joint          episodes      payload R²  max|err| N·m   sag R²    max|err| deg"]
    for j in LOADED:
        model, meas = field(data, "scripted", "payload_model_nm", j), field(data, "scripted", "contact_nm", j)
        pred, sag = field(data, "scripted", "sag_predicted", j), field(data, "scripted", "sag_measured", j)
        for name, keep in (("unsaturated", ~sat), ("all", np.ones_like(sat))):
            rows.append(f"{JOINT_NAMES[j]:<14} {name:<13} {r_squared(model[keep], meas[keep]):8.4f}  "
                        f"{np.nanmax(np.abs(model - meas)[keep]):8.3f}     {r_squared(pred[keep], sag[keep]):8.5f}  "
                        f"{np.rad2deg(np.nanmax(np.abs(pred - sag)[keep])):8.3f}")
    return "\n".join(rows)


def _load_table(data: dict) -> str:
    """Peak / RMS torque, saturation, heat per loaded joint and mass (scripted)."""
    rows = ["JOINT LOAD (scripted, mean over episodes): peak |τ| N·m (% stall) / RMS N·m / % steps saturated / "
            "Σ τ²Δt", "mass (kg)  " + "".join(f"{JOINT_NAMES[j]:<38}" for j in LOADED)]
    for i, m in enumerate(data["masses"]):
        cells = []
        for j in LOADED:
            peak, rms = mean(field(data, "scripted", "peak_nm", j))[i], mean(field(data, "scripted", "rms_nm", j))[i]
            sat, heat = mean(field(data, "scripted", "saturated_frac", j))[i], mean(field(data, "scripted",
                                                                                         "heat_n2m2s", j))[i]
            cells.append(f"{peak:4.2f} ({100 * peak / STALL_NM:3.0f}%) / {rms:4.2f} / {100 * sat:4.1f}% / {heat:5.2f}")
        rows.append(f"{m:<11g}" + "".join(c.ljust(38) for c in cells))
    return "\n".join(rows)


def _sag_table(data: dict) -> str:
    """Hold-window sag per joint, cube drop split, and grip margin (scripted)."""
    rows = ["SAG AND DROP (scripted, hold window): sag measured/predicted (deg); drop total = arm + slip (mm), "
            "predicted arm; grip ΣN (N) and friction margin μΣN/mg",
            "mass (kg)  lift            elbow           wrist_flex      drop  = arm  + slip  pred   ΣN    margin"]
    for i, m in enumerate(data["masses"]):
        sags = "".join(f"{np.rad2deg(mean(field(data, 'scripted', 'sag_measured', j))[i]):6.2f}/"
                       f"{np.rad2deg(mean(field(data, 'scripted', 'sag_predicted', j))[i]):6.2f}   " for j in LOADED)
        d = [1000 * mean(field(data, "scripted", k))[i] for k in ("drop_measured_m", "drop_arm_m", "slip_m",
                                                                    "drop_predicted_m")]
        grip, margin = mean(field(data, "scripted", "grip_force_n"))[i], mean(field(data, "scripted", "grip_margin"))[i]
        rows.append(f"{m:<11g}{sags}{d[0]:5.1f} {d[1]:6.1f} {d[2]:6.1f} {d[3]:6.1f} {grip:6.1f} {margin:6.1f}")
    return "\n".join(rows)


def _capacity_table(data: dict) -> str:
    """Largest static payload at carry height for a few reaches."""
    rows = [f"STATIC PAYLOAD LIMIT at carry height {100 * CARRY_HEIGHT_M:.0f} cm (kg, limiting joint)",
            "reach (cm) " + "".join(f"{100 * r:<16.0f}" for r in CAPACITY_RADII[::2])]
    for limit, cmap in data["capacity"].items():
        h = int(np.argmin(np.abs(cmap.heights_m - CARRY_HEIGHT_M)))
        cells = [f"{cmap.m_max_kg[h, k]:4.2f} {JOINT_NAMES[cmap.joint[h, k]][:8]}" for k in range(0, len(
            CAPACITY_RADII), 2)]
        rows.append(f"{limit:<4} N·m   " + "".join(c.ljust(16) for c in cells))
    return "\n".join(rows)


def _controller_table(data: dict) -> str:
    """Success, error, correction vs needed, and drop for every controller."""
    rows = ["CONTROLLERS: success % / final error cm / at placement: lift & elbow offset θ_cmd−θ* vs needed sag "
            "(deg), horizontal cube−goal (mm) / carry drop (mm)"]
    for c in CONTROLLERS:
        rows.append(f"-- {c}")
        for i, m in enumerate(data["masses"]):
            ok, err = 100 * mean(field(data, c, "success"))[i], 100 * mean(field(data, c, "error_m"))[i]
            corr = [np.rad2deg(mean(field(data, c, "place_correction", j))[i]) for j in (LIFT, ELBOW)]
            need = [np.rad2deg(mean(field(data, c, "place_sag_predicted", j))[i]) for j in (LIFT, ELBOW)]
            offset, drop = (1000 * mean(field(data, c, k))[i] for k in ("place_offset_m", "drop_measured_m"))
            rows.append(f"   {m:<6g} {ok:5.1f}%  {err:4.1f} cm   lift {corr[0]:6.2f} / {need[0]:6.2f}   "
                        f"elbow {corr[1]:6.2f} / {need[1]:6.2f}   offset {offset:5.1f}   drop {drop:5.1f}")
    return "\n".join(rows)


def _residual_table(data: dict) -> str:
    """Mean |residual| per phase (deg, averaged over joints) for both trained policies at every mass."""
    rows = [f"RESIDUAL ACTIVITY: mean |residual| per step (deg, mean of 5 joints); cap {RESIDUAL_CAP_DEG:.2f}",
            "policy  mass    " + "".join(f"{p:<9}" for p in PHASES)]
    for c in ("fixed", "random"):
        for m in data["masses"]:
            grid = np.rad2deg(residual_grid(data, c, m)).mean(axis=1)
            rows.append(f"{c:<8}{m:<8g}" + "".join(f"{v:<9.3f}" for v in grid))
    return "\n".join(rows)


# ================================================================ main


def main() -> None:
    """Run (or load) the sweep, then write the figures and the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50, help="per controller and mass")
    parser.add_argument("--masses", type=float, nargs="+", default=list(EVAL_MASSES_KG))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--plot-only", action="store_true", help="reuse <out>/sweep.pkl")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "sweep.pkl"
    if args.plot_only:
        data = pickle.loads(cache.read_bytes())
    else:
        data = sweep(tuple(args.masses), args.episodes, args.workers)
        cache.write_bytes(pickle.dumps(data))
    figures = (fig_breakdown, fig_validation, fig_sag_drop, fig_capacity, fig_load, fig_correction, fig_success,
               fig_residual)
    names = ("torque_breakdown", "model_validation", "sag_and_drop", "capacity", "load_vs_mass", "correction",
             "success", "residual_by_phase")
    for k, (draw, name) in enumerate(zip(figures, names), start=1):
        draw(data, args.out / f"fig{k}_{name}.png")
    text = report(data)
    (args.out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
