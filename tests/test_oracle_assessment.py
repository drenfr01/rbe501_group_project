"""Orientation gains cannot justify losing nominal grasps or accepting incomplete data."""

from scripts.sim2real.analyze.oracle_acquisition_check import OracleProtocol, assess


def complete_comparison() -> list[dict]:
    """Create complete paired conditions with a nominal-only baseline and perfect oracle."""
    cfg = OracleProtocol().validation
    return [{"controller": name, "mass_kg": mass,
             "case": {"name": f"yaw{angle:g}", "yaw_offset_deg": angle},
             "task": {"placement_seed": seed, "final_success": name == "oracle" or angle == 0,
                      "robot_table_contact_steps": 0, "approach_contact_steps": 0, "warnings": False}}
            for name in ("scripted", "oracle") for mass in cfg.masses_kg
            for angle in cfg.yaw_offsets_deg
            for seed in range(cfg.first_yaw_seed, cfg.first_yaw_seed + cfg.yaw_placements)]


def test_nominal_regression_blocks_large_rotated_success_gain() -> None:
    """Hundreds of recovered rotated cases must not hide one new nominal failure."""
    rows = complete_comparison()
    nominal = next(r for r in rows if r["controller"] == "oracle" and r["case"]["yaw_offset_deg"] == 0)
    nominal["task"]["final_success"] = False
    result = assess(rows, OracleProtocol())
    assert result["rotated_success_gain"]["ci95"][0] > 0
    assert not result["nominal_pass"]
    assert len(result["new_paired_failures"]) == 1
    assert not result["passed_validation"]


def test_incomplete_pairs_cannot_pass() -> None:
    """Matching a subset is insufficient for the frozen validation matrix."""
    rows = complete_comparison()
    seed = OracleProtocol().validation.first_yaw_seed
    rows = [r for r in rows if r["task"]["placement_seed"] != seed]
    result = assess(rows, OracleProtocol())
    assert not result["complete_pairs"]
    assert not result["passed_validation"]
