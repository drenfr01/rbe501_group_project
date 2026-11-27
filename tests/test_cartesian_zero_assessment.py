"""Reject extra paired failures even when an aggregate success threshold passes."""

from copy import deepcopy

from scripts.sim2real.analyze.cartesian_zero_pilot import assess


def successful_rows() -> list[dict]:
    """Represent twenty paired placements in each mass/dynamics condition."""
    episode = {"final_success": True, "ever_success": True,
               "robot_table_contact_steps": 0, "approach_contact_steps": 0,
               "warnings": False}
    return [{"arm_effects": effects, "summary": {"mass_kg": mass, "success_rate": 1.},
             "episodes": [{**episode, "placement_seed": seed} for seed in range(20)]}
            for effects in (False, True) for mass in (.005, .01, .015, .02)]


def test_extra_paired_failure_rejected_at_95_percent() -> None:
    """A new failure cannot be hidden by the per-condition 95-percent threshold."""
    reference = successful_rows()
    learned = deepcopy(reference)
    learned[0]["summary"]["success_rate"] = .95
    learned[0]["episodes"][0].update(final_success=False, ever_success=False)
    result = assess(reference, learned, reference)
    assert not result["passed_stability_gate"]
    assert result["additional_failure_cases"] == [[False, .005, 0]]


def test_matched_stability_does_not_authorize_full_training() -> None:
    """Matching the script establishes stability, not an improvement from RL."""
    rows = successful_rows()
    result = assess(rows, rows, rows)
    assert result["passed_stability_gate"]
    assert not result["useful_benefit_established"]
    assert not result["start_full_training"]
