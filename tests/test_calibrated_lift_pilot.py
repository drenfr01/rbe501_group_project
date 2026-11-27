"""Pilot promotion decisions require completed, contact-free evaluation evidence."""

from scripts.sim2real.analyze.calibrated_lift_pilot import decision


def test_empty_evaluation_cannot_pass():
    """An interrupted pilot must not appear ready for a larger run."""
    assert not decision([])["passed_pilot_gate"]


def test_table_contacts_reject_high_success_policy():
    """Successful lifts alone do not satisfy the pilot's clearance gate."""
    results = [{"summary": {"success_rate": 1., "robot_table_contact_episodes": 2,
                             "episodes": 20, "warning_episodes": 0}}]
    assert not decision(results)["passed_pilot_gate"]
