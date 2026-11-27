"""The failure-replay tool must follow the original task timing and outcomes."""

from scripts.sim2real.analyze.replay_pick_lift import phase_at, transition_events


def test_phase_boundaries_match_scripted_controller():
    """Steps are zero-based and the three pre-lift phases last 40/30/30 steps."""
    assert [phase_at(s) for s in (0, 39, 40, 69, 70, 99, 100, 199)] == [
        "approach", "approach", "descend", "descend", "close", "close", "lift", "lift",
    ]


def test_events_distinguish_brief_success_from_final_failure():
    """Contact/success losses need timestamps, not just a final boolean."""
    frames = [
        {"step": 1, "grasped": False, "success": False, "table_contact": False},
        {"step": 2, "grasped": True, "success": True, "table_contact": True},
        {"step": 3, "grasped": False, "success": False, "table_contact": False},
    ]
    events = transition_events(frames)
    assert {tuple(e.values()) for e in events} == {
        (2, "grasped", True), (2, "success", True), (2, "table_contact", True),
        (3, "grasped", False), (3, "success", False), (3, "table_contact", False),
    }
