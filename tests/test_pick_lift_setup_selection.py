"""Selection uses tuning data only; table clearance is not assumed to be solved."""

from dataclasses import asdict

from scripts.sim2real.analyze.validate_pick_lift_setup import Setup, choose


def test_selection_uses_contact_count_after_success_threshold():
    """A low-success contact-free pose must not win over eligible poses."""
    records = [
        {"setup": asdict(Setup("gate", gated=True)), "success_rate": 0.975, "table_episodes": 39},
        {"setup": asdict(Setup("2mm", .002, True)), "success_rate": 0.95, "table_episodes": 12},
        {"setup": asdict(Setup("8mm", .008, True)), "success_rate": 0.5, "table_episodes": 0},
    ]
    assert choose(records).clearance_m == .002
