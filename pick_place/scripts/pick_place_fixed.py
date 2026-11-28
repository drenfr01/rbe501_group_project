"""Pick-and-place, FIXED mass: every training episode uses the heaviest cube, 1.0 kg.

Why 1.0 kg: the scripted controller places 100% of cubes up to 0.5 kg but only 50% at 1.0 kg (the arm sags and
misses the 2 cm tolerance), while the servos still have ~25% torque headroom, so a policy can learn to fix it.

    uv run python pick_place/scripts/pick_place_fixed.py train --seeds 0 1 2
    uv run python pick_place/scripts/pick_place_fixed.py watch --mass-kg 0.5 --start 0.22 -80 --goal 0.15 60
    uv run python pick_place/scripts/pick_place_fixed.py compare --plot compare.png
"""

from rbe_501_so101.config import ObjectMassConfig
from rbe_501_so101.pick_place.env import PlaceRun, run_cli

MASS_KG = 1.0

RUN = PlaceRun(ObjectMassConfig(mode="fixed", fixed_kg=MASS_KG))  # saved in checkpoints/pick-place/pick_place_fixed_1000g_25mm

if __name__ == "__main__":
    run_cli(RUN, __doc__)
