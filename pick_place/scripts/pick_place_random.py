"""Pick-and-place, RANDOM mass: every training episode draws a cube mass uniformly from 0.05-1.0 kg.

Same range as the fixed-mass run's heaviest cube, so the two runs differ only in how the mass is chosen.

    uv run python pick_place/scripts/pick_place_random.py train --seeds 0 1 2
    uv run python pick_place/scripts/pick_place_random.py watch --mass-kg 0.5 --start 0.22 -80 --goal 0.15 60
    uv run python pick_place/scripts/pick_place_random.py compare --plot compare.png
"""

from rbe_501_so101.config import ObjectMassConfig
from rbe_501_so101.pick_place.env import PlaceRun, run_cli

MIN_KG, MAX_KG = 0.05, 1.0

RUN = PlaceRun(ObjectMassConfig(mode="random", range_kg=(MIN_KG, MAX_KG)))  # saved in .../pick_place_random_50_1000g_25mm

if __name__ == "__main__":
    run_cli(RUN, __doc__)
