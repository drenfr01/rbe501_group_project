"""Train, evaluate, and benchmark the SO-101 reaching policy.

HOW TO RUN (from the project folder):
    python scripts/train_reach.py --algo sac --timesteps 300000 --seeds 0 1 2 --model-dir checkpoints/reach_new
    python scripts/train_reach.py --preset residual_ik --timesteps 1000000 --model-dir checkpoints/reach_1m_residual_ik
    python scripts/train_reach.py --eval --model-dir checkpoints/reach/ --num-episodes 20
    python scripts/train_reach.py --eval --render

Training, evaluation, and policies live in `rbe_501_so101.rl` so they can be
imported and unit-tested without going through argparse.
"""

from rbe_501_so101.rl.cli import main

if __name__ == "__main__":
    main()
