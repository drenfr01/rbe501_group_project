"""Keyboard jog tool for the SO-101 follower arm.

Keys:
  1-6        select joint
  + / =  or  w   move selected joint up by one step
  -      or  s   move selected joint down by one step
  [ / ]      halve / double the step size
  q          quit (torque is disabled, so support the arm first)
"""

import argparse
import select
import sys
import termios
import tty

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def read_positions(robot):
    obs = robot.get_observation()
    return {j: obs[f"{j}.pos"] for j in JOINTS}


def joint_limits(robot):
    # Mirrors LeRobot's DEGREES normalization: 0 is the midpoint of the calibrated raw range.
    limits = {"gripper": (0.0, 100.0)}
    for j in JOINTS[:-1]:
        cal = robot.calibration[j]
        half = (cal.range_max - cal.range_min) / 2 * 360 / 4095
        limits[j] = (-half, half)
    return limits


def table(selected, step, target, actual, limits):
    lines = [f"step: {step:g}", f"   {'joint':15s} {'target':>8s} {'actual':>8s} {'min':>8s} {'max':>8s}"]
    for i, j in enumerate(JOINTS):
        marker = ">" if j == selected else " "
        lo, hi = limits[j]
        lines.append(f"{marker}{i + 1} {j:15s} {target[j]:8.2f} {actual[j]:8.2f} {lo:8.1f} {hi:8.1f}")
    return lines


def redraw(lines, first):
    # Overwrite the previous table in place; clearing the screen floods the scrollback in some terminals.
    out = "" if first else f"\x1b[{len(lines)}F"
    out += "".join(f"{line}\x1b[K\n" for line in lines)
    sys.stdout.write(out)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/so101_follower")
    parser.add_argument("--id", default="rbe501_follower")
    parser.add_argument("--step", type=float, default=2.0, help="initial step (degrees; gripper uses 0-100)")
    args = parser.parse_args()

    robot = SO101Follower(
        SO101FollowerConfig(port=args.port, id=args.id, max_relative_target=10.0)
    )
    robot.connect(calibrate=False)

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        target = read_positions(robot)
        limits = joint_limits(robot)
        selected, step = JOINTS[0], args.step
        print(__doc__.strip() + "\n")
        shown = None

        while True:
            lines = table(selected, step, target, read_positions(robot), limits)
            if lines != shown:
                redraw(lines, first=shown is None)
                shown = lines
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not ready:
                continue
            key = sys.stdin.read(1)

            if key == "q":
                break
            if key in "123456":
                selected = JOINTS[int(key) - 1]
            elif key == "[":
                step = max(step / 2, 0.25)
            elif key == "]":
                step = min(step * 2, 10.0)
            elif key in "+=-ws":
                delta = step if key in "+=w" else -step
                lo, hi = limits[selected]
                target[selected] = min(max(target[selected] + delta, lo), hi)
                sent = robot.send_action({f"{j}.pos": v for j, v in target.items()})
                # Follow what was actually sent, so a joint that lags or is blocked can't accumulate a runaway target.
                target = {j: sent[f"{j}.pos"] for j in JOINTS}
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        robot.disconnect()
        print("Disconnected; torque off.")


if __name__ == "__main__":
    main()
