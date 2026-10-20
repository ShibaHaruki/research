"""Short commands for the unified calibration workflow only."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'verify'])
    args, extra = parser.parse_known_args()
    return subprocess.call([sys.executable, str(ROOT/'f_run/calibrate_liquid.py'),
                            args.action, *extra], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
