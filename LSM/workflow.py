"""Short commands for calibration, state recording, and readout simulation."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'experiment':
        return subprocess.call([sys.executable, str(ROOT/'run_experiment.py'), *sys.argv[2:]], cwd=ROOT)
    entries = {'liquid': 'run_liquid.py', 'training': 'run_training.py', 'test': 'run_test.py',
               'train-test': 'run_train_test.py', 'classify': 'run_test_classification.py',
               'train-test-eval': 'run_train_test_eval.py'}
    if len(sys.argv) > 1 and sys.argv[1] in entries:
        return subprocess.call([sys.executable, str(ROOT/'f_run'/entries[sys.argv[1]]), *sys.argv[2:]], cwd=ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'verify', 'experiment', 'liquid', 'training', 'test', 'train-test', 'classify', 'train-test-eval'])
    args, extra = parser.parse_known_args()
    command = [sys.executable, str(ROOT/'f_run/run_calibration.py'), args.action, *extra]
    return subprocess.call(command, cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
