"""Launch sequential ROME with the isolated interpreter and live progress."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=Path, default=ROOT/'results/related_groups/pilot_groups.json')
    parser.add_argument('--order', choices=['forward','reverse','both'], default='forward')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    command = [str(ROOT/'.venv/bin/python'), '-u', str(ROOT/'scripts/run_multi_edit.py'),
               '--limit', str(args.limit), '--output', str(args.output.resolve()),
               '--groups', str(args.groups.resolve()), '--order', args.order]
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
            if process.wait():
                raise RuntimeError('Multi-edit stage failed. Inspect output; completed verified results remain resumable.')
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


if __name__ == '__main__':
    main()
