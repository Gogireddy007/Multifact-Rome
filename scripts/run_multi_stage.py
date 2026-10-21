"""Launch the baseline with live progress and the isolated interpreter."""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, choices=[1,5,20], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = ROOT/'vendor/rome'
    for name in ['run_multi_edit.py','baseline_support.py']:
        shutil.copy2(ROOT/'scripts'/name, repo/name)
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    command = [str(ROOT/'.venv/bin/python'), '-u', 'run_multi_edit.py',
               '--limit', str(args.limit), '--output', str(args.output.resolve())]
    with subprocess.Popen(command,cwd=repo,env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line,end='',flush=True)
            if process.wait():
                raise RuntimeError('Baseline failed. Saved completed cases remain resumable.')
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise

if __name__ == '__main__': main()
