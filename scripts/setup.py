"""Install the isolated Linux/NVIDIA environment; safe to rerun."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / '.venv'
REPO = ROOT / 'vendor/rome'
COMMIT = '0874014cd9837e4365f3e6f3c71400ef11509e04'
os.environ['MPLBACKEND'] = 'Agg'

def run(args, cwd=None):
    with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Setup failed; see output above.')

def main():
    if sys.platform != 'linux':
        raise SystemExit('Run setup on Linux with an NVIDIA GPU, such as Colab.')
    run([sys.executable, '-m', 'pip', 'install', 'uv==0.13.1'])
    uv = [sys.executable, '-m', 'uv']
    run(uv + ['python', 'install', '3.11'])
    python = str(ENV / 'bin/python')
    if not Path(python).exists():
        run(uv + ['venv', '--python', '3.11', '--seed', str(ENV)])
    run([python, '-m', 'pip', 'install', 'torch==2.5.1',
         '--index-url', 'https://download.pytorch.org/whl/cu121'])
    run([python, '-m', 'pip', 'install', '-r', str(ROOT/'requirements.txt')])
    run([python, '-m', 'pip', 'check'])
    REPO.parent.mkdir(exist_ok=True)
    if not REPO.exists():
        run(['git', 'clone', 'https://github.com/kmeng01/rome.git', str(REPO)])
    changes = subprocess.check_output(['git','-C',str(REPO),'status','--porcelain','--untracked-files=no'],text=True)
    if changes.strip():
        raise RuntimeError('Upstream ROME checkout has modifications; inspect them before setup.')
    run(['git','-C',str(REPO),'checkout','--detach',COMMIT])
    run([python, '-c', 'import torch; from rome import apply_rome_to_model; from experiments.causal_trace import ModelAndTokenizer; assert torch.cuda.is_available(); x=torch.ones((32,32),device="cuda"); assert (x@x)[0,0].item()==32.; print("SETUP COMPLETE:",torch.cuda.get_device_name(0))'], cwd=REPO)

if __name__ == '__main__': main()
