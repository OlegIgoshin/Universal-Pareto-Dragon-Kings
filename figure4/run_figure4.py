#!/usr/bin/env python3
"""Run Figure 4 from source in standalone package."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', choices=['smoke','publication'], default='smoke')
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--out', type=Path, default=Path('figure4_results'))
    args = p.parse_args()
    out = args.out.resolve()
    for panel in ['AB','C','D','EF']:
        cmd = [sys.executable, str(HERE/'simulate.py'), '--profile', args.profile,
               '--panel', panel, '--outdir', str(out/f'4{panel}'),
               '--jobs', str(args.jobs)]
        if panel == 'C': cmd.append('--new-background')
        subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(HERE/'build_fresh_plot.py'),
                    '--results', str(out), '--outdir', str(out/'assembly')],check=True)

if __name__=='__main__': main()
