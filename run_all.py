#!/usr/bin/env python3
"""Recompute and plot Figures 2--5 without using any precomputed numerical data."""
from pathlib import Path
import argparse,subprocess,sys,os,json
from cleanup_outputs import prune_figure
ROOT=Path(__file__).resolve().parent

def run(script,*args,cwd=None):
 cmd=[sys.executable,str(script),*map(str,args)];print('RUN:', ' '.join(cmd),flush=True)
 subprocess.run(cmd,cwd=cwd,env={**os.environ,'MPLBACKEND':'Agg'},check=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=['smoke','publication'],default='smoke');p.add_argument('--jobs',type=int,default=1);p.add_argument('--figures',nargs='+',choices=['2','3','4','5'],default=['2','3','4','5']);p.add_argument('--out',type=Path,default=Path('results'));p.add_argument('--city-workbook',type=Path,default=None,help='optional locally downloaded UN File 21 workbook (otherwise downloaded automatically)');p.add_argument('--keep-intermediates',action='store_true',help='keep per-run Monte Carlo files, plot inputs and other temporary outputs');p.add_argument('--city-K','--city-rank-limit',type=int,default=40,help='minimum qualifying cities and last rank for Figure 5 (default: 40)');a=p.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 if '2' in a.figures:
  run(ROOT/'figure2/reproduce_figure2.py','--profile',a.profile,'--output',out/'figure2')
  if not a.keep_intermediates: prune_figure(out/'figure2','2')
 if '3' in a.figures:
  # Full historical MC mean-field initialization is computed from source.
  run(ROOT/'figure3/source/run_deep_mc.py','--profile',a.profile,'--panels','A','--out',out/'figure3/mc')
  run(ROOT/'figure3/fresh_plot.py','--profile',a.profile,'--mc-dir',out/'figure3/mc','--outdir',out/'figure3')
  if not a.keep_intermediates: prune_figure(out/'figure3','3')
 if '4' in a.figures:
  for panel in ['AB','C','D','EF']:
   args=['--panel',panel,'--profile',a.profile,'--outdir',out/f'figure4/4{panel}','--jobs',a.jobs]
   if panel=='C':args+=['--new-background']
   if panel=='D' and a.profile=='smoke':args+=['--means','1.38','2.12','2.97','4.03']
   run(ROOT/'figure4/simulate.py',*args)
  run(ROOT/'figure4/build_fresh_plot.py','--results',out/'figure4','--outdir',out/'figure4/assembly')
  if not a.keep_intermediates: prune_figure(out/'figure4','4',a.profile)
 if '5' in a.figures:
  # The city-size Figure 5 is a separate self-contained Python package.
  # It downloads official UN File 21 data on the first run, unless supplied locally.
  env={**os.environ,'MPLBACKEND':'Agg','PYTHONPATH':str(ROOT/'figure5/src')+os.pathsep+os.environ.get('PYTHONPATH','')}
  if a.profile=='smoke':
   # Only decreases diagnostic Monte Carlo replicates; observational analysis is unchanged.
   env['PARETO_FIG5_REPLICATES']='1000'
  cmd=[sys.executable,'-m','pareto_figure5','--output-dir',str(out/'figure5'),'--cache-dir',str(ROOT/'.cache/figure5'),'--rank-limit',str(a.city_K)]
  if a.city_workbook is not None:cmd+=['--workbook',str(a.city_workbook.resolve())]
  print('RUN:', ' '.join(cmd),flush=True)
  subprocess.run(cmd,cwd=ROOT,env=env,check=True)
  if not a.keep_intermediates: prune_figure(out/'figure5','5')
 (out/'run_manifest.json').write_text(json.dumps({'profile':a.profile,'figures':a.figures,'jobs':a.jobs,'city_K':a.city_K,'minimal_output':not a.keep_intermediates,'fresh_computation_only':True,'city_workbook_is_external_empirical_input':True},indent=2))
 print('COMPLETE:',out)
if __name__=='__main__':main()
