#!/usr/bin/env python3
"""Assemble original six-panel layout from newly simulated A/B,C,D,E/F outputs.
No archived numerical inputs. The crossover reference is an explicit finite-N
critical bulk-capacity estimate kappa*ln(kappa*N), not the archived revised bulk-matching calculation.
"""
from pathlib import Path
import argparse,json,sys
import pandas as pd,numpy as np
HERE=Path(__file__).resolve().parent
PC=2/3;K=.5

def mean_distribution(d):
 files=sorted((d/'runs').glob('*/distribution.csv'))
 if not files:raise RuntimeError('No new distribution.csv under '+str(d))
 samples=[]
 for f in files:
  df=pd.read_csv(f);mu=float(df.m.to_numpy()@df.p.to_numpy());samples.append((mu,df))
 groups={}
 for mu,df in samples:
  key=round(mu,3);groups.setdefault(key,[]).append(df)
 out=[]
 for mu,dfs in groups.items():
  maxm=max(x.m.max() for x in dfs);array=[]
  for f in dfs:
   arr=np.zeros(maxm+1);arr[f.m.to_numpy(int)]=f.p.to_numpy(float);array.append(arr)
  p=np.mean(array,axis=0);S=np.cumsum(p[::-1])[::-1]
  # each S(m)=Pr(X>=m), matching saved output S definition
  out.append(pd.DataFrame({'q':round(mu/2.12,4),'mean_wealth':mu,'m':np.arange(maxm+1),'S':S}))
 return pd.concat(out,ignore_index=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args()
 out=a.outdir;out.mkdir(parents=True,exist_ok=True)
 src=a.results
 for orig,target in [('4AB/selected_trajectories_early_logsample.csv','selected_trajectories_early_logsample.csv'),('4AB/global_observables.csv','global_observables.csv'),('4C/stationary_distribution_dense.csv','stationary_distribution_dense.csv'),('4EF/EF_summary.csv','EF_summary.csv')]:
  p=src/orig
  if not p.exists():raise RuntimeError('Missing fresh output '+str(p))
  (out/target).write_bytes(p.read_bytes())
 tails=mean_distribution(src/'4D');tails.to_csv(out/'finite_size_tail_curves.csv',index=False)
 # Calculate critical MF using the original FFT-based kernel, no saved array
 sys.path.insert(0,str(HERE.parent/'figure3'))
 from mf_kernels import solve_fixed
 mmax=2000;p,stat=solve_fixed(PC+1e-5/3,.25,3.,'step',mmax,1e-9,600)
 m=np.arange(mmax+1);S=np.cumsum(p[::-1])[::-1]
 pd.DataFrame({'m':m,'S_meanfield':S}).to_csv(out/'critical_meanfield_solution.csv',index=False)
 # The original revised crossover computation was not supplied as code.
 # Use explicitly documented leading-order analytical estimate, not a historical copied number.
 mux=K*np.log(K*300)
 pd.DataFrame({'N':[300],'mbar_x_revised':[mux]}).to_csv(out/'critical_bulk_matching_revised.csv',index=False)
 (out/'crossover_note.json').write_text(json.dumps({'reference':'kappa ln(kappa N) finite-size crossover estimate, NOT original revised matching','kappa':K,'N':300,'mu_x':mux,'original_method_recovered':False},indent=2))
 import subprocess
 subprocess.run([sys.executable,str(HERE/'replot.py'),'--data',str(out),'--outdir',str(out/'figures')],check=True)
 print('Six-panel figure assembled from all fresh outputs at',out/'figures')
if __name__=='__main__':main()
