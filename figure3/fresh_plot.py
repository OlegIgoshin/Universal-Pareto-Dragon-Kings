#!/usr/bin/env python3
"""Fresh Figure 3 A/B/C calculations, with no saved-input arrays."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source/src'))
from mean_field import solve_for_mean,critical_p0,kappa
from mf_kernels import solve_fixed
from scipy.optimize import minimize_scalar
C=.25; PC=critical_p0(C); K=kappa(C)


def matched_theory(frame):
    m=frame.m.to_numpy(float);p=frame.p.to_numpy(float)
    delta=float(frame.eps_rel.iloc[0]);fit=(m>=20)&(m<=200)&(p>=1e-8)
    if fit.sum()<5:raise ValueError('Insufficient MF matching points')
    def evaluate(shift):
        z=delta*(m+shift)
        return .5*delta**2*np.exp(-z)/(-np.expm1(-z))**2
    def objective(shift):
        v=evaluate(shift)
        return float(np.mean((np.log(v[fit])-np.log(p[fit]))**2))
    sol=minimize_scalar(objective,bounds=(0,100),method='bounded')
    if not sol.success:raise RuntimeError('Matching unsuccessful')
    return evaluate(sol.x),{'shift':float(sol.x),'log_rms':float(np.sqrt(sol.fun))}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['smoke','publication'],default='smoke');ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--mc-dir',type=Path,required=True); a=ap.parse_args()
 out=a.outdir;out.mkdir(parents=True,exist_ok=True)
 smoke=a.profile=='smoke';model=json.loads((HERE/'source/configs'/f'{a.profile}.json').read_text())['model']
 mmax=4000 if smoke else int(model['mean_field_mmax']);tol=1e-7 if smoke else float(model['mean_field_tolerance']); iterations=300 if smoke else int(model['mean_field_max_iterations'])
 fig,axes=plt.subplots(1,3,figsize=(11.6,3.6));ax,bx,cx=axes
 # A: mean-wealth versus p0-pc computed from fresh stationary MF solutions
 means=np.linspace(.45,3.2,11 if smoke else 36)
 rec=[]
 for mu in means:
  res=solve_for_mean(float(mu),C,3.,mmax,tol,iterations)
  rec.append({'eps_abs':res.p0-PC,'mean_wealth':res.mean,'converged':res.converged})
 df=pd.DataFrame(rec);df.to_csv(out/'A_meanfield_fresh.csv',index=False)
 ax.semilogx(df.eps_abs,df.mean_wealth,color='C0',label='Mean field')
 mc=pd.read_csv(a.mc_dir/'panelA_summary.csv');good=mc.mean_p0_minus_pc>0
 ax.errorbar(mc.loc[good,'mean_p0_minus_pc'],mc.loc[good,'mean_wealth_fixed'],xerr=mc.loc[good,'se_p0'],fmt='o',color='C1',label='Monte Carlo')
 ax.set(xlabel=r'$p_0-p_c$',ylabel=r'Mean wealth $\langle m\rangle$',xlim=(2e-4,.1),ylim=(.35,3.35));ax.legend(frameon=False)
 # B: solve the original tanh MF recurrence at fixed mean for every curve; no CSV imports
 bframes=[]
 for i,mu in enumerate([1.4,2.6,3.1]):
  res=solve_for_mean(mu,C,3.,mmax,tol,iterations)
  m=np.arange(1,len(res.probability),dtype=int);frame=pd.DataFrame({'curve':f'mean{mu}', 'm':m,'p':res.probability[1:],'eps_rel':(res.p0-PC)/(1-PC),'mean_wealth':res.mean});bframes.append(frame)
  mask=m<=2200
  bx.loglog(m[mask],res.probability[1:][mask],color=f'C{i}',label=fr'MF, $\langle m\rangle={mu}$')
  try:
   theory,info=matched_theory(frame);bx.loglog(m[mask],theory[mask],color=f'C{i}',ls='--')
  except (ValueError,RuntimeError) as exc:print('B theory matching skipped:',exc)
 pd.concat(bframes).to_csv(out/'B_meanfield_fresh.csv',index=False)
 x=np.array([15.,150.]);bx.loglog(x,2/x**2,'k--',lw=1);bx.set(xlabel='Wealth $m$',ylabel='$p(m)$',xlim=(1,2200),ylim=(1e-8,.05));bx.legend(fontsize=7,frameon=False)
 # C: fixed-p0 mean-field solver, all four kernels (tanh and Hill included)
 delta=1e-3 if smoke else 1e-5
 mmaxc=3000 if smoke else 800000
 tol_c=1e-7 if smoke else 1e-12
 niter=250 if smoke else 1200
 rows=[];diag=[]
 for i,rule in enumerate(['step','tanh','hill4','hill2']):
  p,stats=solve_fixed(PC+(1-PC)*delta,C,3.,rule,mmaxc,tol_c,niter);diag.append(stats)
  if not smoke:
   p_large,large_stats=solve_fixed(PC+(1-PC)*delta,C,3.,rule,1600000,tol_c,niter)
   comp_on_common=np.arange(1,20001)
   denom=np.maximum(p_large[comp_on_common],1e-300)
   large_stats['max_relative_difference_m1_to_20000']=float(np.max(np.abs(p[comp_on_common]-p_large[comp_on_common])/denom))
   diag.append(large_stats)
  m=np.arange(8,min(mmaxc,20000)+1);comp=m**2*p[m]/K
  cx.semilogx(m,comp,color=f'C{i}',label={'step':'step','tanh':'tanh','hill4':'Hill n=4','hill2':'Hill n=2'}[rule]);rows += [{'curve':rule,'m':int(j),'compensated':float(y)} for j,y in zip(m[::10],comp[::10])]
 pd.DataFrame(rows).to_csv(out/'C_meanfield_fresh.csv',index=False)
 (out/'C_diagnostics.json').write_text(json.dumps(diag,indent=2))
 cx.axhline(1,color='k',ls='--');cx.set(xlabel='Wealth $m$',ylabel=r'$m^2p(m)/\kappa$',xlim=(8,20000 if not smoke else 3000),ylim=(.22,1.1));cx.legend(fontsize=7,frameon=False)
 for ax0,letter in zip(axes,'ABC'):ax0.text(-.12,1.025,letter,transform=ax0.transAxes,fontsize=13)
 fig.tight_layout()
 for ext in ['pdf','svg','png']:fig.savefig(out/f'Figure3_fresh.{ext}',dpi=250)
 plt.close(fig)
 print('Figure 3 complete; MF C convergence details in C_diagnostics.json')
if __name__=='__main__':main()
