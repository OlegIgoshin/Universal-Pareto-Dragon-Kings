"""Compute Figure 2 Monte Carlo runs and continuum prediction from scratch."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import sys
from importlib.metadata import version, PackageNotFoundError
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {4: '#1f77b4', 10: '#ff7f0e'}


def continuum_density(m, mean, a=0.15):
    """Appendix A5 with p0 fixed by A7; a=abs(eta_inf)>0.

    This is a density in the continuous approximation. It is deliberately not
    normalized by a discrete sum; doing so would change the manuscript theory.
    """
    m = np.asarray(m, dtype=float)
    p0 = 4.0*a / np.expm1(4.0*a*mean)
    k = 4.0*a + p0
    logz = np.log(p0/(4.0*a)) + k*m
    q = np.exp(-np.abs(logz))
    return k*k/(4.0*a) * q/(1.0+q)**2


def _figure(out, tag, fresh):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
        'axes.labelsize':9.2, 'axes.linewidth':0.8, 'xtick.labelsize':8,
        'ytick.labelsize':8, 'xtick.direction':'out', 'ytick.direction':'out',
        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
        'mathtext.fontset':'dejavuserif', 'savefig.facecolor':'white'})
    fig = plt.figure(figsize=(3.81,3.18))
    ax = fig.add_axes([0.151,0.143,0.815,0.845])
    inset = ax.inset_axes([0.505,0.16,0.427,0.61])
    for mean in [10,4]:
        color=COLORS[mean]
        x,p=fresh[mean]
        line=np.linspace(0,40,1000)
        theory=continuum_density(line,mean)
        ax.plot(line,theory,color=color,lw=1.5)
        inset.plot(line,theory,color=color,lw=0.95)
        ax.plot(x,p,'o',mfc='none',mec=color,mew=0.75,ms=3.6,ls='none')
        good=p>0
        inset.plot(x[good],p[good],'o',mfc='none',mec=color,mew=0.5,ms=2.2,ls='none')
    ax.set(xlim=(0,40),ylim=(0,0.19),xlabel='Wealth, $m$',ylabel='$p(m)$')
    ax.set_xticks([0,10,20,30,40]);ax.set_yticks([0,0.05,0.10,0.15])
    ax.xaxis.set_minor_locator(MultipleLocator(5));ax.yaxis.set_minor_locator(MultipleLocator(.025))
    ax.tick_params(which='major',length=3.2,width=.8);ax.tick_params(which='minor',length=2.2,width=.6)
    handles=[]
    for mean in [10,4]:
        handles.extend([Line2D([0],[0],color=COLORS[mean],lw=1.5,
            label=fr'theory, $\langle m\rangle={mean}$'),
            Line2D([0],[0],color=COLORS[mean],marker='o',mfc='none',ms=3.6,
            mew=.75,ls='none',label=fr'MC, $\langle m\rangle={mean}$')])
    ax.legend(handles=handles,ncol=2,frameon=False,fontsize=7.4,
              loc='upper right',borderaxespad=.25,handlelength=1.8,
              handletextpad=.35,columnspacing=.65,labelspacing=.3)
    inset.set(yscale='log',xlim=(-1,30),ylim=(1e-5,.25))
    inset.set_xticks([0,10,20,30]);inset.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1])
    inset.tick_params(labelsize=6.2,length=2,width=.55,pad=1)
    inset.tick_params(which='minor',length=1.4,width=.4)
    for spine in inset.spines.values():spine.set_linewidth(.55)
    for ext in ['pdf','svg','png']:
        fig.savefig(out/f'{tag}.{ext}',dpi=300)
    plt.close(fig)


try:
    from numba import njit
    HAVE_NUMBA=True
except ImportError:
    HAVE_NUMBA=False
    def njit(*args,**kwargs):
        return lambda f:f


@njit(cache=True)
def _numba_kernel(n,mean,a,burn,sweeps,stride,block_sweeps,seed):
    """One sweep is one shuffled perfect matching: N/2 attempted exchanges."""
    np.random.seed(seed)
    wealth=np.full(n,mean,dtype=np.int64)
    order=np.arange(n,dtype=np.int64)
    blocks=np.zeros((sweeps//block_sweeps,n*mean+1),dtype=np.int64)
    for sweep in range(burn+sweeps):
        np.random.shuffle(order)
        for k in range(0,n-1,2):
            i=order[k];j=order[k+1]
            d=wealth[i]-wealth[j]
            eta=-a if d>0 else (a if d<0 else 0.0)
            if np.random.random()<0.5+eta:
                if wealth[j]>0:wealth[i]+=1;wealth[j]-=1
            elif wealth[i]>0:wealth[i]-=1;wealth[j]+=1
        production_sweep=sweep-burn+1
        if production_sweep>0 and production_sweep%stride==0:
            block=(production_sweep-1)//block_sweeps
            for i in range(n):blocks[block,wealth[i]]+=1
    return wealth,blocks


def _numpy_kernel(n,mean,a,burn,sweeps,stride,block_sweeps,seed):
    rng=np.random.default_rng(seed)
    wealth=np.full(n,mean,dtype=np.int64)
    blocks=np.zeros((sweeps//block_sweeps,n*mean+1),dtype=np.int64)
    for sweep in range(burn+sweeps):
        order=rng.permutation(n);i=order[::2];j=order[1::2]
        wi=wealth[i].copy();wj=wealth[j].copy()
        receives=rng.random(n//2)<.5-a*np.sign(wi-wj)
        valid_i=receives&(wj>0);valid_j=(~receives)&(wi>0)
        wi[valid_i]+=1;wj[valid_i]-=1;wi[valid_j]-=1;wj[valid_j]+=1
        wealth[i]=wi;wealth[j]=wj
        production_sweep=sweep-burn+1
        if production_sweep>0 and production_sweep%stride==0:
            block=(production_sweep-1)//block_sweeps
            counts=np.bincount(wealth,minlength=n*mean+1)
            blocks[block]+=counts
    return wealth,blocks


def simulate(out,n=10000,burn=5000,sweeps=10000,stride=10,replicates=2,seed=20261030):
    if n%2 or n<2:raise ValueError('N must be even and at least 2')
    if sweeps<=0 or burn<0 or stride<=0 or replicates<1:raise ValueError('Invalid simulation schedule')
    if sweeps%stride:raise ValueError('Production sweeps must be divisible by sampling stride')
    block_sweeps=math.gcd(sweeps,1000)
    block_sweeps=math.lcm(block_sweeps,stride)
    if sweeps%block_sweeps:raise ValueError('Production sweeps must be divisible by block size')
    fresh={};details=[]; t0=time.monotonic()
    engine='Numba numpy.random legacy MT19937' if HAVE_NUMBA else 'NumPy Generator PCG64'
    for mean in [4,10]:
        accumulator=np.zeros((sweeps//block_sweeps,n*mean+1),dtype=np.int64)
        replica_totals=[]
        for rep in range(replicates):
            run_seed=(seed+1000003*rep+1009*mean)%(2**32)
            kernel=_numba_kernel if HAVE_NUMBA else _numpy_kernel
            wealth,blocks=kernel(n,mean,.15,burn,sweeps,stride,block_sweeps,run_seed)
            assert int(wealth.sum())==n*mean and int(wealth.min())>=0
            assert int(blocks.sum())==n*(sweeps//stride)
            # The pooled empirical mean is exact since every snapshot conserves M.
            assert int(blocks.sum(axis=0)@np.arange(n*mean+1))==n*mean*(sweeps//stride)
            accumulator+=blocks;replica_totals.append(blocks.sum(axis=0))
            pblocks=blocks/blocks.sum(axis=1,keepdims=True)
            half=max(1,len(pblocks)//2)
            tv=float(.5*np.abs(pblocks[:half].mean(axis=0)-pblocks[half:].mean(axis=0)).sum()) if len(pblocks)>1 else None
            details.append({'mean':mean,'replicate':rep+1,'seed':int(run_seed),
                'N':n,'M':n*mean,'burn_sweeps':burn,'production_sweeps':sweeps,
                'sample_stride_sweeps':stride,'sampled_snapshots':sweeps//stride,
                'block_sweeps':block_sweeps,'production_half_TV':tv,
                'final_maximum_wealth':int(wealth.max()),'conserved_total':int(wealth.sum())})
            print(f'Figure2 NEW run mean={mean}, replicate={rep+1}, seed={run_seed}, half-TV={tv}',flush=True)
        counts=accumulator.sum(axis=0);last=int(np.flatnonzero(counts)[-1]);m=np.arange(last+1)
        probability=counts[:last+1]/counts.sum();fresh[mean]=(m,probability)
        with (out/f'new_mc_mean{mean}.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['m','p_new_mc','pooled_count','p_continuum_A5'])
            writer.writerows(zip(m,probability,counts[:last+1],continuum_density(m,mean)))
        np.savez_compressed(out/f'new_mc_mean{mean}_blocks.npz',
            block_counts=accumulator[:,:last+1],replica_counts=np.array(replica_totals)[:,:last+1])
    versions={'python':sys.version.split()[0],'numpy':np.__version__}
    for package in ['matplotlib','numba']:
        try:versions[package]=version(package)
        except PackageNotFoundError:versions[package]=None
    settings={'status':'NEW simulation; not the historical run underlying the supplied PDF',
        'eta_rule':'eta(d)=-0.15*sign(d), d=w_i-w_j',
        'receive_i_probability':'0.5+eta(w_i-w_j)',
        'boundary_rule':'Select direction first; reject if the selected donor has zero wealth.',
        'initial_condition':'Homogeneous: every agent starts with the integer mean wealth.',
        'sweep_definition':'Uniform random perfect matching; one attempted pair interaction per agent, N/2 pair attempts.',
        'means':[4,10],'a':.15,'engine':engine,'software_versions':versions,
        'elapsed_seconds':time.monotonic()-t0,'runs':details,
        'uncertainty':'Snapshots and agent observations are correlated; pooled counts are not independent sample sizes. Half-TV is a stationarity diagnostic, not a proof or a tail error bar.'}
    (out/'new_mc_settings.json').write_text(json.dumps(settings,indent=2))
    _figure(out,'figure2_new_mc',fresh)
    return settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['smoke','publication'],default='smoke')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    simulate(args.output,**({'n':2000,'burn':500,'sweeps':1000,'replicates':1}
                          if args.profile=='smoke' else {}))

if __name__=='__main__':main()
