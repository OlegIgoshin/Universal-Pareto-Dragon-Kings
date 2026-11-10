#!/usr/bin/env python3
"""Independent Monte Carlo replication; saved plot data are never overwritten."""
from pathlib import Path
import argparse,json,math
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def one_run(spec):
    import publication_runner as pr
    out=Path(spec['output_directory']); out.mkdir(parents=True,exist_ok=True)
    p=out/'input_spec.json';p.write_text(json.dumps(spec,indent=2))
    with (out/'run.log').open('a',encoding='utf-8') as log,redirect_stdout(log):
        pr.run(p)
    with np.load(out/'publication_checkpoint.npz',allow_pickle=False) as d:
        w=d['wealth']; ok=int(w.sum())==spec['M'] and bool(np.all(w>=0)) and len(w)==spec['N']
    if not ok:raise RuntimeError('Conservation/nonnegativity check failed: '+spec['run_id'])
    return json.loads((out/'summary.json').read_text())

def aggregate(summaries):
    """Trajectory-level means; SEM is across independent trajectories, not snapshots."""
    rows=[]
    for mu in sorted({s['mu'] for s in summaries}):
        ss=[s for s in summaries if s['mu']==mu]
        p=np.array([s['p0_mean'] for s in ss]); f=np.array([s['max_fraction_mean'] for s in ss])
        p_se=p.std(ddof=1)/len(p)**.5 if len(p)>1 else 0.
        f_se=f.std(ddof=1)/len(f)**.5 if len(f)>1 else 0.
        rows.append(dict(mean_wealth=mu,p0_mean=p.mean(),p0_se=p_se,
                         max_fraction_mean=f.mean(),max_fraction_se=f_se,
                         richest_per_N=mu*f.mean(),bulk_per_N=mu*(1-f.mean()),
                         partition_se=mu*f_se,n_reps=len(p)))
    return pd.DataFrame(rows)


def high_wealth_diagnostics(summaries,specs,outdir):
    """Independent recomputation of Fig. 4F slope and equilibration diagnostics."""
    if not all(any(abs(s['mu']-m)<1e-10 for s in summaries) for m in (6.,8.,12.)):
        return
    import math
    rows=[]; checks=[]
    for mu in (6.,8.,12.):
        rs=[]
        for spec in specs:
            if abs(spec['M']/spec['N']-mu)>1e-10: continue
            d=Path(spec['output_directory'])
            summary=json.loads((d/'summary.json').read_text())
            block=pd.read_csv(d/'block_metrics.csv')
            prod=block.loc[block['phase']=='production']
            if len(prod)==0:raise ValueError('No production blocks for '+spec['run_id'])
            f=prod['max_fraction'].to_numpy(float)
            if not np.isclose(f.mean(),summary['max_fraction_mean'],rtol=1e-10,atol=1e-12):
                raise ValueError('Production block mismatch: '+spec['run_id'])
            half=len(f)//2
            rs.append(dict(mu=mu,initial=spec['initial'],
                bulk_per_N=mu*(1-f.mean()),richest_per_N=mu*f.mean(),
                half_change=mu*(f[:half].mean()-f[half:].mean())))
        if len(rs)<2:continue
        df=pd.DataFrame(rs)
        bulk=df['bulk_per_N'].to_numpy(float)
        se=bulk.std(ddof=1)/math.sqrt(len(bulk))
        rows.append(dict(mean_wealth=mu,richest_per_N=float(df['richest_per_N'].mean()),
                         bulk_per_N=float(bulk.mean()),partition_se=float(se),
                         n_reps=len(df)))
        a=df.loc[df.initial=='exponential','bulk_per_N'].to_numpy(float)
        b=df.loc[df.initial=='precondensed','bulk_per_N'].to_numpy(float)
        if len(a)>1 and len(b)>1:
            group_se=math.hypot(a.std(ddof=1)/math.sqrt(len(a)),
                                b.std(ddof=1)/math.sqrt(len(b)))
            change=df.half_change.to_numpy(float)
            half_se=change.std(ddof=1)/math.sqrt(len(change))
            checks.append(dict(mean_wealth=mu,group_difference=float(a.mean()-b.mean()),
                group_difference_se=float(group_se),
                group_z=float((a.mean()-b.mean())/group_se) if group_se else float('nan'),
                half_change=float(change.mean()),half_change_se=float(half_se),
                half_z=float(change.mean()/half_se) if half_se else float('nan')))
    df=pd.DataFrame(rows).sort_values('mean_wealth')
    df.to_csv(outdir/'Figure4F_highwealth_partition.csv',index=False)
    pd.DataFrame(checks).to_csv(outdir/'Figure4F_convergence_checks.csv',index=False)
    if {6.,12.}.issubset(set(df.mean_wealth)):
        lo=df.loc[np.isclose(df.mean_wealth,6.)].iloc[0]
        hi=df.loc[np.isclose(df.mean_wealth,12.)].iloc[0]
        val=(hi.richest_per_N-lo.richest_per_N)/6.
        err=math.hypot(hi.partition_se,lo.partition_se)/6.
        result={'mean_from':6,'mean_to':12,'fraction_added_wealth_to_richest':float(val),
                'standard_error':float(err),'percentage':100*float(val),
                'percentage_standard_error':100*float(err),
                'note':'Means and SEM are computed over independent production-run averages.'}
        (outdir/'Figure4F_wealth_capture.json').write_text(json.dumps(result,indent=2))

def run_ensemble(args):
    key='EF_historical_specs.json' if args.panel=='EF' else 'D_code_supported_specs.json'
    specs=sorted(json.loads((ROOT/'configs'/key).read_text()),key=lambda s:(s['M']/s['N'],s['seed']))
    if args.means:
        specs=[s for s in specs if any(abs(s['M']/s['N']-x)<.001 for x in args.means)]
    if args.profile=='smoke' and not args.means:
        mu=[min(s['M']/s['N'] for s in specs),max(s['M']/s['N'] for s in specs)]
        specs=[s for s in specs if s['M']/s['N'] in mu]
    reps=args.reps or (2 if args.profile=='smoke' else (12 if args.panel=='EF' else 4))
    if reps>(12 if args.panel=='EF' else 4):raise ValueError('Requested repetitions exceed recovered historical seed set')
    keep=[];counts={}
    for s in specs:
        mu=s['M']/s['N'];counts[mu]=counts.get(mu,0)+1
        if counts[mu]>reps:continue
        s=s.copy()
        if args.profile=='smoke':
            s.update(burnin_sweeps=10000,production_sweeps=10000,block_sweeps=10000,checkpoint_every_blocks=1)
            s['notes']='SMOKE TEST ONLY: shortened schedule, not publication data.'
        s['output_directory']=str((args.outdir/'runs'/s['run_id']).resolve()); keep.append(s)
    if not keep:raise ValueError('No configured mean matches request')
    if args.jobs==1:summaries=[one_run(s) for s in keep]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:summaries=list(pool.map(one_run,keep))
    aggregate(summaries).to_csv(args.outdir/(args.panel+'_summary.csv'),index=False)
    if args.panel=='EF' and args.profile=='publication':
        high_wealth_diagnostics(summaries,keep,args.outdir)
    report={'panel':args.panel,'profile':args.profile,'runs':len(keep),'conserved_wealth':True,'nonnegative_wealth':True,'historical_configuration':args.panel=='EF' and args.profile=='publication','seed_set':('recovered low-wealth EF seeds and Figure4F_wealth_partition historical high-wealth seeds' if args.panel=='EF' else 'historical generator, original raw runs unavailable'),'note':'All numerical tables and plots are computed from newly run simulations; Monte Carlo values may depend on software/Numba versions.'}
    (args.outdir/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def run_ab(args):
    """Fresh A/B trajectory with early logarithmic samples and one continuous RNG stream.

    One Numba call advances the Markov chain from t=0 to the final sweep; adding
    or removing observation times does not reseed or otherwise alter the chain.
    """
    from mc_kernel import _run_step_ab_recorded
    N, M = 300, 1500
    end = 10000 if args.profile == 'smoke' else 3000000
    # Regular monitoring throughout plus dense logarithmic early-time sampling.
    # These times denote complete sweeps (N/2 pair interactions per sweep).
    powers = {2**k for k in range(end.bit_length()) if 2**k <= end}
    times = np.array(sorted(set(range(0, end+1, 1000)) | powers | {0, end}), dtype=np.int64)
    w = np.full(N, 5, dtype=np.int64)
    order = np.arange(N, dtype=np.int64)
    snapshots = _run_step_ab_recorded(w, order, times, 0.25, int(args.seed))
    assert int(w.sum()) == M and bool(np.all(w >= 0))
    # The identities in panel A are selected according to their wealth at the end.
    ids = np.argsort(w)[::-1]
    cols = {'final richest': int(ids[0]), 'final second-richest': int(ids[1])}
    ordinary = [i for i in range(N) if i not in (int(ids[0]), int(ids[1]))][:8]
    cols.update({f'ordinary {j+1}': int(i) for j, i in enumerate(ordinary)})
    # Panel A: powers of two at early times, then every 10,000 sweeps.
    selected = [j for j, t in enumerate(times) if t == 0 or t == end or
                (t > 0 and (int(t) & (int(t)-1)) == 0) or t % 10000 == 0]
    trajectories = [dict(time_sweeps=int(times[j]),
                         **{name: int(snapshots[j, ix]) for name, ix in cols.items()})
                    for j in selected]
    pd.DataFrame(trajectories).to_csv(args.outdir/'selected_trajectories_early_logsample.csv', index=False)
    # Panel B: early logarithmic observations and regular 1,000-sweep samples.
    obs = []
    for t, state in zip(times, snapshots):
        top = np.sort(state)[-2:]
        obs.append(dict(time_sweeps=int(t), m1_instantaneous=int(top[-1]),
                        m2_instantaneous=int(top[-2]), p0=float(np.mean(state == 0))))
    pd.DataFrame(obs).to_csv(args.outdir/'global_observables.csv', index=False)
    np.savez_compressed(args.outdir/'final_state.npz', wealth=w)
    report = {'panel':'AB','profile':args.profile,'N':N,'M':M,
              'initial':'equal wealth 5 per agent','seed':int(args.seed),
              'eta_infinity':0.25,'bias':'step','last_sweep':end,
              'selected_identity_indices':cols,'conserved_wealth':True,
              'nonnegative_wealth':True,'continuous_rng':True,
              'early_time_samples':[int(t) for t in times if t > 0 and t < 1000],
              'trajectory_samples':len(selected),'global_observation_samples':len(obs),
              'historical_reconstruction':False,
              'note':'Fresh seeded realization. A single seeded Numba kernel simulates '
                     'the continuous trajectory; sampling does not restart the RNG.'}
    (args.outdir/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

def run_c(args):
    from c_dense import run_dense
    from mc_kernel import _advance_builtin
    from publication_runner import Spec,_initialize
    reps=args.reps or (1 if args.profile=='smoke' else 4)
    if reps>4:raise ValueError('Historic C seed formula archived for four replicates')
    burn,prod,stride,traj=(1000,2000,50,500) if args.profile=='smoke' else (300000,1200000,500,5000)
    ps=[];bs=[];ds=[];background_records=[]
    for rep in range(1,reps+1):
        seed=args.seed+rep
        spec=Spec(run_id=f'new_background_r{rep}',study='new_C_background',N=300,M=636,seed=seed)
        w=_initialize(spec);order=np.arange(300,dtype=np.int64)
        bg_burn=1000 if args.profile=='smoke' else 1000000
        _advance_builtin(w,order,bg_burn,0,.25,1.,seed)
        background_records.append({'new_seed':seed,'new_background_burn_sweeps':bg_burn,'initialization':'exponential, exact M636'})
        if len(w)!=300 or int(w.sum())!=636:raise ValueError('C background must have N300, M636')
        nz=np.flatnonzero(w==0)
        if not len(nz):raise ValueError('C background has no zero site')
        tagged=int(nz[0]);w[tagged]+=573
        seed=99173+rep*104729
        w,Ht,Hl,Hd,Hb,Hm,tr=run_dense(w,tagged,seed,burn,prod,stride,traj,.25,1209)
        assert int(w.sum())==1209 and np.all(w>=0)
        np.savez_compressed(args.outdir/f'histograms_r{rep}.npz',Htot=Ht,Hbulk_max_removed=Hb,Hmax=Hm,wealth=w)
        ps.append(Ht/Ht.sum());bs.append(Hb/Hb.sum());ds.append(Hm/Hm.sum())
    p=np.mean(ps,axis=0);b=np.mean(bs,axis=0);d=np.mean(ds,axis=0)
    pd.DataFrame(dict(m=np.arange(len(p)),p_total_mean=p,B_liquid_mean=b,p_liquid_population_mean=299/300*b,p_dragon_population_mean=d/300)).to_csv(args.outdir/'stationary_distribution_dense.csv',index=False)
    report={'panel':'C','profile':args.profile,'N':300,'M':1209,'reps':reps,'burnin':burn,'production':prod,'stride':stride,'seeds':[99173+r*104729 for r in range(1,reps+1)],'backgrounds':background_records,'conserved_wealth':True,'nonnegative_wealth':True,'historical_reconstruction':False,'normalization_checks':{'total':float(sum(p)),'bulk_population':float(sum(b)*299/300),'outlier_population':float(sum(d)/300)},'note':'Without original background checkpoints, new-background mode is a statistical replication with explicitly new initial states.'}
    (args.outdir/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',choices=['smoke','publication'],default='smoke')
    ap.add_argument('--panel',choices=['EF','D','AB','C'],default='EF')
    ap.add_argument('--outdir',type=Path,default=ROOT/'new_runs/smoke')
    ap.add_argument('--jobs',type=int,default=1);ap.add_argument('--reps',type=int)
    ap.add_argument('--means',type=float,nargs='+');ap.add_argument('--seed',type=int,default=20261030)
    ap.add_argument('--new-background',action='store_true')
    args=ap.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    if args.panel in ['EF','D']:run_ensemble(args)
    elif args.panel=='AB':run_ab(args)
    else:run_c(args)
if __name__=='__main__':main()
