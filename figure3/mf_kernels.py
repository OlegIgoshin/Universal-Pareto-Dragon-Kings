from __future__ import annotations
import numpy as np
PC=2.0/3.0
KAPPA=0.5
def bias(x,c,scale,rule):
    if rule=='step': return c*np.sign(x)
    if rule=='tanh': return c*np.tanh(x/scale)
    n={'hill2':2,'hill4':4}[rule]
    y=np.abs(x)/scale
    return c*np.sign(x)*(1-1/(1+y**n))


def solve_fixed(p0,c,scale,rule,mmax,tol,maxiter):
    from scipy.fft import rfft,irfft,next_fast_len
    m=np.arange(mmax+1,dtype=float)
    # Critical-tail start; no outcome is assumed converged until diagnostics pass.
    p=np.empty(mmax+1); p[0]=p0; p[1:]=1/(m[1:]+3)**2
    p[1:]*=(1-p0)/p[1:].sum()
    eta=bias(m,c,scale,rule)
    size=next_fast_len(3*mmax+1)
    kernel_fft=rfft(bias(np.arange(-mmax,mmax+1,dtype=float),c,scale,rule),size)
    tiny=np.finfo(float).tiny
    change=float('inf')
    for it in range(maxiter):
        avg=irfft(rfft(p,size)*kernel_fft,size)[mmax:2*mmax+1]
        gain=np.clip(.5*(1-p0)+avg-p0*eta,tiny,1)
        loss=np.clip(.5-avg,tiny,1)
        logs=np.cumsum(np.log(gain[:-1])-np.log(loss[1:])); logs-=logs.max()
        w=np.exp(logs); q=np.r_[p0,(1-p0)*w/w.sum()]
        new=.8*p+.2*q; change=float(np.abs(new-p).sum()); p=new
        if change<tol: break
    avg=irfft(rfft(p,size)*kernel_fft,size)[mmax:2*mmax+1]
    gain=.5*(1-p0)+avg-p0*eta; loss=.5-avg
    flux=p[:-1]*gain[:-1]-p[1:]*loss[1:]
    stats={'rule':rule,'mmax':mmax,'p0':p0,'delta':(p0-PC)/(1-PC),
        'iterations':it+1,'L1_change':change,'converged_L1':change<tol,
        'max_abs_balance_flux':float(np.max(np.abs(flux))),
        'normalization':float(p.sum()),'mean':float(m@p),
        'endpoint_p':float(p[-1]),'endpoint_compensation':float(m[-1]**2*p[-1]/KAPPA),
        'assumed_scale':scale,'tail_grid_warning':'L1 convergence alone does not establish tail/grid convergence'}
    return p,stats


