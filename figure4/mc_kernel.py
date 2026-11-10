"""Exact step/tanh/arctan update kernel copied from the audited original runner."""
import math
import numpy as np
from numba import njit
NUMBA_AVAILABLE=True
@njit(cache=True)
def _advance_builtin(wealth,order,sweeps,kind,c,x0,seed):
    np.random.seed(seed)
    n=wealth.size
    for _ in range(sweeps):
        np.random.shuffle(order)
        for k in range(0,n-1,2):
            i=order[k];j=order[k+1];difference=wealth[i]-wealth[j]
            if kind==0:eta=c if difference>0 else (-c if difference<0 else 0.0)
            elif kind==1:eta=c*math.tanh(difference/x0)
            else:eta=(2.0*c/math.pi)*math.atan(difference/x0)
            if np.random.random()<0.5+eta:
                if wealth[j]>0:wealth[i]+=1;wealth[j]-=1
            elif wealth[i]>0:wealth[i]-=1;wealth[j]+=1


@njit(cache=True)
def _run_step_ab_recorded(wealth, order, observation_times, eta_infinity, seed):
    """Sample one continuous seeded trajectory at arbitrary complete sweep times.

    The sampling grid is read-only and does not affect random draws. All random
    choices use the same NumPy/Numba generator, initialized exactly once.
    """
    np.random.seed(seed)
    n = wealth.size
    saved = np.empty((observation_times.size, n), dtype=np.int64)
    last_t = 0
    for sample in range(observation_times.size):
        t = observation_times[sample]
        for sweep in range(last_t, t):
            np.random.shuffle(order)
            for k in range(0, n-1, 2):
                i = order[k]
                j = order[k+1]
                d = wealth[i]-wealth[j]
                eta = eta_infinity if d>0 else (-eta_infinity if d<0 else 0.0)
                if np.random.random() < 0.5 + eta:
                    if wealth[j]>0:
                        wealth[i]+=1
                        wealth[j]-=1
                elif wealth[i]>0:
                    wealth[i]-=1
                    wealth[j]+=1
        saved[sample,:] = wealth
        last_t = t
    return saved
