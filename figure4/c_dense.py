"""Exact historic C dense-accumulation kernel; original initial states are separate inputs."""
import numpy as np
from numba import njit

@njit(cache=True)
def run_dense(w, tagged, seed, burn, prod, sample_every, traj_every, eta, maxwealth):
    np.random.seed(seed); n=w.size; order=np.arange(n,dtype=np.int64)
    Htot=np.zeros(maxwealth+1,dtype=np.int64); Hliq=np.zeros(maxwealth+1,dtype=np.int64); Hdrag=np.zeros(maxwealth+1,dtype=np.int64); Hbulkmax=np.zeros(maxwealth+1,dtype=np.int64); Hmax=np.zeros(maxwealth+1,dtype=np.int64)
    ntraj=1+(burn+prod)//traj_every; traj=np.zeros((ntraj,9),dtype=np.float64); ti=0
    # initial stats helper expanded inline
    def fillrow(idx,sweep,phase):
        m1=-1; m2=-1; m3=-1; z=0
        for ii in range(n):
            x=w[ii]
            if x==0: z+=1
            if x>m1: m3=m2; m2=m1; m1=x
            elif x>m2: m3=m2; m2=x
            elif x>m3: m3=x
        traj[idx,0]=sweep; traj[idx,1]=phase; traj[idx,2]=w[tagged]; traj[idx,3]=m1; traj[idx,4]=m2; traj[idx,5]=m3
        traj[idx,6]=z/n; traj[idx,7]=z/(n-1); traj[idx,8]=(maxwealth-w[tagged])/(n-1)
    fillrow(ti,0,0); ti+=1
    total=burn+prod
    for sweep in range(1,total+1):
        np.random.shuffle(order)
        for k in range(0,n-1,2):
            i=order[k]; j=order[k+1]; d=w[i]-w[j]
            e=eta if d>0 else (-eta if d<0 else 0.0)
            if np.random.random() < 0.5+e:
                if w[j]>0: w[i]+=1; w[j]-=1
            elif w[i]>0: w[i]-=1; w[j]+=1
        if sweep % traj_every == 0:
            fillrow(ti,sweep,1 if sweep<=burn else 2); ti+=1
        if sweep>burn and (sweep-burn)%sample_every==0:
            imax=0; vmax=w[0]
            for ii in range(1,n):
                if w[ii]>vmax:
                    vmax=w[ii]; imax=ii
            for ii in range(n):
                x=w[ii]; Htot[x]+=1
                if ii!=tagged: Hliq[x]+=1
                if ii!=imax: Hbulkmax[x]+=1
            Hdrag[w[tagged]]+=1
            Hmax[vmax]+=1
    return w,Htot,Hliq,Hdrag,Hbulkmax,Hmax,traj[:ti]

