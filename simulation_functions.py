import numpy as np
import matplotlib.pyplot as plt

def calc_dndt(k,K,n,lam):
    return np.matmul(k*(np.outer(np.ones(len(n)),n)+ K)**(-1),n) - lam*n

def calc_J(k,K,n,lam):
    return k*K/(K**2 + 2*K*(np.outer(np.ones(len(n)),n)) + (np.outer(np.ones(len(n)),n))**2) - lam*np.identity(len(n))

def calc_IPR(w,v):
        IPR = np.sum((v[np.argsort(w)][-1] * np.conjugate(v[np.argsort(w)][-1]))**2)
        return IPR

def simulate_dynamics(k,K,lam,n0,thresh = 0.0001, cutoff = 500000, dt = 0.01):
    n = np.array(n0)
    g = k.shape[0]
    nt = np.zeros((g,cutoff))
    for i in range(cutoff):
        n = n + calc_dndt(k,K,n,lam)*dt
        nt[:,i] = n
        if i > 0 and np.max(np.abs((nt[:,i] - nt[:,i-1])/nt[:,i])) < thresh:
            nt = nt[:,:i+1]
            break

    return nt