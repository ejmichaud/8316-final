
import os
import itertools
import h5py
import argparse

import random
import numpy as np
from numpy.random import rand
from numba import jit
from tqdm import tqdm
import torch

def initialize(N):   
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

@jit(nopython=True)
def hamiltonian(iArr):
    """Assumes J = 1.0"""
    return -1 * np.sum(iArr * (np.concatenate((iArr[:, 1:], iArr[:, :1]), axis=1) \
                               + np.concatenate((iArr[1:, :], iArr[:1, :]), axis=0)))

def flip(i,j,iArr,Beta):
    """Flips the spin at site (i,j) in the array iArr.
    The temperature is given by Beta = 1/kT."""
    N = iArr.shape[0]
    top = iArr[(i-1)%N, j]
    bottom = iArr[(i+1)%N, j]
    left = iArr[i, (j-1)%N]
    right = iArr[i, (j+1)%N]
    dE = 2 * iArr[i, j] * (top + bottom + left + right)
    # if dE > 0 and the Metropolis condition does not pass
    if dE > 0 and np.random.rand() > np.exp(-Beta * dE):
        # do not flip the spin
        return iArr
    else:
        # flip the spin
        iArr[i, j] *= -1
        return iArr

def mag(iArr):
    return np.sum(iArr)

def update_rand(iArr,N,TM1):
    """Updates the array iArr using the Metropolis algorithm.
    The temperature is given by Beta = 1/kT."""
    indices = list(itertools.product(range(N), repeat=2))
    np.random.shuffle(indices)
    for i,j in indices:
        iArr = flip(i,j,iArr,TM1)
    return iArr

# def runTemp(iT,iN,images,fig,ax,eqSteps=500,mcSteps=500):
#     pArr = initialize(iN)         # initialise
#     #initial variables? 
#     beta=1.0/iT 
#     for i in range(eqSteps):         # equilibrate
#         update_rand(pArr, iN, beta)   
    
#     energies = []
#     magnetizations = []
#     for i in range(mcSteps):
#         update_rand(pArr, iN, beta)           
#         Ene = hamiltonian(pArr, iN)     # calculate the energy
#         Mag = mag(pArr)        # calculate the magnetisation
#         energies.append(Ene)
#         magnetizations.append(Mag)

#     #compute the values for E,M,C,X here
#     E = np.mean(energies)
#     M = np.mean(magnetizations)
#     C = np.var(energies)/(iT**2)
#     X = np.var(magnetizations)/iT
#     return E,M,C,X

class Ising():
    def __init__(self, iN, Temp):
        self.N   = iN
        self.T   = Temp
        self.arr = self.initialize()
        self.steps = 300
        #History over simulatinp
        self.E   = np.array([])
        self.M   = np.array([])
        self.C   = np.array([])
        self.X   = np.array([])
        self.nsim = 1000
        
    def initialize(self):   
        return initialize(self.N)
    
    def simulate(self):
        beta = 1./self.T
        for i in range(self.steps):
            update_rand(self.arr, self.N, beta)           
            Ene = hamiltonian(self.arr)
            Mag = mag(self.arr)
            #Now save energy magnetization 
            self.E   = np.append(self.E,Ene)
            self.M   = np.append(self.M,Mag)
            #Now COMPUTE specific Heat and Magnetic suscpetilibity
            #HINT, consider what the meaning of RMS of Energy and Magnetization are
            #Perhaps consider a sliding window over the last hundred steps
            pC  = np.var(self.E[-100:]) / (self.T**2)
            pX  = np.var(self.M[-100:]) / self.T
            self.C   = np.append(self.C,pC)
            self.X   = np.append(self.X,pX)

    def simulate_save(self, path, verbose=False):
        filename = os.path.join(path, f'{self.N}_{self.T}_{self.nsim}.pt')
        data = torch.empty((self.nsim,self.N,self.N), dtype=torch.int)
        mags = torch.empty(self.nsim, dtype=torch.float)
        TM1  = 1./self.T
        for n in tqdm(range(self.nsim), disable=not verbose):
            self.initialize() 
            self.simulate()
            pMag = mag(self.arr)
            data[n,:,:] = torch.tensor(self.arr)
            mags[n] = pMag
        torch.save({'data': data, 'mag': mags}, filename)

    def lastAvg(self):
        avgE = np.mean(self.E[500:-1])
        avgM = np.mean(self.M[500:-1])
        avgC = np.std(self.E[500:-1])
        avgX = np.std(self.M[500:-1])
        return avgE,avgM,avgC,avgX
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ising model simulation')
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='temperature of the simulation')
    parser.add_argument('--size', type=int, default=32,
                        help='size of the lattice')
    parser.add_argument('--nsim', type=int, default=2000,
                        help='number of simulations')
    parser.add_argument('--save_dir', type=str, default='data',
                        help='directory to save the data (assumes existence)')
    parser.add_argument('--verbose', action='store_true',
                        help='verbosity flag')
    
    args = parser.parse_args()
    test = Ising(args.size, args.temperature)
    test.nsim=args.nsim
    test.simulate_save(args.save_dir, args.verbose)

