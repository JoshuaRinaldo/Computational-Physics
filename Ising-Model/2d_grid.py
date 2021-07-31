# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:37:06 2021

@author: joshu
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

font = {'size'   : 18}
matplotlib.rc('font', **font)

#Here we visualize a simulation of the ferromagnetic Ising model. The code makes 
#use of monte carlo simulation to determine if the spin state of a dipole will
#'flip' from one spin state to another. The figures demonstrate an evolution
#towards chaos.


def initialize(N,p):
    spin,E,M = np.ones((N,N)),0.,0. #creates 2d array of ones
    
    for ix in range(1,N):
        for iy in range(1,N):
            if rd.random()>p:
                spin[ix][iy] = -1 #sets ~0.6 as down
            E = E - spin[ix-1][iy]*spin[ix][iy]- spin[ix][iy-1]*spin[ix][iy] #calculates E and M
            M = M + spin[ix][iy]
    return spin, E-spin[N-1]*spin[0], M+spin[0]

def update(N,spin,kT,dE,E,M):
    flip = 0 #creates flip variable 0 is no flip 1 is flip
    if dE<0.0:
        flip = 1
    else:
        if kT==0:
            flip = 0 #if kT==0 rd.random will certainly be greater than exp
        else:
            exp = np.exp(-dE/kT)
            if rd.random()<exp: #this is the monte-carlo part of the simulation
                flip = 1
    if flip==1:  #checks if flip is 1. If flip is one, we change the spin state.
        E = E + dE
        M = M -2*spin
        spin = -spin
    return E,M,spin


#N is the number length of the grid we are analyzing.
#kT is the tempurature of the system. Higher kT = faster tendency towards chaos
#p is the probability of the spin states being negative. This sets the initial
#condition of the grid.
#shots are the steps that we would like to visualize the grid on.
def run(N,kT,p,shots):
    #creates our grid
    spins, En, Mn = initialize(N,p)
    
    #This is a fine number of iterations for visualization, but it can be changed
    #to simulate a shorter or longer time-evolution.
    Iter = N*500
    sns.heatmap(spins)
    for iteration in range(0,Iter+1):
        #randomly selects the grid position we consider flipping the spin of
        ix = rd.randint(0,N-1)
        iy = rd.randint(0,N-1)
        #calculates dE, which is dependent upon the adjacent spin states
        dE = 2*spins[ix][iy] * (spins[ix-1][iy] + spins[(ix+1)%N][iy])* (spins[ix][iy-1] + spins[ix][(iy+1)%N])
        #calculates dE
         
        #figure out if we need to change the spins 
        En,Mn,spins[ix][iy] = update(N,spins[ix][iy],kT,dE,En,Mn)
        #plots/saves figure
        if iteration in shots:
            plt.close()
            ax = sns.heatmap(spins)
            title = "2D Spin States at Iteration {}".format(iteration)
            ax.set_title(title,pad=20)
            ax.set_xlabel("Spin Position")
            ax.set_ylabel("Spin Position")
            ax.collections[0].colorbar.set_label("Spin State")
            plt.tight_layout()
            plt.show()
            #plt.savefig(title+".png",bbox_inches='tight',dpi=100)
            
            #pause for viewing time
            plt.pause(5)
    


#this is one of my personal favourite time-evolutions
#I think it does a good job of showing the tendancy
#towards chaos
run(50,1,0.995,[500,5000,20000])

