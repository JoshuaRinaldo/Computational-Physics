# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:37:06 2021

@author: joshu
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit as tm
import matplotlib

def initialize(N,p):
    spin,E,M = [1]*N,0.,0. #creates list of ones
    
    for i in range(1,N):
        if rd.random()>p: #randomly generates up and down spins according to a defined probability, p.
            spin[i] = -1 #sets ~0.6 as down
        E = E - spin[i-1]*spin[i] #calculates E and M
        M = M + spin[i]
    return spin, E-spin[N-1]*spin[0], M+spin[0]

def update(N,spin,kT,dE,E,M):
    flip = 0 #creates flip varaible. If 1 then flip, if 0 then no flip.
    if dE<0.0:
        flip = 1
    else:
        exp = np.exp(-dE/kT)
        if rd.random()<exp: #determines if there is a random flip that occurs, dependant on dE and kT
            flip = 1
    if flip==1: #if flip is yes, then we change the spin state and calculate new E and M values.
        E = E + dE
        M = M -2*spin
        spin = -spin
    return E,M,spin


def run(N,kT,p):
    E = []
    M = []
    spins, En, Mn = initialize(N,p)
    E.append(En/N) #appends initial result
    M.append(Mn)
    spinlist = np.array(spins)
    Iter = N*150 #determines number of iterations per spin position
    for iteration in range(0,Iter+1):
        i = rd.randint(0,N-1) #pick random position
        dE = 2*spins[i] * (spins[i-1] + spins[(i+1)%N]) #calculate dE
        

        En,Mn,spins[i] = update(N,spins[i],kT,dE,En,Mn) #determines if there is a flip

        spins = np.array(spins) #save values
        E.append(En/N) #saves our E and M values for later comparison
        M.append(Mn/N)
        spinlist = np.vstack((spinlist,spins))
    
    
    Beta = 1/kT
    E_an = -np.tanh(Beta) #calculates analytical E
    
    
    return spinlist, E, E_an, M #retun spins and values


E_over_kt = []
E_over_kt_an = []
M_over_kt = []
S_numerical = []
kt_list = []


#This is what I use to calculate entropy. This must be run for a considerable amount of time
#before it becomes consistent with the analytical solution.

kt_over_ep = np.linspace(0.1,6,200)

for kt in kt_over_ep:
    kt_list.append(kt)
    
    E_mc = []
    M_mc = []
    start = tm.default_timer()
    for n in range(2): #selects number of mc samples
        all_spins, E, E_an, M= run(30,kt,0.6)
    
        E = E[3000:]
        M = M[3000:] #taking away time from the approach to correct values
    
        E = sum(E)/len(E) #calculating E and M
        M = sum(M)/len(M)
        
        E_mc.append(E)
        M_mc.append(M)
    

    E = sum(E_mc)/len(E_mc)
    M = sum(M_mc)/len(M_mc) #averages over mc samples
    

    E_over_kt.append(E)
    E_over_kt_an.append(E_an)
    M_over_kt.append(M)



#plotting 1d over time. Not worth doing if you're checking mc solutions (feel
#free to uncomment and check that it works though)
#for E,M, and En
font = {'size'   : 22}
#matplotlib.rc('font', **font)
#
#print(all_spins.shape)
#all_spins = np.rot90(all_spins)
#ax = sns.heatmap(all_spins)
#ax.set_title("1D Spin States Over Time",pad=20)
#ax.set_xlabel("Trial #")
#ax.set_ylabel("Spin position")
#ax.set_xticks([0,4000,7000])
#ax.set_xticklabels(['$0$','$4\cdot 10^3$','$7\cdot 10^3$'])
#ax.collections[0].colorbar.set_label("Spin State")
#
#plt.show()
#plt.savefig("grid_over_time_1d.png",bbox_inches='tight',dpi=100)
#
#plt.pause(8)
#plt.close()





plt.plot(kt_over_ep,E_over_kt_an,color = "g", linewidth = 3,label = "Analytical")
plt.plot(kt_over_ep,E_over_kt,"k.", markersize = 5,label = "Numerical")

plt.legend()
plt.title("Monte-Carlo Approximation of Energy")
plt.xlabel("$kT/\epsilon$")
plt.ylabel("$E/N\epsilon$")
plt.show()

plt.pause(8)
plt.close()

plt.plot(kt_over_ep,np.zeros(len(kt_over_ep)),color = "g", linewidth = 3,label = "Analytical")
plt.plot(kt_over_ep,M_over_kt,"k.",markersize = 5,label = "Numerical")

plt.legend()
plt.title("Monte-Carlo Approximation of Magnetization")
plt.xlabel("$kT/\epsilon$")
plt.ylabel("$M/N$")
plt.show()

plt.pause(8)
plt.close()





#this calculates our analytical and numerical solutions
s = []
for kTn in kt_over_ep:
    sn = (np.log(2*np.cosh(1/kTn))-(1/kTn)*np.tanh(1/kTn))
    s.append(sn) #calculates analytical solution

S = np.zeros(len(E_over_kt))

for i in range(1,len(E_over_kt)-1):
    S[i] = S[i-1]+(E_over_kt[i+1]-E_over_kt[i])/kt_list[i] #calculates numerical solution

S = S[:len(S)-1]

plt.plot(kt_over_ep,s,color = "g", linewidth = 3,label = "Analytical")
plt.plot(kt_list[1:],S,"k.",markersize = 5,label = "Numerical")

plt.legend()
plt.title("Monte-Carlo Approximation of Entropy")
plt.xlabel("$kT/\epsilon$")
plt.ylabel("$S/Nk$")
plt.show()
