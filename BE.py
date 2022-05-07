#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:40:10 2022

@author: mengkel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = [line.strip('\n') for line in open('mass16.txt')][39:]
sep_n = [6,8]
sep_z = [11,13]
sep_be = [54, 63]

N = [int(line[sep_n[0]:sep_n[1]+1]) for line in data]
Z = [int(line[sep_z[0]:sep_z[1]+1]) for line in data]
A = [N[i]+Z[i] for i in range(len(N))]
BE = np.array([float(line[sep_be[0]:sep_be[1]+1].replace('#', '')) for line in data]) # binding energy in MeV

#T = np.unique(A)
#plt.plot(np.unique(G[:,0]), BE_max)

def Uniq_BE(A, BE):
    G = np.array([np.array(A), BE]).T
    S =  np.unique(A, return_index = True)
    A = S[0]
    G_new = np.split(G[:,1], (S[1])[1:])
    BE_max = [np.array(G_new[i]).max() for i in range(len(G_new))]
    return A, BE_max


def SEMF(Z, N):
    aV, aS, aC, aA, delta = 15.75, 17.8, 0.711, 23.7, 11.18
    Z, N = np.atleast_1d(Z), np.atleast_1d(N)
    # Total number of nucleons
    A = Z + N
    sgn = np.zeros(len(Z))
    for i in range(len(Z)):
        if (Z[i]%2==0) and (N[i]%2==0):
            sgn[i] = 1
        if (Z[i]%2==1) and (N[i]%2==1):
            sgn[i] = -1
    
    # The SEMF for the average binding energy per nucleon.
    E = (aV - aS / A**(1/3) - aC * Z**2 / A**(4/3) -
         aA * (A-2*Z)**2/A**2 + sgn * delta/A**(3/2))
    if Z.shape[0] == 1:
        return float(E)
    return E, sgn

def cal_mse(A, Y, beta):
    y_hat = A.dot(beta)
    mse = np.square(y_hat-Y).mean()
    return mse

def cal_beta(A, Y):
    beta = (np.linalg.pinv((A.T).dot(A)).dot(A.T)).dot(Y)
    return beta

def std_data(X):
    X_std = stats.zscore(X, axis = 0)
    b = np.ones((len(X),1))
    X_new = np.concatenate((b,X_std), axis = 1)
    return X_new

sgn = SEMF(Z, N)

'''
#plot BE nuclear chart with Ridge regression
Z, N = np.atleast_1d(Z), np.atleast_1d(N)
A = Z + N
sgn = np.zeros(np.array(Z).shape)
sgn[(Z%2) & (N%2)] = -1
sgn[~(Z%2) & ~(N%2)] = +1
X = np.array([A**(-1/3), Z**2/A**(4/3), (A-2**Z)**2/A**2,  sgn / A**(3/2)]).T
X_new = std_data(X)
Y = BE
beta = cal_beta(X_new, Y)
Y_hat = X_new.dot(beta)

M = np.zeros([max(Z)+1, max(N)+1])
M2 = np.zeros([max(Z)+1, max(N)+1])
BE_pre = Y_hat
for i in range(len(Z)):
    M[Z[i],N[i]] = BE[i]
    M2[Z[i], N[i]] = BE_pre[i]
    
D = (M - M2)
diff = np.where((D <=-850)|(D >= 940), 0, D)
masked = np.ma.masked_where(diff==0,diff) 
fig = plt.figure(figsize = (8.1,4)) 
plt.imshow(masked,origin='lower')
plt.title('Binding Energy Discrepancy(keV)')
plt.xlabel('Number of Neutrons')
plt.ylabel('Number of Protons (Z)')
plt.tight_layout()

plt.ylim(20,90)
plt.xlim(15,140)
plt.colorbar()
plt.savefig('BE_LR.png', dpi = 300)
#G, BE_exp = Uniq_BE(A, BE)  
#Y_hat = Uniq_BE(A, Y_hat)[1]  
#plt.plot(G, BE_exp)
#plt.plot(G, Y_hat)
'''

'''
# PLot Lasso
Nt = 3000 
lambd = np.logspace(-5,1,100)
#xtrain = stats.zscore(X[:Nt, :], axis = 0);  
xtrain = X_new[:Nt, :]
ytrain = Y[:Nt]
MSE_lasso = np.zeros([xtrain.shape[1], len(lambd)])

#color=cm.rainbow(np.linspace(0,1,13))

for i in range(len(lambd)):
    reg = sl.Lasso(alpha=lambd[i])
    reg.fit(xtrain, ytrain)
    MSE_lasso[:,i] = reg.coef_

for i in range(len(MSE_lasso)):
    plt.plot(lambd, MSE_lasso[i, :], label = r'$\beta$' + str(i+1))
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('Standarized Coefficients')
plt.axis('tight')
plt.legend(loc = 'upper right')
'''


    
    
'''
#plot Max BE AME vs A
G, BE_exp = Uniq_BE(A, BE)  
BE_pre = np.array(SEMF(Z,N))
BE_theo = Uniq_BE(A, BE_pre)[1]
plt.plot(G, BE_exp, label = 'Mass-2016')
plt.plot(G, BE_theo, label = 'FRDM')
plt.legend()
plt.ylim(7.0,9.0)
plt.xlim(0, 300)
plt.xlabel('A = N + Z')
plt.ylabel('E (MeV)')
'''    
    
'''   
#plot BE in nuclear chart (all in KeV)

M = np.zeros([max(Z)+1, max(N)+1])
M2 = np.zeros([max(Z)+1, max(N)+1])
BE_pre = np.array(SEMF(Z,N))
for i in range(len(Z)):
    M[Z[i],N[i]] = BE[i]
    M2[Z[i], N[i]] = BE_pre[i]*1000
    
D = (M - M2)
diff = np.where((D <=-85)|(D >= 150), 0, D)
masked = np.ma.masked_where(diff==0,diff) 
fig = plt.figure(figsize = (8.1,4)) 
plt.imshow(masked,origin='lower')
plt.ylim(20,90)
plt.xlim(15,140)
cbar = plt.colorbar()
cbar.set_label('Binding Energy per nucleon (keV)')
plt.title('BInding Energy Discrepancy')
plt.xlabel('Number of Neutrons')
plt.ylabel('Number of Protons (Z)')
plt.tight_layout()
plt.savefig('BE_SEMF.png', dpi = 300)
'''    