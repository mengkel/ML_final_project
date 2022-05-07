#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:38:42 2022

@author: mengkel
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt




def cal_mse(A, Y, beta):
    y_hat = A.dot(beta)
    mse = np.square(y_hat-Y).mean()
    return mse

def cal_beta(A, Y):
    beta = (np.linalg.pinv((A.T).dot(A)+0.01*np.identity(8)).dot(A.T)).dot(Y)
    return beta

def std_data(X):
    X_std = stats.zscore(X, axis = 0)
    b = np.ones((len(X),1))
    X_new = np.concatenate((b,X_std), axis = 1)
    return X_new

def SEMF(Z, N):
    aV, aS, aC, aA, delta = 15.75, 17.8, 0.711, 23.7, 11.18
    Z, N = np.atleast_1d(Z), np.atleast_1d(N)
    # Total number of nucleons
    A = Z + N
    sgn = np.zeros(len(Z))
    shell = np.zeros(len(Z))
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
    return E

data = [line.strip('\n') for line in open('mass16.txt')][200:]
c = 2.998*10**8
sep_n = [6,8]
sep_z = [11,13]
sep_a = [16,19]
sep_me = [29,40]
sep_mass = [96, 112]
sep_be = [54, 63]

N = [int(line[sep_n[0]:sep_n[1]+1]) for line in data]
Z = [int(line[sep_z[0]:sep_z[1]+1]) for line in data]
A = np.array([int(line[sep_a[0]:sep_a[1]+1]) for line in data])
BE = np.array([float(line[sep_be[0]:sep_be[1]+1].replace('#', '')) for line in data])

ME = np.array([float(line[sep_me[0]:sep_me[1]+1].replace('#', '')) for line in data]) # in kev
MN = np.array([float(line[sep_mass[0]:sep_mass[1]+1].replace(' ','').replace('#', '')) for line in data])
MN = MN*10**(-6) #mass number in u
unit = 931.505 #convert u to MeV/c^2
MASS = MN*unit
Mass_excess = (MN - A) * unit


Z, N = np.atleast_1d(Z), np.atleast_1d(N)
sgn = np.zeros(len(Z))
shell = np.zeros(len(Z))

for i in range(len(Z)):
    if (Z[i]%2==0) and (N[i]%2==0):
        sgn[i] = 1
    if (Z[i]%2==1) and (N[i]%2==1):
        sgn[i] = -1
    if (50 <= Z[i] <= 82) or (50 <= N[i] <= 82):
        shell[i] = -1
    if (80 <= Z[i] <= 130) or (82 <= N[i] <= 130):
        shell[i] = -2
    if (Z[i] >= 130) or (N[i] >=130):
        shell[i] = -3
        
X = np.array([Z, N, A, A*A**(-1/3), A*Z**2/A**(4/3), (A-2*Z)**2/A,  A * sgn / A**(3/2)]).T
X = std_data(X)
y = MASS
beta = cal_beta(X,y)
y_pre = X.dot(beta)
#M = cal_mse(x,y, beta)


'''
# training set 
Ntrain = np.array([1900, 2100, 2300, 2500, 2700, 2900, 3000, 3100, 3200, 3300, 3400])
MSE = np.zeros([len(Ntrain), 2])  
for n in range(len(Ntrain)):
    Xtrain = X[0:Ntrain[n], :]; Ytrain = y[0:Ntrain[n]]
    Xtest = X[Ntrain[n]:,:]; Ytest = y[Ntrain[n]:]
    beta = cal_beta(Xtrain, Ytrain)    
    MSE[n,0] = cal_mse(Xtrain,Ytrain, beta)
    MSE[n,1] = cal_mse(Xtest, Ytest, beta)
print(MSE)
plt.plot(Ntrain, MSE[:,0], marker = '*', label = 'train')
plt.plot(Ntrain, MSE[:,1], marker = '*', label = 'test')
plt.xlabel('size of training set')
plt.ylabel('MSE (MeV)')
plt.legend()
#plt.savefig('MSE_mass_0424.png', dpi = 300)
plt.show()
'''



#'''
mn = MASS[0]
mp = MASS[1]

M = np.zeros([max(Z)+1, max(N)+1])
M2 = np.zeros([max(Z)+1, max(N)+1])
BE_term=[]
for i in range(len(Z)):
    M[Z[i],N[i]] = MASS[i]
    M2[Z[i], N[i]] = y_pre[i]
#    M2[Z[i], N[i]] = mn*N[i] + mp*Z[i]- A[i]*BE[i]/1000
    
diff = abs(M2 - M)
masked = np.ma.masked_where(diff==0,diff)  
fig = plt.figure(figsize = (7.1,4))
plt.imshow(masked, origin='lower', cmap='OrRd')
#plt.title('Mass Discrepancy (MeV)')
plt.xlabel('Number of Neutrons')
plt.ylabel('Number of Protons (Z)')
plt.colorbar()
plt.tight_layout()
plt.ylim(18, 130)
plt.xlim(18, 180)
plt.savefig('MASS_RR.png', dpi = 300)
#'''

'''
# PLot Lasso
#Nt = 3400 
lambd = np.logspace(-3,3,200)
#xtrain = stats.zscore(X[:Nt, :], axis = 0);  
b = np.ones([len(y), 1])
xtrain = np.concatenate((b,X), axis = 1)
ytrain = y
MSE_lasso = np.zeros([xtrain.shape[1], len(lambd)])

#color=cm.rainbow(np.linspace(0,1,13))

for i in range(len(lambd)):
    reg = sl.Lasso(alpha=lambd[i])
    reg.fit(xtrain, ytrain)
    MSE_lasso[:,i] = reg.coef_

for i in range(len(MSE_lasso)):
    plt.plot(lambd, MSE_lasso[i, :], label = r'$\beta$' + str(i))
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('Standarized Coefficients')
plt.axis('tight')
plt.legend(loc = 'upper right')
'''















