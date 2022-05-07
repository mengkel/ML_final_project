#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:50:03 2022

@author: mengkel
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.linear_model as sl
from sklearn.metrics import r2_score

def cal_pair(Z, N):
    Z, N = np.atleast_1d(Z), np.atleast_1d(N)
    A = Z+N
    sgn = np.zeros(len(Z))
    for i in range(len(Z)):
        if (Z[i]%2==0) and (N[i]%2==0):
            sgn[i] = 1
        if (Z[i]%2==1) and (N[i]%2==1):
            sgn[i] = -1
    return A, sgn

def std_data(X):
    X_std = stats.zscore(X, axis = 0)
    b = np.ones((len(X),1))
    X_new = np.concatenate((b,X_std), axis = 1)
    return X_new

def cal_beta_r(l, A, Y):
    beta = (np.linalg.pinv((A.T).dot(A)+l*np.identity(8)).dot(A.T)).dot(Y)
    return beta

def cal_mse(A, Y, beta):
    y_hat = A.dot(beta)
    mse = np.square(y_hat-Y).mean()
    return mse

data = [line.strip('\n') for line in open('mass16.txt')][39:]
sep_n = [6,8]
sep_z = [11,13]
sep_me = [29,40]
sep_mass = [96, 112]

N = [int(line[sep_n[0]:sep_n[1]+1]) for line in data]
Z = [int(line[sep_z[0]:sep_z[1]+1]) for line in data]
ME = np.array([float(line[sep_me[0]:sep_me[1]+1].replace('#', '')) for line in data]) # in kev
MN = np.array([float(line[sep_mass[0]:sep_mass[1]+1].replace(' ','').replace('#', '')) for line in data])
MN = MN*10**(-6) #mass number in u
unit = 931.505 #convert u to MeV/c^2
MASS = MN*unit
mn = MASS[0]
mp = MASS[1]
A, sgn = cal_pair(Z, N) 
Mass_excess = (MN - A) * unit


def get_exp(ele, Z, N, ME):
    IND = [index for index, element in enumerate(Z) if element == ele]
    N_ele = [N[i] for i in iter(IND)]
    ME_ele = [ME[i]*10**(-3) for i in iter(IND)] #Mass Excess of Fe in MeV 
    return N_ele, ME_ele

ele = 71
#plot experimental data of specific element    
N_exp, ME_exp = get_exp(ele, Z, N, ME)


def create_test(ele, num, Z, N, Nt, MASS):
    Z_test = ele*np.ones(num+1)
    N_test = [N_exp[-1]+i+1 for i in range(num+1)]
    Z = np.concatenate((Z, Z_test), axis=0)
    N = np.concatenate((N, N_test), axis=0)
    A, sgn = cal_pair(Z, N) 
    X = np.array([Z, N, A, A*A**(-1/3), A*Z**2/A**(4/3), (A-2*Z)**2/A,  A * sgn / A**(3/2)]).T
    x = std_data(X)
    x_train = x[:Nt, :]
    x_test = x[Nt:3436,:]
    y_train = MASS[:Nt]
    y_test = MASS[Nt:]
    return Z, N, A, x_train, x_test, y_train, y_test

def plot_pre(beta, ele, Z, N, A, x_train, x_test, y_train):
    y_hat_train = x_train.dot(beta)
    y_hat_test = x_test.dot(beta)
    Y = np.concatenate((y_hat_train, y_hat_test), axis=0)
    IND = [index for index, element in enumerate(Z) if element == ele]
    N_pre = [N[i] for i in iter(IND)]
    M_pre = [Y[i] for i in iter(IND)] #Mass Excess of Fe in MeV 
    ME_pre = M_pre - (ele+np.array(N_pre))*931.505 
    return N_pre, ME_pre

Z, N, A, x_train, x_test, y_train, y_test = create_test(ele, 0, Z, N, 3200, MASS)  
  
# plot ridge regression prediction
L = np.logspace(-3, 0, 10)
MSE = np.zeros([len(L),2])

for i in range(len(L)):
    beta = cal_beta_r(L[i], x_train,y_train)
    mse = cal_mse(x_train, y_train, beta)
    mse1 = cal_mse(x_test, y_test, beta)
    MSE[i,0] = mse
    MSE[i,1] = mse1
plt.plot(L, MSE[:,0], label = 'train', marker = '*')
plt.plot(L, MSE[:,1], label = 'test', marker = '*')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylabel('MSE')
plt.xlabel(r'$\lambda$')
plt.savefig('ridge_panelty.png', dpi = 250)   



'''
# plot Lasso regression prediction
lambd = np.linspace(10,18,50)
#Coeff_lasso = np.zeros([x_train.shape[1], len(lambd)])

#color=cm.rainbow(np.linspace(0,1,13))
MSE = np.zeros([len(lambd), 2])

for i in range(len(lambd)):
    reg = sl.Lasso(alpha=lambd[i])
    reg.fit(x_train, y_train)
    coe = reg.coef_
    y_hat = reg.predict(x_train)
    y_hat_test = reg.predict(x_test)
    mse = np.square(y_hat-y_train).mean()
    mse1 = np.square(y_hat_test-y_test).mean()
    MSE[i,0] = mse; MSE[i,1] = mse1
    
plt.plot(lambd, MSE[:,0], label = 'train')
plt.plot(lambd, MSE[:,1], label = 'test')
print(MSE)
plt.legend()

#for i in range(len(Coeff_lasso)):
#    plt.plot(lambd, Coeff_lasso[i, :], label = r'$\beta$' + str(i+1))
    
plt.xscale('log')
plt.xlabel(r'$\lambda$')
plt.ylabel('Standarized Coefficients')
plt.axis('tight')
plt.legend(loc = 'upper right')
'''