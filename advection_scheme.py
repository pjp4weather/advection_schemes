#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def euler_upwind(y):
    """
    Euler foreward scheme in time and upwind in space
    """
    dy = -u0 * (y - np.roll(y,1))/dx *dt
    return y + dy
    
def lax_wend(y):
    """
    Lax -Wendroff approximation of the advetion equation
    """
    dy = dt * (-u0 * (np.roll(y,-1) - np.roll(y,1))/(2.*dx)  \
               + u0**2. * dt / 2. * (np.roll(y,-1) - 2.*y + np.roll(y,1))/(dx**2))
    return y + dy

def spectral(y):
    """
    Spectral method for the advection scheme
    """
    
    kmin,kmax = 1,len(y)/2
    k = np.arange(kmin,kmax)
    
    A_arr = np.fft.fft(y)
    
    alpha = A_arr.real[kmin:kmax]
    beta = A_arr.imag[kmin:kmax]
    
    dalpha = 2*np.pi * u0 *k/L *beta*dt
    dbeta =  -2*np.pi * u0 *k/L *alpha *dt
    
    alpha_new = alpha + dalpha
    beta_new = beta + dbeta
    
    dalpha = 2*np.pi * u0 *k/L *beta_new*dt
    dbeta =  -2*np.pi * u0 *k/L *alpha_new *dt
    
    alpha = alpha + dalpha
    beta = beta + dbeta
    
    A_arr[kmin:kmax] = alpha + 1j * beta
    
    A_arr[kmin+kmax+1:] = np.flip(alpha - 1j * beta,0)
    
    return np.fft.ifft(A_arr)


    


#%%
L = 2500e3
dx = 25e2
u0 = 10.
dt = 25.


iters = int(250e3/dt)

Larr = np.arange(0,L+dx,dx)

C0 = np.zeros_like(Larr)
C0[(Larr<=1375e3) & (Larr>=1125e3)] = 5.

#C0 = np.sin(10*np.pi* Larr/L)

Ceuler, Clax_wend, Cspectral = map(lambda a:np.array(a), [C0,C0,C0]) 

for i in range(iters):
    Ceuler = euler_upwind(Ceuler)
    Clax_wend = lax_wend(Clax_wend)
    Cspectral = spectral(Cspectral)
    
    
plt.close("all")
plt.plot(Larr/1000.,C0, label="analytical solution")
plt.plot(Larr/1000.,Ceuler, label="euler upwind")
plt.plot(Larr/1000.,Clax_wend, label="lax wendroff")
plt.plot(Larr/1000.,Cspectral, label="spectral")
plt.xlabel("L [km]")
plt.ylabel("C")
plt.legend()