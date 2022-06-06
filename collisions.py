# Cloud droplets collision-coalescence code

import math
import numpy as np
import random as random
import matplotlib.pyplot as plt

V = 1E6
n_0 = 100	# [cm^-3]
n_0 *= 1E6
N = 1000		# number of superdoplets
xi_init = n_0 * V / N
r_bar = 30.531	# [microns]
r_bar *= 1E-6
v_bar = (4 * math.pi * r_bar *  * 3) / 3
r_min = 0.5E-6
r_max = 60E-6
b = 1.5E3
E = 1

phi = n_0 * v_bar
gamma  =  b * phi

particles_r = np.zeros(N)

def PDF(v):
	return np.exp(-v / v_bar) / v_bar
	
def P_jk_Golovin(r_j, r_k):
	v_j  =  (4 * math.pi * (r_j) ** 3) / 3.0
	v_k  =  (4 * math.pi * (r_k) ** 3) / 3.0
	return b * (v_j + v_k) * dt / V

def P_jk_SD_Golovin(xi_j, xi_k, r_j, r_k):
	return max(xi_j, xi_k) * P_jk_Golovin(r_j, r_k)

def P_jk_SDLS_Golovin(xi_j, xi_k, r_j, r_k, N_p):
	return P_jk_SD_Golovin(xi_j, xi_k, r_j, r_k) * N_p * (N_p - 1.0) / (2.0 * math.floor(N_p / 2.0))

def collision(xi_j, xi_k, r_j, r_k, j):
	out  =  N
	
	if xi_j  ==  1 and xi_k  ==  1:
		xi_j_new = 0
		xi_k_new = 1
		r_j_new = 0
		r_k_new = pow(r_j ** 3 + r_k ** 3, 1.0 / 3.0)
		out  =  j
		
	elif xi_j == xi_k:
		xi_j_new = np.floor(xi_j / 2.0)
		xi_k_new = xi_j - xi_j_new
		r_j_new = pow(r_j ** 3 + r_k ** 3, 1 / 3)
		r_k_new = r_j_new
		
	elif xi_j > xi_k:
		xi_j_new = xi_j - xi_k
		xi_k_new = xi_k
		r_j_new = r_j
		r_k_new = pow(r_j ** 3 + r_k ** 3, 1 / 3)
		
	else:
		xi_j_new = xi_j
		xi_k_new = xi_k - xi_j
		r_j_new = pow(r_j ** 3 + r_k ** 3, 1 / 3)
		r_k_new = r_k
		
	return xi_j_new, xi_k_new, r_j_new, r_k_new, out

def random_list(N):
	L = []
	for i in range(N):
		m = random.randint(0, i)
		L.append(i)
		if m != i:
			L[m], L[i] = L[i], L[m]
	return(L)
	
def n(xi_array):
	return np.sum(xi_array) / V
	
def u(r):
	if r < 40E-6:
		return 1.19E8 * r ** 2
#	elif r < 600E-6:
	else:
		return 8.0E3 * r

	
def R(xi_array,  particles_r):
	return (math.pi / (6.0 * V)) * np.sum(xi_array * pow(2.0 * particles_r, 3) * 1.19E8 * particles_r ** 2)
	
z_0 = 1E-18
def Z(xi_array,  particles_r):
	z  =  (1 / V) * np.sum(xi_array * pow(2.0 * particles_r, 6))
	return 10.0 * np.log10(z / z_0)

# CDF initialisation
L_CDF = int(1E4)
CDF_arr = np.zeros(L_CDF)
CDF_r_arr = np.logspace(-8, -4.2, L_CDF)
CDF_min, CDF_max = 0, 0

for i in range(1, len(CDF_r_arr)):
	v = (4.0 * math.pi * CDF_r_arr[i] ** 3) / 3.0
	dv  =  (4.0 * math.pi * CDF_r_arr[i] ** 3) / 3.0 - (4.0 * math.pi * CDF_r_arr[i-1] ** 3) / 3.0
	CDF_arr[i] = CDF_arr[i-1] + PDF(v) * dv
	
	if CDF_min == 0 and CDF_r_arr[i] > r_min:
		CDF_min = CDF_arr[i]
		
	elif CDF_max == 0 and CDF_r_arr[i] > r_max:
		CDF_max = CDF_arr[i] 

# Inverse random sampling	
for k in range(len(particles_r)):
	psi = random.random()
	
	while psi < CDF_min or psi > CDF_max:
		psi = random.random()
		
	j = 0
	for i in range(len(CDF_arr)):
		if CDF_arr[i] > psi:
			j = i-1
			break
			
	CDF1 = CDF_arr[j];	CDF2 = CDF_arr[j+1];	r1 = CDF_r_arr[j];	r2 = CDF_r_arr[j+1]
	r = r1 + (psi - CDF1) * (r2 - r1) / (CDF2 - CDF1)
	particles_r[k] = r

xi_array = xi_init * np.ones(N)

# Monte Carlo
T = 400
dt = .01
L_t = int(T / dt)
t_array = np.linspace(0, T, L_t)
n_array = np.zeros(L_t)
R_array = np.zeros(L_t)
Z_array = np.zeros(L_t)
u_array = np.zeros(L_t)
for i,  t in enumerate(t_array):
	out_list = []
	xi_list  =  random_list(len(xi_array))
	
	for l in range(int(np.floor(len(xi_array) / 2))):
		j,  k  =  2 * l,  2 * l+1
		xi_j,  xi_k,  r_j,  r_k  =  xi_array[j],  xi_array[k],  particles_r[j],  particles_r[k]
		psi  =  random.random()
		
		if psi < P_jk_SDLS_Golovin(xi_j,  xi_k,  r_j,  r_k,  len(xi_array)):
			xi_array[j],  xi_array[k],  particles_r[j],  particles_r[k],  out  =   collision(xi_j,  xi_k,  r_j,  r_k,  j)
			
			if out ! =  N:
				out_list.append(out)
				
	xi_array = np.delete(xi_array, out_list)
	particles_r = np.delete(particles_r, out_list)
	n_array[i] = n(xi_array)
	R_array[i] = R(xi_array,  particles_r)
	Z_array[i] = Z(xi_array,  particles_r)
	u_sum = 0
	
	for l in range(len(particles_r)):
		u_sum += u(particles_r[l])
	u_array[i] = u_sum / len(particles_r)
	
plt.figure(0)
plt.plot(t_array, 1E-6 * n_array, 'b', label = 'Calculations results')
plt.plot(t_array, 1E-6 * n_0 * np.exp(-gamma * t_array), 'c', linewidth = 3, alpha = .3, label = 'Model solution')
plt.xlim(0, max(t_array))
plt.ylim(0, 1E-6 * n_0)
plt.legend()
plt.xlabel('Time $t$ [s]')
plt.ylabel('Droplet number density $n$ [cm$^{-3}$]')
plt.savefig('n.png')

plt.figure(1)
plt.plot(t_array,  R_array, 'b')
plt.xlabel('Time $t$ [s]')
plt.ylabel('Precipitation rate $R$')
plt.xlim(0, max(t_array))
plt.ylim(min(R_array), max(R_array))
plt.savefig('R.png')

plt.figure(2)
plt.plot(t_array,  Z_array, 'b')
plt.xlim(0, max(t_array))
plt.ylim(10, max(Z_array))
plt.xlabel('Time $t$ [s]')
plt.ylabel('Radar reflectivity factor $Z$ [dBz]')
plt.savefig('Z.png')
	
plt.figure(3)
plt.plot(t_array,  u_array, 'b')
plt.xlim(0, max(t_array))
plt.ylim(0, max(u_array))
plt.xlabel('Time $t$ [s]')
plt.ylabel('Velocity $u$ [m / s]')
plt.savefig('u.png')
