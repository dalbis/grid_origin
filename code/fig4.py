# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:30:50 2017

@author: dalbis
"""



from grid_functions import g_ht_k,K_ht_k
import numpy as np
import pylab as pl
from simlib import ensureDir

figures_path='../figures'
ensureDir(figures_path)

def get_max_frequency(tau_s,tau_l,mu,
                      sigma=0.0625,speed=0.25,L=1.0,input_mean=0.4,N=900,ret_curve=False):
  
  
  fran=np.arange(0,5,0.001)  
  amp=input_mean*L**2/(2*np.pi*sigma**2)
    
  K_ht_k_short= lambda k: K_ht_k(1./tau_s,1./tau_l,1.,1.,mu,0.,speed,k)
  
  
  Kht_vect=K_ht_k_short(2*np.pi*fran)
  ght2_vect=g_ht_k(amp,sigma,2*np.pi*fran)**2
  corr_ht_est=Kht_vect*ght2_vect/L**2
  
  density=N/L**2
  raw_eigs=corr_ht_est*density
  
  
  max_freq=fran[raw_eigs.argmax()]
  max_eig=raw_eigs.max()
  
  if ret_curve is True:
    return fran,raw_eigs,max_freq,max_eig
  return max_freq,max_eig


#%%
## COMPUTE MAX FREQUENCY AND EIGENVALUE FOR A REGION OF THE PARAMETER SPACE

def_a=1.
tau_s_def=0.1
tau_l_def=0.16
mu_def=1.06

tau_s_ran=np.arange(0.02,0.162,0.002)
tau_l_ran=np.arange(0.1,0.402,0.002)


mu_ran=-np.arange(0.8,1.201,0.01)

max_freq_mat_taul=np.zeros((len(tau_l_ran),len(mu_ran)))
max_freq_mat_taus=np.zeros((len(tau_s_ran),len(mu_ran)))

max_eig_mat_taul=np.zeros_like(max_freq_mat_taul)
max_eig_mat_taus=np.zeros_like(max_freq_mat_taus)


for tau_l_idx,tau_l in enumerate(tau_l_ran):
  for mu_idx,mu in enumerate(mu_ran):
    max_freq,max_eig=get_max_frequency(tau_s_def,tau_l,mu)
    max_freq_mat_taul[tau_l_idx,mu_idx]=max_freq
    max_eig_mat_taul[tau_l_idx,mu_idx]=max_eig
      

for tau_s_idx,tau_s in enumerate(tau_s_ran):
  for mu_idx,mu in enumerate(mu_ran):
    max_freq,max_eig=get_max_frequency(tau_s,tau_l_def,mu)
    max_freq_mat_taus[tau_s_idx,mu_idx]=max_freq
    max_eig_mat_taus[tau_s_idx,mu_idx]=max_eig


#%%%%% PLOT MAX FREQUENCIES AND MAX EIGENVALUES


def get_contour(CS,idx):
  path = CS.collections[idx].get_paths()[0]
  v = path.vertices
  x = v[:,0]
  y = v[:,1]
  return x,y,v

def interpolate_contour(CS,idx):
  x,y,v=get_contour(CS,idx)
  z = np.polyfit(x, y, 3)
  p = np.poly1d(z)
  v[:,1]=p(x)

import plotlib as pp
    
pl.rc('font',size=12)
pp.set_tick_size(5)



freq_levels=[-1,0,1,1.5,2,2.5,3,4,5] 
mod_depth_levels=[0,1,5,10,20,30,40,50]


X_taul, Y_taul = np.meshgrid(tau_l_ran, 1+mu_ran)
X_taus, Y_taus = np.meshgrid(tau_s_ran, 1+mu_ran)


pl.figure(figsize=(8,6))
pl.subplots_adjust(bottom=0.2,wspace=0.3,hspace=0.5,right=0.9)



ax1=pl.subplot(2,2,1)
ax2=pl.subplot(2,2,3)
ax3=pl.subplot(2,2,2)
ax4=pl.subplot(2,2,4)

axes=[ax1,ax3,ax2,ax4]

# max_freq - tau_l
pl.sca(ax1)
CS_maxfreq_taul=pl.contourf(X_taul,Y_taul,max_freq_mat_taul.T,
                            levels=freq_levels,cmap='Greens')
                            
CS_maxfreq_taul_lines=pl.contour(X_taul,Y_taul,max_freq_mat_taul.T,
                                 levels=freq_levels,colors='k')

interpolate_contour(CS_maxfreq_taul_lines,1)

freq0_taul=get_contour(CS_maxfreq_taul_lines,1)


pl.clabel(CS_maxfreq_taul_lines, inline=0, fontsize=11,fmt='%.1f',colors='k')



# max eig -tau_l

pl.sca(ax3)

CS_maxeig_taul=pl.contourf(X_taul,Y_taul,max_eig_mat_taul.T-def_a,levels=mod_depth_levels,cmap='Blues',vmin=-2)

CS_maxeig_taul_lines=pl.contour(X_taul,Y_taul,max_eig_mat_taul.T-def_a,levels=mod_depth_levels,colors='k')

eig0_taul=get_contour(CS_maxeig_taul_lines,0)

pl.clabel(CS_maxeig_taul_lines, inline=0, fontsize=11,fmt='%d',colors='k')



# max_freq - tau_s
pl.sca(ax2)

CS_maxfreq_taus=pl.contourf(X_taus,Y_taus,max_freq_mat_taus.T,
                            levels=freq_levels,cmap='Greens')
                            
CS_maxfreq_taus_lines=pl.contour(X_taus,Y_taus,max_freq_mat_taus.T,
                                 levels=freq_levels,colors='k')

interpolate_contour(CS_maxfreq_taus_lines,1)

freq0_taus=get_contour(CS_maxfreq_taus_lines,1)

pl.clabel(CS_maxfreq_taus_lines, inline=0, fontsize=11,fmt='%.1f',colors='k')


  
# max eig - tau_s

pl.sca(ax4)

CS_maxeig_taus=pl.contourf(X_taus,Y_taus,max_eig_mat_taus.T-def_a,levels=mod_depth_levels,cmap='Blues',vmin=-5)
CS_maxeig_taus_lines=pl.contour(X_taus,Y_taus,max_eig_mat_taus.T-def_a,levels=mod_depth_levels,colors='k')

eig0_taus=get_contour(CS_maxeig_taus_lines,0)


pl.clabel(CS_maxeig_taus_lines, inline=0,fontsize=11,fmt='%d',colors='k')


# plot default values and zero line
for idx,ax in enumerate(axes):
  pl.sca(ax)
  if idx<2:
    pl.plot(tau_l_def,1-mu_def,'*k',ms=12)
    
    pl.xlim(0.1,0.4)
    pl.xticks([0.1,0.2,0.3,0.4])
    ax.set_xticks(np.arange(0.1,0.4,0.02), minor = True)

    pl.xlabel('tau_l [s]')

  else:
    pl.plot(tau_s_def,1-mu_def,'*k',ms=12)
    pl.xlim(0.02,0.16)
    pl.xlabel('tau_s [s]')
    pl.xticks([0.02,0.07,0.12,0.16])
    ax.set_xticks(np.arange(0.02,0.16,0.01), minor = True)
  
  if idx in (0,2):
    pl.ylabel('Kernel integral')
    pl.title('Critical spatial frequency')
  else:
    pl.title('Largest eigenvalue')
    
  pl.tick_params(which='minor', length=3)
  pp.custom_axes()    
  pl.axhline(0,color='k',lw=2,ls='--')  
  pl.yticks([-0.2,-0.1,0,0.1,0.2])
  ax.set_yticks(np.arange(-0.2,0.2,0.05), minor = True)

  pl.ylim(-0.2,0.2)



# black fills to mask-out uninteresting parameter values
pl.sca(ax1)
x,y,v=eig0_taul
pl.fill_between(x, (1+mu_ran).min(),y, facecolor='k', interpolate=True)


pl.sca(ax3)
x,y,v=freq0_taul
pl.fill_between(x, y, (1+mu_ran).max(), facecolor='k', interpolate=True)


pl.sca(ax2)
x,y,v=eig0_taus
pl.fill_between(x, (1+mu_ran).min(),y, facecolor='k', interpolate=True)


pl.sca(ax4)
x,y,v=freq0_taus
pl.fill_between(x, y, (1+mu_ran).max(),facecolor='k', interpolate=True)

pl.savefig(figures_path+'/fig4.eps', dpi=300,transparent=True)

  
