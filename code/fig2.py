  # -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 17:21:01 2016

@author: dalbis
"""


### RANDOM WALK ===============================================================

import numpy as np
import pylab as pl
from numpy.random import randn,seed,rand
import plotlib as pp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from simlib import ensureDir

figures_path='../figures'
ensureDir(figures_path)


dt = 0.05                  
speed = 0.25                
theta_sigma = 0.7
tau_max=0.8

sim_time = 2.        
arena_shape = 'square'     

seed(3)
  
num_samples=10
theta=rand(num_samples)*2*np.pi
time_vect=np.arange(0,sim_time+dt,dt)                      

p_vect = np.zeros((num_samples,2,len(time_vect)))       
theta_vect = np.zeros((num_samples,len(time_vect)))     
p = np.zeros((num_samples,2))

periodic_bounds=True

t_max_vect = np.arange(0.08,1,0.04)

for t_idx in range(len(time_vect)):
  p0 = p
  p_vect[:,:,t_idx]=np.squeeze(p0)
  theta_vect[:,t_idx]=theta
  theta = theta+theta_sigma*randn(num_samples)*np.sqrt(dt)
  v = speed*np.array([np.sin(theta),np.cos(theta)])
  p=p0+v.T*dt

pl.rc('font',size=18)

jet = cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=num_samples)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

pl.figure(figsize=(9,3.5))
pl.subplots_adjust(bottom=0.2,hspace=0.5,wspace=0.4)
ax=pl.subplot(121,aspect='equal')

tau_max_idx=np.where(time_vect==tau_max)[0][0]



import matplotlib.pyplot as plt

circle1 = plt.Circle((0, 0),speed*tau_max, color=[0.75,0.75,0.75])
ax.add_artist(circle1)    
for idx in xrange(num_samples):
  pl.plot(p_vect[idx,0,:],p_vect[idx,1,:],color=scalarMap.to_rgba(idx),linewidth=2.)
  pl.plot(p_vect[idx,0,tau_max_idx],p_vect[idx,1,tau_max_idx],'.k',markersize=16,color=scalarMap.to_rgba(idx))

pl.xlim(-0.4,0.4)
pl.ylim(-0.4,0.4)
pp.custom_axes()

pl.xticks([])
pl.yticks([])

pl.savefig(figures_path+'/fig2.eps',bbox_inches='tight',dpi=300)



