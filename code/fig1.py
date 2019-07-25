# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:06:05 2016

@author: dalbis
"""



from numpy import arange,pi,argmax
from plotlib import custom_axes
import pylab as pl
from grid_functions import compute_teo_eigs,get_step_response,K_t_sign,K_ft_k,get_params
from grid_params import GridRateParams
from simlib import ensureDir

figures_path='../figures'
ensureDir(figures_path)

# ADAPTATION KERNEL 

paramMap = GridRateParams.gau_grid_small_arena_biphasic_neg

p=get_params(paramMap)          

b1=1/p.tau1
b2=1/p.tau2
b3=1/p.tau3

eigs_freqs,eigs=compute_teo_eigs(None,paramMap,teo_input_pw=True)
eigs_freq_max=eigs_freqs[eigs.argmax()]

eigs_samp_freqs=arange(0,6,1/p.L)
eigs_samp_freqs,samp_eigs=compute_teo_eigs(None,paramMap,teo_input_pw=True,freqs=eigs_samp_freqs)

t_vect,step_vect,resp_vect,K_t=get_step_response(paramMap,step_amp=1.,step_start=-1.,t_min=-1.,t_max=6.)
f_vect=arange(0,5,0.01)
K_vect=K_t_sign(b1,b2,b3,p.mu1,p.mu2,p.mu3,t_vect)

pl.rc('font',size=18)
pl.figure(figsize=(10,3.2))
pl.subplots_adjust(left=0.15,right=0.95,bottom=0.2,top=0.9,hspace=0.2,wspace=0.6)
pl.subplot(1,2,1)
pl.plot(t_vect,K_vect,'-k',lw=2)
custom_axes()
pl.xlabel('Time [s]')
pl.ylabel('Impulse Response [a.u.]')
pl.xlim(-.5,1.5)
pl.yticks([-1,0,1,2,3])


amp_vect=abs(K_ft_k(b1,b2,b3,p.mu1,p.mu2,p.mu3,2*pi*f_vect))
pl.subplot(1,2,2)
pl.plot(f_vect,amp_vect,'-k',lw=2)
custom_axes()
pl.xlabel('Frequency [Hz]')
pl.ylabel('Amplitude [a.u.]')
max_freq=f_vect[argmax(amp_vect)]
print max_freq
pl.axvline(max_freq,color='k',lw=1.5)
pl.yticks([0,0.1,0.2])
pl.savefig(figures_path+'/fig1.eps')