# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 17:50:03 2017

@author: dalbis
"""


from grid_inputs import GridInputs
from grid_params import GridRateParams
from grid_corr_space import GridCorrSpace
from grid_functions import map_merge,get_params,compute_teo_eigs
import pylab as pl
import numpy as np
import plotlib as pp

from simlib import ensureDir

figures_path='../figures'
ensureDir(figures_path)


inputs_seed=89
n=60

num_gau_mix_ran=np.arange(2,24,2)

all_mean_pw_profs=[]
all_mean_pw_profs_norm=[]

all_teonum_scale_factors=[]
all_num_eigs=[]

all_teo_eigs=[]

for num_gau_mix in num_gau_mix_ran:
  param_map=map_merge(GridRateParams.gau_mix_small_arena_biphasic_neg,
                      {'num_gau_mix':num_gau_mix,
                       'n':n,
                       'inputs_seed':inputs_seed
                      })
  p=get_params(param_map)                    
  
  inputs=GridInputs(param_map,keys_to_load=['in_freqs','in_mean_pw_profile','random_amps'])
  in_freqs=inputs.in_freqs
  
  corr=GridCorrSpace(param_map,keys_to_load=['eigs','CC_teo'])
  num_eigs=np.real(np.linalg.eigvals(corr.CC_teo-np.diag([p.a]*p.n**2)))
  all_num_eigs.append(num_eigs)
  
  eigs_freqs,teo_eigs=compute_teo_eigs(inputs,param_map,teo_input_pw=True)
  all_teo_eigs.append(np.real(teo_eigs))
    
  all_mean_pw_profs.append(inputs.in_mean_pw_profile)
  all_mean_pw_profs_norm.append(inputs.in_mean_pw_profile/inputs.in_mean_pw_profile[3])


  phases=np.random.uniform(0,1,size=(p.n**2,num_gau_mix))

  alphas=np.abs(np.sum(inputs.random_amps*np.exp(-1j*2*np.pi*phases),axis=1))
  betas=inputs.random_amps.sum(axis=1)
  
  all_teonum_scale_factors.append(np.mean((alphas/betas)**2))


#%%

all_mean_pw_profs=np.array(all_mean_pw_profs)
all_mean_pw_profs_norm=np.array(all_mean_pw_profs_norm)


p=get_params(param_map)
const_factor=(p.L**2*p.input_mean/(2*np.pi*p.sigma**2))**2
  
    
gk_profile = lambda k: p.sigma**2*2*np.pi*np.exp(-k**2*p.sigma**2/2.)


num_gau_mix_cont=np.arange(num_gau_mix_ran[0],num_gau_mix_ran[-1],0.01)  


# As: uniformely distibuted variable in the interval (0,1)
As_m2 =lambda n: 1./3


## beta: sum of uniformely distributed variables in the interval (0,1)
beta_m1= lambda n: n/2.
beta_m2= lambda n: n/12.+n**2/4.
beta_var=lambda n: n/12.


# alpha sum of complex exponentials with coefficients uniformely distributed variables in the interval (0,1) and random phases

alpha_m1=lambda n: np.sqrt(n*np.pi/12)
alpha_m2=lambda n: n/3.


alpha_var=lambda n: alpha_m2(n)-alpha_m1(n)**2

const_factor=(p.L**2*p.input_mean/(2*np.pi*p.sigma**2))**2


full_var_teo_scale_factor_pw= lambda n: (alpha_m1(n)/beta_m1(n))**2*(alpha_var(n)/alpha_m1(n)**2+beta_var(n)/beta_m1(n)**2+1)

full_teo_scale_factor_pw= lambda n: (alpha_m1(n)/beta_m1(n))**2*(alpha_m2(n)/alpha_m1(n)**2+beta_m2(n)/beta_m1(n)**2-1)

teo_scale_factor_pw= lambda n: np.pi/(3*n)*(4/np.pi+1./(3*n))
teo_scale_factor_simple= lambda n: 4./(3*n)
  
num_scale_factor=all_mean_pw_profs[:,1]/(const_factor*gk_profile(2*np.pi)**2)



#%%


### PLOT SCALE FACTOR

pl.figure(figsize=(4,3))
pl.plot(num_gau_mix_cont,teo_scale_factor_pw(num_gau_mix_cont),lw=2,color=pp.red)
pl.plot(num_gau_mix_ran,num_scale_factor,'.k',ms=12)
pp.custom_axes()
pl.ylim([0,0.8])
pl.xlim(1.,23)
pl.xticks([2,6,10,14,18,22])
pl.yticks([0,0.2,0.4,0.6,0.8])
pl.xlabel('Number of superimposed fields')
pl.ylabel('Scale factor')
pl.savefig(figures_path+'/fig10a.eps',bbox_inches='tight',dpi=300)

#%%

param_map=map_merge(GridRateParams.gau_mix_small_arena_biphasic_neg,
                      {'num_gau_mix':1,
                       'n':60,
                       'inputs_seed':89
                      })

p=get_params(param_map) 

## PLOT THEORETICAL VS NUMERICAL MAXIMAL EIGENVALUE
all_num_eigs=np.array(all_num_eigs)
all_teo_eigs=np.array(all_teo_eigs)

max_num_eigs=all_num_eigs.max(axis=1)
max_teo_eigs=all_teo_eigs.max(axis=1)




freqs,teo_eigs=compute_teo_eigs(None,param_map,teo_input_pw=True)
  

pl.figure(figsize=(4,3))
pl.plot(num_gau_mix_cont,(teo_scale_factor_pw(num_gau_mix_cont)*teo_eigs.max())-p.a,lw=2,color=pp.red)
pl.plot(num_gau_mix_ran,max_num_eigs,'.k',ms=12)
pp.custom_axes()
pl.xlim(1.,23)
pl.xticks([2,6,10,14,18,22])
pl.ylim(-2,25)
pl.xlabel('Number of superimposed fields')
pl.ylabel('Maximum eigenvalue [1/s]')
pl.savefig(figures_path+'/fig10b.eps',bbox_inches='tight',dpi=300)


