# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:04:13 2017

@author: dalbis
"""


from grid_corr_space import GridCorrSpace
from grid_params import GridRateParams
from grid_functions import compute_gaussian_teo_corr,get_params
import pylab as pl
import plotlib as pp
import numpy as np
from simlib import ensureDir

figures_path='../figures'
ensureDir(figures_path)

paramMap=GridRateParams.gau_grid_small_arena_biphasic_neg
p=get_params(paramMap)
paramMap['amp']=p.input_mean*p.L**2/(2*np.pi*p.sigma**2)
paramMap['b1']=1/p.tau1
paramMap['b2']=1/p.tau2
paramMap['b3']=1/p.tau3

corr=GridCorrSpace(paramMap,use_theory=True,force_gen_corr=False)

gau_C_mean,gau_C_mean_ft,gau_corr_prof,gau_corr_prof_ht,uran,fran_num,kran=compute_gaussian_teo_corr(True,paramMap)


pl.rc('font',size=13)
pp.set_tick_size(5)

pl.figure(figsize=(4,3))
pl.subplots_adjust(left=0.3,bottom=0.2)
pl.plot(uran*100,gau_corr_prof,'-k',label='theory',lw=2)
pl.xlim(0,0.5)
pl.ylim([-0.07,0.07])
pp.custom_axes()
pl.yticks([-0.07,0,0.07])
pl.xticks(np.array([0,0.2,0.4,0.6])*100)

pl.xlabel('Input receptive field distance')
pl.ylabel('Input correlation C')
pl.savefig(figures_path+'/fig3.eps',bbox_inches='tight',dpi=300)
