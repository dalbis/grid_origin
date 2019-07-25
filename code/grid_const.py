# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 11:48:00 2016

@author: dalbis
"""


class FilterType:
  FILTER_INPUT='FILTER_INPUT'
  FILTER_OUTPUT='FILTER_OUTPUT'
  
class ModelType:

  MODEL_RATE='MODEL_RATE'
  MODEL_SPIKING='MODEL_SPIKING'
  MODEL_RATE_AVG='MODEL_RATE_AVG'

  MODEL_CORR_SPACE='MODEL_CORR_SPACE'
  
class DistType:
  J0_EXP='Exp'
  J0_NORM='Norm'
  J0_HALF_EXP='HalfExp'
  

class InputType:
  
  INPUT_GAU_GRID='GaussianGrid' 
  INPUT_GAU_NOISY_CENTERS='GaussianNoisyCenters'  
  INPUT_GAU_RANDOM_CENTERS='GaussianRandomCenters'

  # sum of positive and negativa Gaussians: shifted to have positive rates and rescaled to have fixed mean
  INPUT_GAU_MIX='MixOfGau'             

  # sum of positive and negativa Gaussians: shifted to have fixed mean (positive and negative rates!!)
  INPUT_GAU_MIX_NEG='MixOfGauNeg'      

  # sum of positiva Gaussians with variable amps rescaled to fixed mean
  INPUT_GAU_MIX_POS='MixOfGauPos'      

  # sum of positiva Gaussians with fixed amps rescaled to fixed mean
  INPUT_GAU_MIX_POS_FIXAMP='MixOfGauPosFixAmp' 
  
  INPUT_RAND='Random'
  INPUT_RAND_CORR='RandomCorrelated'
  
  INPUT_BVC='BVC'
  
  # perfect grid plus addditive noise
  INPUT_NOISY_GRID='NoisyGrid'  

  # perfect grid plus addditive noise and noisy orientation
  INPUT_NOISY_GRID_JITTER='NoisyGridJitter'  

  # perfect grids with two different angles plus addditive noise
  INPUT_NOISY_GRID_TWO_ANGLES='NoisyGridTwoAngles'  
  
  # scattered grid-field locations
  INPUT_NOISY_GRID_SCATTERED_FIELDS='NoisyGridScatter'