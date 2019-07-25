# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:29:24 2014

@author: dalbis
"""

import numpy as np
import pylab as pl
from matplotlib.ticker import MaxNLocator,FixedLocator
from gridlib import get_rhombus
from matplotlib import collections
from matplotlib.colors import Normalize,LinearSegmentedColormap
import matplotlib.pyplot as plt
from numpy.fft import fft2,fftshift,fftfreq
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib

blue=np.array([30,144,255])/255.
red=np.array([200,0,0])/255.
green=np.array([0,180,0])/255.
violet=np.array([138,43,226])/255.
gray=np.array([100,100,100])/255.
lightGray=np.array([200,200,200])/255.
middleGray=np.array([125,125,125])/255.

darkRed=np.array([139,0,0])/255.
orangeRed=np.array([255,69,0])/255.
orange=np.array([255,140,0])/255.

darkBlue=np.array([0, 0, 140])/255. 
blue=np.array([0, 0, 240])/255. 
dogeBlue=np.array([30, 144, 255])/255. 
skyBlue=np.array([135, 206, 235])/255.


def set_y_max_ticks(n):
  pl.gca().yaxis.set_major_locator(MaxNLocator(n))

def set_x_max_ticks(n):
  pl.gca().xaxis.set_major_locator(MaxNLocator(n))


def subplots(rows,cols):
  fig=pl.figure(figsize=(5*cols,4*rows))  
  axes=[]
  ax_idx=1
  for row_idx in xrange(rows):
    for col_idx in xrange(cols):
      ax=pl.subplot(rows,cols,ax_idx)
      axes.append(ax)
      custom_axes()
      ax_idx+=1
  return fig,axes  
  
def save_fig(tag,file_path,plot_path,ext='eps'):
  fig_path='%s%s_%s.%s'%(plot_path,file_path,tag,ext)
  pl.savefig(fig_path,bbox_inches='tight',dpi=300)


def plot_weight_dist(snap_times,J_vect,alpha=0.2,lines=False):
  pl.plot(snap_times,(J_vect.T),color=[.0,.0,.0,alpha])
  pl.grid(b=True, which='major', color='gray',linestyle=':',axis='x',linewidth=1.)
  

def plot_dft_angles(snap_times,angles,amps,sim_time,logx=False,logy=False,plot_wheel=False,lw=1.5,cmap='hsv',wheel_axis=None,label_size=10):
  """
  Plot 2D-DFT amplitudes at a specific frequency as a function of time and angle
  """
  
  if logy : pl.gca().set_yscale('log')
  if logx : pl.gca().set_xscale('log')  
  
  ax=pl.gca()
  custom_axes()
  ax.set_xlim(snap_times[1] if pl.gca().get_xscale()=='log' else 0,sim_time)
    
  det_lines = LineCollection([list(zip(snap_times[1:], amp[1:])) for amp in amps],
                               linewidths=lw,
                               linestyles='-')

  ax.set_ylim((np.amin(amps), np.amax(amps)))
  
  det_lines.set_array(angles)
  det_lines.set_clim(0,180)
  det_lines.set_cmap(cmap)
  
  ax.add_collection(det_lines)
  pl.sci(det_lines) 
    

  if plot_wheel is True:
    color_steps = 2056
    pos = ax.get_position() 
    wheel_pos= [pos.x0+pos.width/20., pos.y0+pos.height*0.7,  pos.width / 4.0, pos.height / 4.0] 
    
    if wheel_axis is None:
      wheel_axis = pl.gcf().add_axes(wheel_pos, projection='polar')
    else:
      wheel_axis.projection='polar'

    wheel_axis._direction = 2*np.pi  
      
    norm = matplotlib.colors.Normalize(0.0, 180.)
  
    cb = matplotlib.colorbar.ColorbarBase(wheel_axis, cmap=cm.get_cmap(cmap,color_steps),
                                       norm=norm,
                                       orientation='horizontal',ticks=[0, 30, 60, 90, 120,150])
    cb.ax.tick_params(labelsize=label_size) 
    cb.outline.set_visible(False)   
    
  return ax
  
  

def plot_radial_profiles(freqs,time,profiles,eigs,plot_freqs,plot_teo=True,drift_profiles=None):
  
  assert(len(plot_freqs)<=5)
  n=len(eigs)
  eigs_1d=eigs[n/2,:]
  eigs_1d_pos=eigs_1d[n/2:]
  
  colors=['k','b','r','gray','g']

  #pl.figure(figsize=(12,5))
  #pl.subplot(111)
  #pl.subplots_adjust(left=0.1)

  for idx in xrange(len(plot_freqs)):    
    freq_idx=np.where(freqs==plot_freqs[idx])
    pl.plot(time,np.squeeze(profiles[:,freq_idx]),color=colors[idx],linestyle='-',label='%.2f 1/m'%freqs[freq_idx])

    if plot_teo is True:
      teo_profile=np.squeeze(profiles[0,freq_idx])*np.exp(time*eigs_1d_pos[freq_idx])
      pl.plot(time,teo_profile,color=colors[idx],linestyle='--')#label='teo. %.2f 1/m'%freqs[freq0_idx])

    if drift_profiles is not None:
      pl.plot(time,np.squeeze(drift_profiles[:,freq_idx]),color=colors[idx],linestyle=':')

  pl.xlabel('Time [s]')
  pl.ylabel('Amplitude')
  pl.legend(loc='upper left',prop={'size':10})
  custom_axes()



def plot_tiled_corr(C,n,cmap='seismic',midpoint_norm=True,midpoint=0.,title=None):
  N=n**2
    
  # reshape to four dimensions
  C4d=C.reshape(n,n,n,n)

  # create tiled matrix
  C_tiled=np.zeros((N,N))
  for i in xrange(n):
    for j in xrange(n):
      C_tiled[n*i:n*(i+1),n*j:n*(j+1)]=C4d[i,j,:,:]

  # plot correlation matrix      
  fig=pl.figure(figsize=(12,10))
  pl.subplot(111,aspect='equal')
  pl.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05)
  noframe()      
  
  if midpoint_norm:
    pl.pcolormesh(C_tiled,cmap=cmap,norm=MidPointNorm(midpoint=midpoint))
  else:
    pl.pcolormesh(C_tiled,cmap=cmap)
  
  # draw separating lines
  for i in xrange(n):
    pl.axhline(y=i*n,color='k')
    pl.axvline(x=i*n,color='k')
  colorbar()
  custom_axes()

  if title is not None:
    fig.canvas.set_window_title(title)
    
  return C_tiled
  



  
def minmax(mat,dec=1):
  fact=10.**dec
  mat_min,mat_max=np.floor(np.amin(mat*fact))/fact,np.ceil(np.amax(mat)*fact)/fact
  return mat_min,mat_max
  
def mimic_alpha(color,alpha,bgcolor=np.array([1,1,1])):
  return (1 - alpha) * bgcolor + alpha*color

def gen_animation(M,scores,delta_snap,dt,vmin=None,vmax=None):
  """
  Generate an animation of weights/rates evolution plus gridness score
  """
  
  n=M.shape[0]
  tc=n/50. if n>100 else 1
    
  num_snaps=M.shape[2]
  sim_time=num_snaps*delta_snap*dt
  
  fig = plt.figure(figsize=(15,7.5))
  fig.set_size_inches([15,7.5])
  ax = fig.add_subplot(211)
  ax.set_aspect('equal')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  
  vmax=vmax if vmax is not None else np.amax(M)
  vmin=vmin if vmin is not None else np.amin(M) 
    
  im=ax.pcolormesh(M[:,:,0],vmin=vmin,vmax=vmax)    
  #colorbar(obj=im)
  pl.title('%.2f / %.2f'%(vmin,vmax))
  
  if len(scores)>0:
    score_text=pl.text(tc,tc,'%.2f'%(0.), color='black',fontsize=10, weight='bold',bbox={'facecolor':'white'})    
    ax = fig.add_subplot(212)
    custom_axes()
    line,=pl.plot(0,scores[0],'-k')
    pl.xlim([0,sim_time])
    pl.ylim([-0.5,2])
    pl.xlabel('Time [s]')
    pl.ylabel('Grid score')
    
  pl.tight_layout()
  
  def update_img(i):  
    im.set_array(M[:,:,i].ravel())
    if len(scores)>0:
      score_text.set_text('%.2f'%scores[i])
      line.set_data(np.arange(0,i)*delta_snap*dt,scores[0:i])
    return im
  
  ani = FuncAnimation(fig,update_img,frames=num_snaps,interval=20,blit=False)
  return ani
  
circle_xy = lambda r,phi : (r*np.cos(phi), r*np.sin(phi))

def plot_matrixDFT(M,dt,circle_radius=np.NaN,cmap='gray',circle_color='r',lw=1.5,plot_circle=True):
  n=M.shape[0]
  freqs = fftshift(fftfreq(n,d=dt))
  df=abs(freqs[1]-freqs[0])
  M_dft=fftshift(abs(fft2(M)))
  M_dft[n/2,n/2]=0.
  mesh=pl.pcolormesh(freqs-df/2.,freqs-df/2.,M_dft,rasterized=True,cmap=cmap)
  #pl.title('%.2e / %.2e'%(np.amin(M_dft),np.amax(M_dft)),fontsize=12)
  if circle_radius is not np.NaN:
    if plot_circle:
      x,y=circle_xy(circle_radius,np.arange(0,2*np.pi,0.01))
      pl.plot(np.array(x),np.array(y), c=circle_color,ls='-',lw=1.5)
    pl.xlim(-2*circle_radius,2*circle_radius)  
    pl.ylim(-2*circle_radius,2*circle_radius)  
    
  return M_dft,mesh
   
def plot_matrixDFT_evo(M,dt,circle_radius=np.NaN,plot_snaps=25,title=None):
  """
  Plot the evolution of a matrix in 100 snapshots
  """
  
  assert(len(M.shape)==3)
  num_snaps=M.shape[2]
  factor=1
  
  n_row=int(np.ceil(np.sqrt(plot_snaps)))

  if num_snaps > plot_snaps:
    assert(np.remainder(num_snaps,plot_snaps)==0)
    factor = num_snaps/plot_snaps 

  n=M.shape[0]
  assert(np.remainder(n,2)==0)
  
  fig=pl.figure(figsize=(13,13))
  pl.subplots_adjust(left=0.1,right=0.9,wspace=0.2,hspace=0.1,bottom=0.05,top=0.95)  
  
  for snap_idx in range(0,num_snaps,factor):
    subplot_idx = int(snap_idx/factor+1)
    pl.subplot(n_row,n_row,subplot_idx,aspect='equal')
    Mt=M[:,:,snap_idx]
    plot_matrixDFT(Mt,dt,circle_radius=circle_radius)
    noframe()

  if title is not None:
    fig.canvas.set_window_title(title)
    
  return fig
  
def plot_matrix(M,vmin=None,vmax=None,X=None,Y=None,dec=6,title=None,show_cbar=True,cmap='jet'):
  sig_min,sig_max=minmax(M,dec=dec)

  vmin=sig_min if vmin is None else vmin
  vmax=sig_max if vmax is None else vmax
  
  fig=pl.figure()
  pl.subplot(111,aspect='equal')
  noframe()
  pl.pcolormesh(M,vmin=vmin,vmax=vmax,cmap=cmap)
  if show_cbar is True:
    colorbar()
  
  if title is not None:
    fig.canvas.set_window_title(title)

def plot_matrix_evo(M,vmin=None,vmax=None,labels=None,label_str='%d',plot_snaps=25,dec=3,cmap='jet',title=None,
                    common_scale=False,x=None,circle_radius=None,circle_color='r',labelx=1,zoom_circle=False,min_max_str='%.2e / %.2e'):
  """
  Plot the evolution of a matrix in 100 snapshots
  """
  sig_min,sig_max=minmax(M,dec=dec)

  vmin=sig_min if vmin is None else vmin
  vmax=sig_max if vmax is None else vmax
    
  assert(len(M.shape)==3)
  num_snaps=M.shape[2]
  factor=1
  
  n_row=int(np.ceil(np.sqrt(plot_snaps)))
  if num_snaps > plot_snaps:
    assert(np.remainder(num_snaps,plot_snaps)==0)
    factor = num_snaps/plot_snaps 

  fig=pl.figure(figsize=(13,13))
  pl.subplots_adjust(left=0.1,right=0.9,wspace=0.2,hspace=0.1,bottom=0.05,top=0.95)  
  
  axes=[]
  
  for snap_idx in range(0,num_snaps,factor):
    subplot_idx = int(snap_idx/factor+1)
    ax=pl.subplot(n_row,n_row,subplot_idx,aspect='equal')
    axes.append(ax)
    if x is not None:
      dt = abs(np.diff(x)[0])
      if common_scale is True:
         mesh=pl.pcolormesh(x,x,M[:,:,snap_idx],vmin=vmin,vmax=vmax,cmap=cmap,rasterized=True)
      else:
         mesh=pl.pcolormesh(x,x,M[:,:,snap_idx],cmap=cmap,rasterized=True)

      if circle_radius is not None:
        cx,cy=circle_xy(circle_radius,np.arange(0,2*np.pi,0.01))
        pl.plot(np.array(cx)+dt/2,np.array(cy)+dt/2, c=circle_color,ls='-')
        if zoom_circle:
          pl.xlim(-1.5*circle_radius,1.5*circle_radius)
          pl.ylim(-1.5*circle_radius,1.5*circle_radius)
    else:       
        if common_scale is True:
          mesh=pl.pcolormesh(M[:,:,snap_idx].T,vmin=vmin,vmax=vmax,cmap=cmap,rasterized=True)
        else:
          mesh=pl.pcolormesh(M[:,:,snap_idx].T,cmap=cmap,rasterized=True)
      
    noframe()
    pl.title(min_max_str%(np.amin(M[:,:,snap_idx]),np.amax(M[:,:,snap_idx])),fontsize=12)

    if labels is not None:
      pl.text(labelx,labelx,label_str%labels[snap_idx], color='black',fontsize=10, weight='bold',bbox={'facecolor':'white'})

  
  if common_scale is True:  
    cbar_ax = fig.add_axes([0.91, 0.06, 0.007, 0.06 ])    
    pl.colorbar(mesh, cax=cbar_ax,ticks=np.array([np.ceil(vmin*10**(dec-1))/10**(dec-1),round((vmax-abs(vmin))/2,dec-1),np.floor(vmax*10**(dec-1))/10**(dec-1)]))
    for label in (cbar_ax.get_xticklabels() + cbar_ax.get_yticklabels()):
      label.set_fontsize(10)

  if title is not None:
    fig.canvas.set_window_title(title)
  return fig,axes

def plot_2dfourier_coeffs(signal,num_comp=5,norm=Normalize(),vmin=np.NaN,vmax=np.NaN):
  """
  Plots 2D Fourier coefficients of an hexagonal grid
  """
  
  ran=np.arange(-num_comp,num_comp+2)-0.5
  X,Y=np.meshgrid(ran,ran)

  zero_idx=(len(signal)-1)/2
  signal_slice=signal[zero_idx-num_comp:zero_idx+num_comp+1,zero_idx-num_comp:zero_idx+num_comp+1]
  sig_min,sig_max=minmax(signal_slice,dec=0)

  if vmin is np.NaN:
    vmin=sig_min
    
  if vmax is np.NaN:
    vmax=sig_max
      
  custom_axes()
  pl.pcolormesh(X,Y,signal_slice,cmap='gist_yarg',norm=norm,rasterized=True,vmin=vmin,vmax=vmax) 

  pl.xlim([-num_comp-0.5,num_comp+0.5])
  pl.ylim([-num_comp-0.5,num_comp+0.5])


def plot_on_rhombus(R,side,alpha,num_samp,samples,signal,side_symbol=None,plot_axes=True,plot_cbar=True,clim=None,norm=Normalize(),cmap='jet',plot_rhombus=False):
  """
  Plot a function on a rhomboidal primary cell of an hexagonal lattice.
  R: rhombus of the lattice primary cell
  side: Side-length of the rhombus
  num_samp: sumber of samples in the lattice
  samples: lattice samples
  signal: signal to plot
  """
  
  side_neg = '%.2f'%(-side/2.) if side_symbol is None else '-'+side_symbol+'/2'
  side_pos = '%.2f'%(side/2.) if side_symbol is None else side_symbol+'/2'
    
  dR = get_rhombus(side/np.sqrt(num_samp)*1.01,np.pi/6+alpha)
  
  R_grid = [dR-samples[idx,:] for idx in np.arange(num_samp)]
    
  ax=pl.gca()
  custom_axes()
  poly = collections.PolyCollection(R_grid,cmap=cmap,norm=norm,linewidths=0,rasterized=True)
  poly.set_array(signal)
  poly.set_edgecolors('')
  
  if clim is not None:
    poly.set_clim(clim)
    
  ax.add_collection(poly,autolim=False)
  
  xmin = np.amin(R[:,0])
  xmax = np.amax(R[:,0])
  ymin = np.amin(R[:,1])
  ymax = np.amax(R[:,1])
  
  noframe()
  
  pl.xlim([xmin-0.1,xmax+0.1])
  pl.ylim([ymin-0.1,ymax+0.1])
    
  if plot_cbar is True:
    colorbar(obj=poly)
  

  
  if plot_rhombus is True:
    pl.plot(R[[0,1],0]+abs(dR[[0,1],0]),R[[0,1],1]+abs(dR[0,1]),'-k',linewidth=1)
    pl.plot(R[[0,3],0]+abs(dR[[0,3],0]),R[[0,3],1]+abs(dR[0,1]),'-k',linewidth=1)
    pl.plot(R[[1,2],0]+abs(dR[[1,2],0]),R[[1,2],1]+abs(dR[0,1]),'-k',linewidth=1)
    pl.plot(R[[2,3],0]+abs(dR[[2,3],0]),R[[2,3],1]+abs(dR[0,1]),'-k',linewidth=1)
        
  if plot_axes is True:
    # plot first axis line
    pl.plot(R[0:2,0],R[0:2,1],'-k',linewidth=1.2)
    
    # midpoint
    mid_point=np.mean(R[0:2,:],axis=0)
    
    # tick labels
    pl.text(mid_point[0],mid_point[1]-0.2*side,'0',horizontalalignment='center')
    pl.text(R[0,0],R[0,1]-0.2*side,side_neg,horizontalalignment='center')
    pl.text(R[1,0],R[1,1]-0.2*side,side_pos,horizontalalignment='center')
    
    # ticks
    tick_start=R[0,:]
    tick_end=tick_start-np.array([0,1])*0.02*side
    pl.plot([tick_start[0],tick_end[0]],[tick_start[1],tick_end[1]],'-k',linewidth=1.2)
    
    tick_start=mid_point
    tick_end=tick_start-np.array([0,1])*0.02*side
    pl.plot([tick_start[0],tick_end[0]],[tick_start[1],tick_end[1]],'-k',linewidth=1.2)
    
    tick_start=R[1,:]
    tick_end=tick_start-np.array([0,1])*0.02*side
    pl.plot([tick_start[0],tick_end[0]],[tick_start[1],tick_end[1]],'-k',linewidth=1.2)
    
    # plot second axis line
    pl.plot(R[[0,3],0],R[[0,3],1],'-k',linewidth=1.2)
    
    # midpoint
    mid_point=np.mean(R[3:5,:],axis=0)
      
    # tick labels
    pl.text(mid_point[0]-0.06*side,mid_point[1],'0',verticalalignment='center',horizontalalignment='right')
    pl.text(R[0,0]-0.06*side,R[0,1],side_neg,verticalalignment='center',horizontalalignment='right')
    pl.text(R[3,0]-0.06*side,R[3,1],side_pos,verticalalignment='center',horizontalalignment='right')
    
    # ticks
    tick_start=R[0,:]
    tick_end=tick_start-np.array([np.sqrt(3)/2,0.5])*0.02*side
    pl.plot([tick_start[0],tick_end[0]],[tick_start[1],tick_end[1]],'-k',linewidth=1.2)
    
    tick_start=mid_point
    tick_end=tick_start-np.array([np.sqrt(3)/2,0.5])*0.02*side
    pl.plot([tick_start[0],tick_end[0]],[tick_start[1],tick_end[1]],'-k',linewidth=1.2)
    
    tick_start=R[3,:]
    tick_end=tick_start-np.array([np.sqrt(3)/2,0.5])*0.02*side
    pl.plot([tick_start[0],tick_end[0]],[tick_start[1],tick_end[1]],'-k',linewidth=1.2)
  
  return poly
  
def bar(x,ax=None,N=None,color='k',alpha=1,fill=True,edgecolor='k',linewidth=2,t=None,width=0.8):
  N=len(x) if N is None else N
  ax=pl.gca() if ax is None else ax
  custom_axes()
  
  if t is None:
    ax.bar(np.arange(N),x[0:N],color=color,alpha=alpha,fill=fill,edgecolor=edgecolor,linewidth=linewidth,align='center',width=width)  
    ax.set_xlim(np.array([0,N])-0.5)
    ax.set_xticks(np.arange(N))
  else:
    ax.bar(t,x[0:N],color=color,alpha=alpha,fill=fill,edgecolor=edgecolor,linewidth=linewidth,align='center')  
    
def plot_grid(SX,SY,x,show_cbar=True,show_axes=True,change_ticks=True,cmap='gist_yarg'):
  custom_axes()
  pl.pcolormesh(SX,SY,x,cmap=cmap,rasterized=True)
  if show_cbar is True:
    colorbar(change_ticks)
  L=int(round(np.amax(SX)))
  pl.xlim([-L/2,L/2])
  pl.ylim([-L/2,L/2])
  pl.xticks(np.arange(-L,L+1,1))
  pl.yticks(np.arange(-L,L+1,1))
  if show_axes is False:
    noframe()
  
def colorbar(change_ticks=True,obj=None,num_int=6,ax=None,cax=None,orientation='vertical',shrink=0.5,fixed_ticks=None):
  if obj is None:
    cbar=pl.colorbar(shrink=shrink,aspect=15,ax=ax,cax=cax,orientation=orientation)
  else:
    cbar=pl.colorbar(obj,shrink=shrink,aspect=15,ax=ax,cax=cax,orientation=orientation)
    
  if change_ticks is True:
    
    if fixed_ticks is not None:
      cbar.locator = FixedLocator(fixed_ticks)
    else:
      cbar.locator = MaxNLocator(num_int)
    cbar.update_ticks()
  return cbar
  
def get_barticks(data,decimals=2,vmin=None,vmax=None):
  if vmin is None:
    vmin = min(np.ravel(data))
  if vmax is None:
    vmax = max(np.ravel(data))
  factor = float(10**decimals)
  return [np.ceil(vmin*factor)/factor,round((vmax-abs(vmin))/2,decimals),np.floor(vmax*factor)/factor]


def set_axes_width(width):

  pl.rcParams['xtick.major.width'] = width
  pl.rcParams['ytick.major.width'] = width
  pl.rcParams['xtick.minor.width'] = width
  pl.rcParams['ytick.minor.width'] = width
  pl.rcParams['axes.linewidth'] = width


def set_tick_size(size,minor_offset=1):
  pl.rcParams['xtick.major.size'] = size
  pl.rcParams['ytick.major.size'] = size
  pl.rcParams['xtick.minor.size'] = size-minor_offset
  pl.rcParams['ytick.minor.size'] = size-minor_offset
  
def init_plot_conf():
  """
  Initialize customized matplolib configuration
  """
  pl.ion()
  pl.rc('font',size=14)
  pl.rc('lines',linewidth=1)
#  params = {'legend.fontsize': 18,
#            'legend.linewidth': 4}
  pl.rc('xtick',direction='out')
  pl.rc('ytick',direction='out')
  set_axes_width(1)
  set_tick_size(5)
  
  #pl.rcParams.update(params)
  #pl.rcParams['font.family'] = 'sans-serif'
  #pl.rcParams['font.sans-serif'] = ['Arial']  
  
def noframe(ax=None):
  """
  Set square axes and removes frame
  """
  ax=ax if ax is not None else pl.gca()
  ax.axes.get_yaxis().set_visible(False)
  ax.axes.get_xaxis().set_visible(False)
  ax.set_frame_on(False)


def noticks(ax=None):
  """
  Set square axes and removes frame
  """
  ax=ax if ax is not None else pl.gca()
  ax.set_xticks([])
  ax.set_yticks([])
      
def plot_point(x,y,linestyle='--k',linewidth=1):
  """
  Plots a point
  """
  xmin,xmax = pl.xlim()
  ymin,ymax = pl.ylim()
  pl.plot([x,x],[ymin,y],linestyle,linewidth=linewidth)
  pl.plot([xmin,x],[y,y],linestyle,linewidth=linewidth)
  pl.plot(x,y,'ok')

def broken_axis(ax,xlim,ylim_top,ylim_bottom,xlabel,ratio=0.75):
  
  pl.subplot(ax)
  ax.xaxis.set_ticks([])
  ax.yaxis.set_ticks([])
  ax.spines['top'].set_color('none')
  ax.spines['right'].set_color('none')
  
  l,b,w,h = ax.get_position().bounds 
    
  ax_bottom = pl.axes([l,b,w,h*(ratio-0.05)])
  ax_top = pl.axes([l,b+h*ratio,w,h*(1-ratio)])

  # zoom-in / limit the view to different portions of the data
  ax_top.set_ylim(ylim_top)   
  ax_bottom.set_ylim(ylim_bottom) 
  ax_top.set_xlim(xlim)
  ax_bottom.set_xlim(xlim)

  
  # hide the spines between ax and ax2
  ax_top.set_xticks([])
  ax_top.yaxis.set_ticks_position('left')
  ax_top.spines['bottom'].set_color('none')
  ax_top.spines['top'].set_color('none')
  ax_top.spines['right'].set_color('none')

  ax_bottom.xaxis.tick_bottom()
  ax_bottom.yaxis.set_ticks_position('left')
  ax_bottom.spines['top'].set_color('none')
  ax_bottom.spines['right'].set_color('none')
  
  ax_bottom.set_xlabel(xlabel)

  d = .015 # how big to make the diagonal lines in axes coordinates
  # arguments to pass plot, just so we don't keep repeating them
  kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
  ax_top.plot((-d,+d),(-d,+d), **kwargs)      # top-left diagonal

  kwargs.update(transform=ax_bottom.transAxes)  # switch to the bottom axes
  ax_bottom.plot((-d,+d),(1-d,1+d), **kwargs)   # bottom-left diagonal
  
  return ax_top,ax_bottom

def fix_math_font(ax=None,fontsize=20):
  """
  Fizes the font of tick labels in math mode
  """
  from matplotlib.font_manager import FontProperties
  
  ax=ax if ax is not None else pl.gca()
  for label in ax.get_xticklabels()+ax.get_yticklabels():
    if '$' in label.get_text():
      label.set_fontproperties(FontProperties(size=fontsize))
      
def custom_axes(ax=None):
  """
  Customizes axes
  """
  ax=ax if ax is not None else pl.gca()
  if 'right' in ax.spines.keys():
    ax.spines['right'].set_color('none')
  if 'top' in ax.spines.keys():  
    ax.spines['top'].set_color('none')
    
  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')

def adjust_spines(ax=None,spines=['left','bottom'],data_bounds=False,offset=5):
  """
  Adjusts axes spines
  """
  ax=ax if ax is not None else pl.gca()

  for loc, spine in ax.spines.items():
    if loc in spines:
      spine.set_position(('outward',offset)) 
      spine.set_smart_bounds(data_bounds)
    else:
      spine.set_color('none') # don't draw spine

  # turn off ticks np.where there is no spine
  if 'left' in spines:
    ax.yaxis.set_ticks_position('left')
  else:
    # no yaxis ticks
    ax.yaxis.set_ticks([])

  if 'bottom' in spines:
    ax.xaxis.set_ticks_position('bottom')
  else:
    # no xaxis ticks
    ax.xaxis.set_ticks([])
    
    
def make_color_manager(parameter_range, cmap='YlOrBr', start=0, stop=255):
    """Return color manager, which returns color based on parameter value.

    Parameters
    ----------
    parameter_range : 2-tuple
        minimum and maximum value of parameter
    cmap : str
        name of a matplotlib colormap (see matplotlib.pyplot.cm)
    start, stop: int
        limit colormap to this range (0 <= start < stop <= 255)
    """
    colormap = getattr(pl.cm, cmap)
    pmin, pmax = parameter_range
    def color_manager(val):
        """Return color based on parameter value `val`."""
        assert pmin <= val <= pmax
        val_norm = (val - pmin) * float(stop - start) / (pmax - pmin)
        idx = int(val_norm) + start
        return colormap(idx)
    return color_manager




def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
                
             
from numpy import ma


class MidPointNorm(Normalize):    
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")       
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = np.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint            
            resdat[resdat>0] /= abs(vmax - midpoint)            
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = np.array(resdat, mask=result.mask, copy=False)                

        if is_scalar:
            result = result[0]            
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if pl.cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)  
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0: 
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint
                
                
init_plot_conf()
