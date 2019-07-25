import numpy as np
from numpy.fft import fft2,fftfreq,fftshift
from numpy.random import rand
from scipy.ndimage import rotate
from scipy.signal import fftconvolve
from scipy.stats.stats import pearsonr
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from simlib import print_progress
from time import clock

  
def get_angle_amps(num_dft,freq_idx,nx):
  """
  compute the amplitudes of modes of the same frequencies but different angles
  """
  
  # all radii and all angles  
  yr, xr = np.indices((nx,nx))
  all_r =  np.around( np.sqrt((xr - nx/2)**2 + (yr - nx/2)**2))
  all_ang =  np.arctan2(yr-nx/2,xr-nx/2)*180/ np.pi+180
  all_ang[all_ang==360]=0
  
  # flat dfts
  allr_flat=all_r.reshape(nx**2)
  all_ang_flat = all_ang.reshape(nx**2)
  num_dft_flat=num_dft.reshape(nx**2,num_dft.shape[2])

  # indexes
  idxs= np.arange(nx**2)[allr_flat==freq_idx]
  uidxs=idxs[0:len(idxs)/2]
  
  # take as zero the angle of the fastest growing mode
  max_ang=all_ang_flat[ np.argmax(num_dft_flat[:,-1])]
  angles=np.remainder(all_ang_flat-max_ang,180)[uidxs]

  # amplitudes
  amps=  [ np.squeeze(num_dft_flat[idx,:]) for idx in uidxs]
  
  return angles,amps
  
def get_grid_params(J,L,n,num_steps=50,return_cx=False):
  
  dx=L/n
  X,Y=np.mgrid[-L/2:L/2:dx,-L/2:L/2:dx]
  pos=np.array([np.ravel(X), np.ravel(Y)])
  
  if return_cx is True:
    score,best_outr,angle,spacing,cx=gridness(J,L/n,
                                           computeAngle=True,doPlot=False,
                                           num_steps=num_steps,return_cx=True)   
  else:
    score,best_outr,angle,spacing=gridness(J,L/n,
                                           computeAngle=True,doPlot=False,
                                           num_steps=num_steps,return_cx=False)   
                                         
  ref_grid=simple_grid_fun(pos,cos_period=spacing/2*np.sqrt(3),angle=-angle,phase=[0, 0]).reshape(n,n)
  phase=get_grid_phase(J,ref_grid,L/n,doPlot=False,use_crosscorr=True)

  if return_cx is True:
    return score, spacing,angle,phase,cx
  else:    
    return score, spacing,angle,phase
  
def compute_scores_evo(J_vect,n,L,num_steps=50):
  """
  Computes gridness scores for a matrix at different time points
  J_vect = N x num_snaps
  """
  

  num_snaps=J_vect.shape[1]
  assert(J_vect.shape[0]==n**2)
  start_clock=clock()
  best_score=-1      
  
  scores=np.zeros(num_snaps)
  spacings=np.zeros(num_snaps)
  angles=np.zeros(num_snaps)
  phases=np.zeros((2,num_snaps))
  
  for snap_idx in xrange(num_snaps):
    print_progress(snap_idx,num_snaps,start_clock=start_clock)

    J=J_vect[:,snap_idx]
    
    score,spacing,angle,phase= get_grid_params(J.reshape(n,n),L,n,num_steps=num_steps)

    best_score=max(best_score,score)
    scores[snap_idx]=score
    spacings[snap_idx]=spacing
    angles[snap_idx]=angle
    phases[:,snap_idx]=phase
    
  score_string='final_score: %.2f    best_score: %.2f    mean_score: %.2f\n'%(score,best_score,np.mean(scores))  
  print score_string

  return scores,spacings,angles,phases      

        
def dft2d_num(M_evo,L,n,nozero=True):
  """
  Computes the 2D DFT of a n x n x time_samples matrix wrt the first two dimensions.
  The DC component is set to zero
  """
  
  assert(len(M_evo.shape)==3)
  assert(M_evo.shape[0]==M_evo.shape[1])

  allfreqs = fftshift(fftfreq(n,d=L/n))
  freqs=allfreqs[n/2:]
  M_dft_evo=fftshift(abs(fft2(M_evo,axes=[0,1])),axes=[0,1])
  if nozero is True:
    M_dft_evo[n/2,n/2,:]=0
  return M_dft_evo,freqs,allfreqs

def dft2d_teo(J0_dft,eigs,time,n):
  """
  Compute the theoretical DFT solution of a linear dynamical system
  given the eigenvalues and the initial condition
  """
  N=n**2
  teo_dft=J0_dft.reshape(N,1)*np.exp(time[np.newaxis,:]*eigs.reshape(N,1))
  teo_dft=teo_dft.reshape(n,n,len(time))
  teo_dft[n/2,n/2]=0

  return teo_dft
  
  
def radial_profile(data,norm=False):
  """
  Compute radial profile of a 2D function sampled on a square domain,
  assumes the function is centered in the middle of the square

  # TEST:
  #  
  #  ran=np.arange(-1.01,1.01,0.01)
  #  SSX,SSY = meshgrid(ran,ran)
  #  
  #  T=np.exp(-(SSX**2+SSY**2))
  #  P=radial_profile(T,norm=True)
  #  
  #  pl.figure()
  #  pl.subplot(111,aspect='equal')
  #  pl.pcolormesh(SSX,SSY,T)
  #  custom_axes()
  #  colorbar()
  #  
  #  pl.figure()
  #  pl.plot(ran[101:],P)
  
  """
  
  assert(len(data.shape)==2)
  assert(data.shape[0]==data.shape[1])


  center=np.array(data.shape)/2
  yr, xr = np.indices((data.shape))
  r = np.around(np.sqrt((xr - center[0])**2 + (yr - center[1])**2))
  r = r.astype(int)

  profile =np. bincount(r.ravel(), data.ravel())
  
  if norm is True:
    nr = np.bincount(r.ravel())
    profile/=nr
    
  profile=profile[:len(data)/2]    
  
  return profile
  

def dft2d_profiles(M_dft_evo):
  
  assert(len(M_dft_evo.shape)==3)
  assert(M_dft_evo.shape[0]==M_dft_evo.shape[1])

  num_snaps=M_dft_evo.shape[2]
  profiles = np.array([radial_profile(abs(M_dft_evo[:,:,idx]),norm=True) for idx in xrange(num_snaps)])
  return profiles


def gridness_evo(M,dx,num_steps=50):
  """
  Compute gridness evolution
  """
  scores=[]
  spacings=[]
  assert(len(M.shape)==3)
  num_snaps=M.shape[2]
  print 'Computing scores...'
  for idx in xrange(num_snaps):
    score,best_outr,orientation,spacing=gridness(M[:,:,idx],dx,computeAngle=False,doPlot=False,num_steps=num_steps)
    scores.append(score)
    spacings.append(spacing)
    print_progress(idx,num_snaps)
  return scores,spacings
        
def unique_rows(a):
  """
  Removes duplicates rows from a matrix
  """
  a = np.ascontiguousnp.array(a)
  unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
  return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
  
  
def detect_peaks(cx,size=2):
  """
  Takes an image and detect the peaks usingthe local maximum filter.
  Returns a boolean mask of the peaks (i.e. 1 when
  the pixel's value is the neighborhood maximum, 0 otherwise)
  """

  # define an size-connected neighborhood
  neighborhood = generate_binary_structure(size,size)

  #apply the local maximum filter; all pixel of maximal value in their neighborhood are set to 1
  local_max = maximum_filter(cx, footprint=neighborhood)==cx
  #local_max is a mask that contains the peaks we are looking for, but also the background. In order to isolate the peaks we must remove the background from the mask.

  #we create the mask of the background
  background = (cx==0)

  #a little technicality: we must erode the background in order to successfully subtract it form local_max, otherwise a line will 
  #appear along the background border (artifact of the local maximum filter)
  eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

  #we obtain the final mask, containing only peaks, by removing the background from the local_max mask
  detected_peaks = local_max.astype(int) - eroded_background.astype(int)

  return detected_peaks

def detect_six_closest_peaks(cx,doPlot=False):
  """
  Detects the six peaks closest to the center of the autocorrelogram
  cx: autocorrelogram matrix
  """

  # indexes to cut a circle in the auto-correlation matrix
  SX,SY = np.meshgrid(range(cx.shape[0]),range(cx.shape[1]))
  
  if np.remainder(cx.shape[0],2)==1:
    tile_center = np.array([[(cx.shape[0]+1)/2-1, (cx.shape[1]+1)/2-1]]).T
  else:
    tile_center = np.array([[(cx.shape[0])/2, (cx.shape[1])/2]]).T
  

  peaks=detect_peaks(cx)
  peaks_xy = np.array([SX[peaks==1],SY[peaks==1]])
  
  if peaks_xy.shape[1]<6:
    print 'Warning: less than 6 peaks found!!!'
    return np.zeros((2,6)),tile_center
    
  else:
    peaks_dist = np.sqrt(sum((peaks_xy-tile_center)**2,0))
    
    sort_idxs = np.argsort(peaks_dist)
    
    peaks_dist=peaks_dist[sort_idxs]
    peaks_xy=peaks_xy[:,sort_idxs]
  
    # filter out center and peaks too close to the center (duplicates)
    to_retain_idxs=peaks_dist>2
    
    
    if sum(to_retain_idxs)<6:
      print 'Warning: less than 6 peaks to retain!!!'
      return np.zeros((2,6)),tile_center
      
    else:      
      peaks_dist=peaks_dist[to_retain_idxs]
      peaks_xy=peaks_xy[:,to_retain_idxs]   
      idxs=np.arange(6)
    
    
      if doPlot is True:
        import pylab as pl  
        pl.pcolormesh(cx)
        pl.scatter(peaks_xy[0,idxs]+0.5,peaks_xy[1,idxs]+0.5)
        pl.scatter(tile_center[0]+0.5,tile_center[1]+0.5)
        
      return peaks_xy[:,idxs],tile_center

    


def get_grid_phase(x,x0,ds,doPlot=False,use_crosscorr=True):
  """
  Return the grid phase relative to a given reference gridness
  x: grid for which the phase has to be estimated
  x0: reference grid
  """
  if use_crosscorr is True:
    cx=norm_crosscorr(x,x0,type='full')
    # cutout the central part
    n=(cx.shape[0]+1)/2
    cx=cx[n-1-n/2:n-1+n/2,n-1-n/2:n-1+n/2]    
  else:
    cx=x
    n=cx.shape[0]    
  
  L=n*ds
  SX,SY,tiles=get_tiles(L,ds)
  
  peaks=detect_peaks(cx)

  if peaks.sum()>0:
    peaks_xy = np.array([SX[peaks==1],SY[peaks==1]])
    peaks_dist = sum(peaks_xy**2,0)
    
    idxs = np.argmin(peaks_dist)
  
    if doPlot is True:
      import pylab as pl  
      pl.pcolormesh(SX,SY,cx)
      pl.scatter(peaks_xy[0,idxs],peaks_xy[1,idxs])
      pl.plot(0,0,'.g')
  
    phase = peaks_xy[:,idxs]
  else:
    phase=np.NaN
    
  return phase


def get_grid_spacing_and_orientation(cx,ds,doPlot=False,compute_angle=True):
  """
  Returns the grid orientation given the autocorrelogram
  ds: space discretization step
  cx: autocorrelogram matrix
  :returns: an angle in radiants
  """

  peaks,center = detect_six_closest_peaks(cx)           # get six closest peaks
  
  #print '%d peaks detected: '%peaks.shape[1]
  
  cent_peaks=peaks-center                               # center them
  #print cent_peaks
  
  peak_dists = np.sqrt(sum(cent_peaks**2,0))               # peak distances
  norm_peaks = cent_peaks/peak_dists                    # normalize to unit norm

  norm_peak_1quad_idxs=np.bitwise_and(norm_peaks[1,:]>0,norm_peaks[0,:]>0)          # indexes of the peaks in the first quadrant x>0 and y>0
  
  #print
  #print peak_dists.mean()*ds
  #print median(peak_dists)*ds
  #print ds
  #print 
  
  spacing=np.mean(peak_dists)*ds
  
  #print 'get_grid_spacing_and_orientation spacing=%.2f'%spacing
  
  angle=np.NaN

  if compute_angle is True:
    # if we have at least one peak in the first quadrant
    if any(norm_peak_1quad_idxs) == True: 
      norm_peaks_1quad=norm_peaks[:,norm_peak_1quad_idxs]                           # normalized coordinates of the peaks in the first quadrant
      norm_orientation_peak_idx=np.argmin(norm_peaks_1quad[1,:])                       # index of the peak with minumum y 
      norm_orientation_peak=norm_peaks_1quad[:,norm_orientation_peak_idx]          # normalized coordinates of the peak with minimum y
      
      peaks_1quad = peaks[:,norm_peak_1quad_idxs]                                 # coordinates of the peaks in the first quadrant
      orientation_peak=peaks_1quad[:,norm_orientation_peak_idx]                   # coordinates of the peak with minimum y 
  
      angle = np.arccos(norm_orientation_peak[0])  # calculate angle
      
      if angle <0:
        angle=angle+np.pi/3
  
      if doPlot is True:
        import pylab as pl  
        pl.pcolormesh(cx/(cx[center[0],center[1]]),vmax=1.,cmap='binary',rasterized=True)
        #pl.colorbar(shrink=0.5,aspect=15)
        #pl.scatter(orientation_peak[0]+0.5,orientation_peak[1]+0.5)
        pl.plot([center[0]+.5,orientation_peak[0]+.5],[center[1]+.5,orientation_peak[1]+.5],'-y',linewidth=2)

        for i in xrange(6):
          pl.scatter(peaks[0,i]+0.5,peaks[1,i]+0.5,c='r')
          
        pl.scatter(center[0]+0.5,center[1]+0.5,c='r')
        hlen=cx.shape[0]/3.
        pl.xlim([center[0]-hlen,center[0]+hlen])
        pl.ylim([center[1]-hlen,center[1]+hlen])
    else:
      pass
      #print "no peaks in the first quadrant"

  return angle,spacing


def fr_fun(h,gain=.1,th=0,sat=1,type='arctan'):
  """
  Threshold-saturation firing rate function 
  h: input
  sat: saturation level
  gain: gain
  th: threshold
  """
  if type == 'arctan':
    return sat*2/np.pi*np.arctan(gain*(h-th))*0.5*(np.sign(h-th) + 1)
  elif type == 'sigmoid':
    return sat*1/(1+np.exp(-gain*(h-th)))
  elif type == 'rectified':
    return h*0.5*(np.sign(h-th) + 1)
  elif type=='linear':
    return h

def pf_fun(pos,center=np.array([0,0]),sigma=0.05,amp=1):
  """
  Gaussian place-field input function
  pos: position
  center: center
  sigma: place field width
  amp: maximal amplitude
  """
  
  # multiple positions one center
  if len(pos.shape)>1 and len(center.shape)==1:
    center = np.array([center]).T
  # one position multiple centers
  if len(pos.shape)==1 and len(center.shape)>1:
   pos = np.array([pos]).T
  return np.exp(-sum((pos-center)**2,0)/(2*sigma**2))*amp

def grid_fun(pos,freq=7,angle=0,phase=[0, 0]):
  """
  Grid tuning function
  pos: x,y position or 2xN array of positions
  freq: grid spatial frequency [cycles/m]
  angle: grid orientation [rad] in the range [0,pi/3)
  phase: spatial phase 2D vector 
  """
  cos_freq=2*freq/np.sqrt(3)
  alpha=np.array([np.pi*i/3+angle+np.pi/6 for i in np.arange(3)])
  k=2*np.pi*cos_freq*np.array([np.cos(alpha),np.sin(alpha)]).T
  if len(pos.shape)>1:
    phase = np.array([phase]).T
  return sum(np.cos(np.dot(k,pos-phase)),0)

def simple_grid_fun(pos,cos_period=1,angle=0,phase=[0, 0]):
  """
  Another function for a grid with a simpler mathematical description
  """
  alpha=np.array([np.pi*i/3+angle for i in np.arange(3)])
  k=2*np.pi/cos_period*np.array([np.cos(alpha),np.sin(alpha)]).T
  if len(pos.shape)>1:
    phase = np.array([phase]).T
  return sum(np.cos(np.dot(k,pos+phase)),0)
  

def norm_crosscorr(x,y,type='full',pearson=True):
  """
  Normalized cross-correlogram
  """
  n = fftconvolve(np.ones(x.shape),np.ones(x.shape),type)
  cx=np.divide(fftconvolve(np.flipud(np.fliplr(x)),y,type),n)
  if pearson is True:
    return (cx-x.mean()**2)/x.var()
  else:
    return cx
    
def norm_autocorr(x,type='full',pearson=True):
  """
  Normalized autocorrelation, we divide about the amount of overlap which is given by the autoconvolution of a matrix of ones
  """
  x0 = x-x.mean()
  #return fftconvolve(flipud(fliplr(x)),x,type)
  n = fftconvolve(np.ones(x0.shape),np.ones(x0.shape),type)
  cx=np.divide(fftconvolve(np.flipud(np.fliplr(x0)),x0,type),n)
  if pearson is True:
    return cx/x.var()
  else:
    return cx

def comp_score(cx,idxs):
  """
  Calculates the gridness score for an autocorrelation pattern and a given array of indexes for elements to retain.
  For the final gridness score the elements shall be outside an inner radius around the central peak and inside an outer radius
  containing the six closest peaks
  cx: autocorrelogram
  idxs: array of indexes for the elements to retain
  """
  deg_ran = [60, 120, 30, 90, 150]   # angles for the gridness score    
  c = np.zeros(len(deg_ran))            # correlation for each rotation angle
  cx_in = cx[idxs[0,:],idxs[1,:]]    # elements of the autocorellation pattern to retain

  # calculate correlation for the five angles
  for deg_idx in range(len(deg_ran)):
    rot = rotate(cx,deg_ran[deg_idx],reshape=False)
    rot_in = rot[idxs[0,:],idxs[1,:]]
    c[deg_idx]=pearsonr(cx_in,rot_in)[0]

  # gridness score for this radius
  score=np.mean(c[0:2])-np.mean(c[2:]) 
  return score

def gridness(x,ds,doPlot=False,computeAngle=False,num_steps=20,score_th_for_orientation=0.3,axes=None,cx=None,pearson=True,return_cx=False):
  """
  Computes the gridness score of a pattern
  x: pattern 
  doPolt: plots the autocorrelogram and the gridness score
  """
  if cx is None:
    cx = norm_autocorr(x,pearson=pearson)                                         # compute the normalized autocorrelation of the pattern
    
  angle,spacing=get_grid_spacing_and_orientation(cx,ds,doPlot=False,compute_angle=False)

  max_outr=np.ceil(spacing*2.5/ds)
  min_outr=np.floor(spacing*0.7/ds)

    
  #max_outr = floor(cx.shape[0]/2)*2/3.                         # maximum radius for the gridness
  #min_outr = 10                                                # minimum radius for the gridness score
  outr_ran = np.arange(min_outr,max_outr,max_outr/num_steps)       # range of radii for the gridness score
  best_score = 0                                                # best gridness score
  best_outr = min_outr                                          # best radius


  # indexes to cut a circle in the auto-correlation matrix
  SX,SY = np.meshgrid(range(cx.shape[0]),range(cx.shape[1]))
  tiles= np.array([np.ravel(SX), np.ravel(SY)])
  if np.remainder(cx.shape[0],2)==1:
    tile_center = np.array([[(cx.shape[0]+1)/2-1, (cx.shape[1]+1)/2-1]]).T
  else:
    tile_center = np.array([[(cx.shape[0])/2, (cx.shape[1])/2]]).T
  tiles_dist = np.sqrt(sum((tiles-tile_center)**2,0))

  # loop over increasing radii and retain the best score
  for outr_idx in range(len(outr_ran)):

    # compute score for the current outer radius
    idxs=tiles[:,tiles_dist<outr_ran[outr_idx]]
    score = comp_score(cx,idxs)

    # retain best score
    if score > best_score:
      best_score = score
      best_outr = outr_ran[outr_idx]

  # take as inner radius half of the outer radius and recompute the score
  in_r = best_outr/2
  idxs= tiles[:,np.logical_and(tiles_dist>in_r,tiles_dist<best_outr)]
  best_score = comp_score(cx,idxs)

  # plot if requested
  if doPlot is True:
    import pylab as pl      
    ax=pl.gca() if axes is None else axes
    pl.sca(ax)
    pl.axis('equal')
    pl.pcolormesh(cx,rasterized=True)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    theta_ran = np.arange(0,2*np.pi,0.1)
    pl.plot(best_outr*np.cos(theta_ran)+tile_center[0],best_outr*np.sin(theta_ran)+tile_center[1],'w')
    pl.plot(max_outr*np.cos(theta_ran)+tile_center[0],max_outr*np.sin(theta_ran)+tile_center[1],'k')
    pl.plot((spacing/ds)*np.cos(theta_ran)+tile_center[0],(spacing/ds)*np.sin(theta_ran)+tile_center[1],'g')
    pl.plot(in_r*np.cos(theta_ran)+tile_center[0],in_r*np.sin(theta_ran)+tile_center[1],'w')
    pl.text(10,10,'%.2f'%best_score, color='black',fontsize=10, weight='bold',bbox={'facecolor':'white'})

  # calculate angle if there we pass a threshold for the gridness
  angle=np.NaN
  if computeAngle is True and best_score > score_th_for_orientation:
    angle,spacing = get_grid_spacing_and_orientation(cx,ds,compute_angle=True)
  
  if return_cx  is True:
    return best_score,best_outr,angle,spacing,cx
  else:
    return best_score,best_outr,angle,spacing

def get_tiles(L=0.6,ds=0.01):
  """
  Returns the positions of the vertices of a square grid of side length L.
  The parameter ds indicates the grid spacing.
  """
  SX,SY = np.meshgrid(np.arange(-L/2.,L/2.,ds),np.arange(-L/2.,L/2.,ds))
  tiles= np.array([np.ravel(SX), np.ravel(SY)])
  return SX,SY,tiles
  
def get_tiles_int(L=0.6,num_samp=200):
  """
  Returns the positions of the vertices of a square grid of side length L.
  The parameter ds indicates the grid spacing.
  """
  samples=np.arange(-num_samp/2,num_samp/2)/float(num_samp)*L
  SX,SY = np.meshgrid(samples,samples)
  tiles= np.array([np.ravel(SX), np.ravel(SY)])
  return SX,SY,tiles
  
def divide_triangle(parent_triangle,grid_vertices,level=0,max_level=3,prec=9):
  """
  Recursively tassellates an equilateral triangles. 
  parent_triangle: input list of the vertices of the triangle to tasselate
  grid_vertices: output set of the vertices of the tassellated grid
  level: current level of the recursion
  max_level: desired level of recursive tassellation  
  The algorithm works like this. The triangle is divided in four equilateral
  child triangles by taking the midpoints of its edges. The central child triangle
  has vertices given by the thee midpoints, which are computed by the function
  get_child_triangle. Than the vertices of the other three sibling triangles are
  computed by the function get_sibling_triangles. After this subdivision,
  the function is called recursively for generated child triangle.
  """
  
  # the two main functions of the algorithm  
  get_child_triangle = lambda parent_triangle: [ tuple(np.around(0.5*(np.array(parent_triangle[i])+np.array(parent_triangle[i-1])),6)) for i in range(3) ]
  get_sibling_triangles = lambda parent_triangle,child_triangle: [ [ parent_triangle[p-3], child_triangle[p-3], child_triangle[p-2] ] for p in range(3)]

  child_triangle=get_child_triangle(parent_triangle)   # get the central child triangle
  [grid_vertices.add(v) for v in child_triangle if v not in grid_vertices]       # add it to the final set of vertices
  child_triangles=[child_triangle]

  if level<max_level:  
    child_triangles+= get_sibling_triangles(parent_triangle,child_triangle)
    for new_parent_triangle in child_triangles:
      divide_triangle(new_parent_triangle,grid_vertices,level+1,max_level,prec)
  else:
    return 
  
#def get_hex_tiling(side,angle,max_level=2,prec=9):
#  """
#  Returns a triangular tiling of an hexagonal space. The hexagon which defines
#  the space has a given side and angle. The tiling algorithm is recursive, and 
#  the parameter max_level controls the level of recursion at which the algorithm
#  shall stop. The parameter prec indicates the number of decimal positions considered
#  for the precision of the vertex coordinates.
#  """
#  
#  # we join the results of tiling the 6 equilateral triangles the compose 
#  # this smalle hexagon
#  for i in range(6):  
#    
#    # the three vertices of the triangle to tile
#    v0 = (0,0)
#    v1 = tuple(np.around(side*np.array((np.cos(angle-np.pi/6+np.pi/3*i),np.sin(angle-np.pi/6+np.pi/3*i))),prec))
#    v2 = tuple(np.around(side*np.array((np.cos(angle+np.pi/6+np.pi/3*i),np.sin(angle+np.pi/6+np.pi/3*i))),prec))#(l*cos(angle+pi/6+pi/3*i),l*sin(angle+pi/6+pi/3*i))
#    
#    # tile the triangle and add the vertices to the global set
#    parent_triangle = (v0,v1,v2)
#    [grid_vertices.add(v) for v in parent_triangle if not v in grid_vertices]
#    divide_triangle(parent_triangle,grid_vertices,max_level=max_level) 
#  
#  # remove central vertex
#  print 'num_phases: %d'%len(grid_vertices)
#   
#  grid_vertices=np.array(list(grid_vertices))
#  
#  return grid_vertices
 
 
def get_all_phases(freq,angle,num_phases=100):
  """
  Returns a set of phases evenly distributed within the whole phase space
  """
  # the elementary phase space is an hexagon   
  side=np.sqrt(3)/(3*freq)
  hexagon=get_hexagon(side,angle)

  # first we get the phases uniformly spaced on a parallelogram
  axes=(0,1)
  phases=get_phases_on_pgram(freq,angle,num_phases,axes=axes)
  
  # then we shift these phases by +-lshift in the direction of the largest
  # diagonal of the parralelogram
  lshift=np.sqrt(3)/(6*freq)
  alpha1=angle+np.pi/6+axes[0]*np.pi/3
  alpha2=angle+np.pi/6+axes[1]*np.pi/3
  shift= lshift*(np.array([np.cos(alpha1)+np.cos(alpha2),np.sin(alpha1)+np.sin(alpha2)]))
  phases_shift1=phases+np.array([shift])
  phases_shift2=phases-np.array([shift])
  
  # we stack the three set of phases obtained
  all_phases=np.vstack((phases,phases_shift1,phases_shift2))
  
  # we discard the phases outside the elementary phase space  
  idxs=points_inside_poly(all_phases,hexagon)
  all_phases=all_phases[idxs,:]
  return all_phases
  
  
  
def get_hexagon(side,angle):
  """
  Returns the vertices of an hexagon of a given side length and oriented according
  to a given angle. The first and the last vertices are the same (this is to have)
  a closed line whan plotting the hexagon.
  """
  verts=np.zeros((7,2))
  for i in range(7):
    alpha=angle+np.pi/6+i*np.pi/3
    verts[i,0]=side*np.cos(alpha)
    verts[i,1]=side*np.sin(alpha)
  return verts
  
def get_rhombus(side,angle=np.pi/6):
  """
  Returns the vertices of a rhombus with edges oriented 60 degrees apart. 
  The first and the last vertices are the same (this is to have)
  a closed line whan plotting the polygon.
  """
  verts=np.zeros((5,2))
  verts[1,:]=side*np.array([np.cos(angle),np.sin(angle)])
  verts[3,:]=side*np.array([np.cos(angle+np.pi/3),np.sin(angle+np.pi/3)])
  verts[2,:]=verts[1,:]+verts[3,:]
  center=(verts[1,:]+verts[3,:])/2
  verts-=center
  return verts
  
def get_simple_hexagon(side,angle):
  """
  Same as get_hexagon but without the pi/6 offset in the orientation
  """
  verts=np.zeros((7,2))
  for i in range(7):
    alpha=angle+i*np.pi/3
    verts[i,0]=side*np.cos(alpha)
    verts[i,1]=side*np.sin(alpha)
  return verts
  
#def plot_hexagon(side,angle,color=np.array([250,150,0])/255.):
#  verts=get_hexagon(side,angle)
#  #pl.plot(verts[:,0],verts[:,1],color=color,linewidth=2)
  
  
def get_phases_on_pgram(freq,angle,num_phases=36,axes=(0,1)):
  """
  Returns a set of phases uniformely sampled within a parallelogram.
  The parallelogram is the space spanned by two vectors oriented as two
  of the three grid axes and having length equal to double the period of the 
  cosine waves that form the grid.
  """
  # period of the cosines of the grid with the given parameter
  l=np.sqrt(3)/(2*freq)
  dl =l/(np.sqrt(num_phases)/2)
  ran=np.arange(-l,l,dl)+dl/2
  
  # the angles of the two axes    
  alpha1=angle+np.pi/6+axes[0]*np.pi/3
  alpha2=angle+np.pi/6+axes[1]*np.pi/3
   
  # points on the first axis
  x_phases1=np.cos(alpha1)*ran
  y_phases1=np.sin(alpha1)*ran

  # points on the first axis
  x_phases2=np.cos(alpha2)*ran
  y_phases2=np.sin(alpha2)*ran

  # points spanned by the two axes  
  X1,X2=np.meshgrid(x_phases1,x_phases2)
  Y1,Y2=np.meshgrid(y_phases1,y_phases2)
  X,Y=X1+X2,Y1+Y2
  phases = np.array([np.ravel(X), np.ravel(Y)]).T
  return phases   
    
def get_phases_on_axes(freq,angle,num_phases=60,axes=(0,1,2)):
  """
  Returns a set of phases such that the sum of grids with these phases is 
  flat, i.e., all grids cancel out. This is obtained by sampling phases on
  three lines with a length that is the double of the cosine
  period. The three lines are 60 degrees apart and are tilted by 90 degrees
  with respect to the original grid angle.
  """  
  # period of the cosines of the grid with the given parameter
  l=np.sqrt(3)/(2*freq)

  phases_per_axis = num_phases/len(axes)
  dl =l/phases_per_axis
  ran=np.arange(-l,l,dl)+dl/2

  x_phases = np.array([]) 
  y_phases = np.array([])   

  for i in axes:
    x_phases=np.concatenate((x_phases,np.cos(angle+np.pi/6+i*np.pi/3)*ran))
    y_phases=np.concatenate((y_phases,np.sin(angle+np.pi/6+i*np.pi/3)*ran))
    
  phases = np.array((x_phases,y_phases)).T
  return phases
  
  
################################
#### 2D GRIDS AND LATTICE ######
################################


  

def fourier_on_lattice_slow(side,p1_rec,p2_rec,samples,signal,num_comp=5):
  ran = range(-num_comp,num_comp+1)
    
  # output FT matrix 
  F = np.zeros((num_comp*2+1,num_comp*2+1),dtype=complex)
 
  s1 = np.dot(samples,p1_rec)
  s2 = np.dot(samples,p2_rec)
  
  # loop over Fourier components
  for a in ran:
    for b in ran:      
      F[a+num_comp,b+num_comp]=np.dot(signal,np.exp(-1j*(a*s1+b*s2)))

  # normalize      
  V=side*side*np.sqrt(3)/2.
  F=F*V/len(samples)
  
  return F
  
def fourier_on_lattice(side,p1_rec,p2_rec,samples,signal,num_comp=5):
  """
  Numerical Fourier series over Bravais lattice
  side: side-length of the rhomboidal primary cell of the direct lattice
  p1_rec: primary vector of the reciprocal lattice
  p2_rec: primary vector of the reciprocal lattice
  samples: grid samples in the lattice
  signal: signal of which the Fourier transform shaould be taken
  num_comp: number of Fourier coefficients to estimate
  """
  
  ran = range(-num_comp,num_comp+1)
        
  s1 = np.dot(samples,p1_rec)
  s2 = np.dot(samples,p2_rec)
  
  s12 = np.array([s1,s2])
  A,B = np.meshgrid(ran,ran)
  ab= np.array([np.ravel(A), np.ravel(B)]).T
 
  F= np.dot(np.exp(-1j*np.dot(ab,s12)),signal)
  
  if len(signal.shape)>1:
    F=F.reshape(num_comp*2+1,num_comp*2+1,signal.shape[1])
  else:
    F=F.reshape(num_comp*2+1,num_comp*2+1)
        
  # normalize      
  V=side*side*np.sqrt(3)/2.
  F=F*V/len(samples)
  
  return F        

def inverse_fourier_on_lattice(side,p1_rec,p2_rec,samples,F):
  num_comp =(F.shape[0]-1)/2 
  num_inst = F.shape[2]
  
  ran = range(-num_comp,num_comp+1)

  s1 = np.dot(samples,p1_rec)
  s2 = np.dot(samples,p2_rec)
  
  s12 = np.array([s1,s2])
  A,B = np.meshgrid(ran,ran)
  ab= np.array([np.ravel(A), np.ravel(B)]).T
  
  F=F.reshape((num_comp*2+1)**2,num_inst)
 # signal = zeros(len(samples),dtype=complex)
  signal= np.dot(F.T,np.exp(1j*np.dot(ab,s12)))

#  for a in ran:
#    for b in ran:  
#      signal+=F[a+num_comp,b+num_comp]*np.exp(1j*(a*dot(samples,p1_rec)+b*dot(samples,p2_rec)))
      
  # normalize      
  V=side*side*np.sqrt(3)/2
  signal=np.real(signal)/V
  
  return signal
  

  
  
#################
#### TESTING ####
#################



def test_orientation_and_spacing_detection():
  """
  A function to test grid orientation detection
  """
  import pylab as pl  

  ang_range=np.arange(0,np.pi/3,np.pi/3/25)
  grid_freq=3
  grid_spacing=1./grid_freq
  max_grid_phase_x = 2.0/grid_freq        
  max_grid_phase_y = 1.0/grid_freq*np.sqrt(3)
  phases = np.zeros((2,25)).T
  phases[:,0] = rand(25)*max_grid_phase_x
  phases[:,1] = rand(25)*max_grid_phase_y

  L=0.6
  ds = 0.01
  SX,SY,tiles=get_tiles(L,ds)
  
  pl.figure(figsize=(10,10))
  idx=1
  for ang in ang_range:
    ax=pl.subplot(5,5,idx,aspect='equal')
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_frame_on(False)
    x=grid_fun(tiles,angle=ang,freq=3,phase=phases[idx-1,:])
    x=x.reshape(SX.shape)
    cx=norm_autocorr(x)
    pl.axis('equal')    
    est_angle,est_spacing=get_grid_spacing_and_orientation(cx,ds,doPlot=True)
    idx+=1
    ang_deg=ang*360/(2*np.pi)
    est_angle_deg=est_angle*360/(2*np.pi)
    pl.text(20,20,'Real Ang.: %.2f\nEst Ang.: %.2f'%(ang_deg,est_angle_deg),fontsize=9,color='k',weight='bold',bbox={'facecolor':'white','edgecolor':'white'})
    pl.text(20,80,'Real Sp.: %.2f\nEst Sp.: %.2f'%(grid_spacing,est_spacing),fontsize=9,color='k',weight='bold',bbox={'facecolor':'white'})
  pl.subplots_adjust(hspace=0.1,wspace=0.1,left=0.05,right=0.95,top=0.95,bottom=0.05)



def get_phase_lattice(T,grid_angle):  
  
  # phase space
  R_T=get_rhombus(T,np.pi/6+grid_angle)
  
  # unit vectors of the direct lattice
  u1 = T*np.array([np.sin(2*np.pi/3+grid_angle), -np.cos(2*np.pi/3+grid_angle)])
  u2 = T*np.array([-np.sin(grid_angle), np.cos(grid_angle)])
  
  
  U = np.vstack([u1, u2]).T
  U_rec = 2*np.pi*(np.linalg.inv(U)).T
  
  # unit vectors of the reciprocal lattice
  u1_rec = U_rec[:,0]
  u2_rec = U_rec[:,1]

  return R_T,u1,u2,u1_rec,u2_rec


def get_phase_samples(n,u1,u2):

  # phase samples
  ran = np.arange(-n/2.,n/2.)/n
  u1_phases = np.array([u1])*ran[:,np.newaxis]
  u2_phases = np.array([u2])*ran[:,np.newaxis]
  
  X1,X2=np.meshgrid(u1_phases[:,0],u2_phases[:,0])
  Y1,Y2=np.meshgrid(u1_phases[:,1],u2_phases[:,1])
  X,Y=X1+X2,Y1+Y2
  phases = np.array([np.ravel(X), np.ravel(Y)]).T
  return phases

  
  
def get_space_samples(nx,L):
  
  # space samples
  ran = np.arange(-nx/2.,nx/2.)/nx
  
  # ortogonal unit vectors for space
  v1=(L)*np.array([0,1])
  v2=(L)*np.array([1,0])
  
  v1_pos = np.array([v1])*ran[:,np.newaxis]
  v2_pos = np.array([v2])*ran[:,np.newaxis]
  
  
  X1,X2=np.meshgrid(v1_pos[:,0],v2_pos[:,0])
  Y1,Y2=np.meshgrid(v1_pos[:,1],v2_pos[:,1])
  X,Y=X1+X2,Y1+Y2
  pos = np.array([np.ravel(X), np.ravel(Y)]).T
  
  return pos



def get_square_signal(N,NX,pos,phases,T):
  
  N=len(phases)
  NX=len(pos)

  angles=np.array([np.pi/2*i for i in np.arange(2)])
  k=2*np.pi/T*np.array([np.cos(angles),np.sin(angles)]).T
  
  pos_x = pos[:,0]
  pos_y = pos[:,1]
  
  phases_x = phases[:,0]
  phases_y = phases[:,1]
  
  pp_x = pos_x[np.newaxis,:]+phases_x[:,np.newaxis]
  pp_y = pos_y[np.newaxis,:]+phases_y[:,np.newaxis]
  
  g=np.zeros((N,NX))
  
  for i in range(2):
    g+=np.cos(k[i,0]*pp_x+k[i,1]*pp_y)  
  return g  


def get_grid_signal(N,NX,pos,phases,T,grid_angle):
  
  N=len(phases)
  NX=len(pos)
  T_cos = T/2*np.sqrt(3) 
  
  angles=np.array([np.pi*i/3+grid_angle for i in np.arange(3)])
  k=2*np.pi/T_cos*np.array([np.cos(angles),np.sin(angles)]).T
  
  pos_x = pos[:,0]
  pos_y = pos[:,1]
  
  phases_x = phases[:,0]
  phases_y = phases[:,1]
  
  pp_x = pos_x[np.newaxis,:]+phases_x[:,np.newaxis]
  pp_y = pos_y[np.newaxis,:]+phases_y[:,np.newaxis]
  
  g=np.zeros((N,NX))
  
  for i in range(3):
    g+=np.cos(k[i,0]*pp_x+k[i,1]*pp_y)  
  return g  

  

def get_filt_noise(N,NX,pos,mean,variance,sigma_x):
  
  nx=np.int(np.sqrt(NX))
  xi_raw=np.sqrt(variance)*np.random.randn(N,NX)                      
  xi_raw=xi_raw.reshape(N,np.int(nx),np.int(nx))

  if sigma_x>0:  
    filt_x=np.exp(-np.sum(pos**2,1)/(2*sigma_x**2))
    filt_x=filt_x.reshape(nx,nx)  
    filt_x_ft=np.fft.fft2(filt_x)
    xi_ft = np.fft.fft2(xi_raw)
    xi_filt_ft= np.multiply(xi_ft,filt_x_ft[np.newaxis,:,:])
    xi = np.real(np.fft.ifft2(xi_filt_ft))
  else:  
    xi=xi_raw
  
  xi=xi/np.sqrt(np.var(xi))*np.sqrt(variance)
  xi+=mean
  xi=xi.reshape(N,NX)
  
  return xi
  


    
#def test_phase_detection():
#  """
#  Tests phase detection
#  """
#  import pylab as pl  
#  L=4.
#  ds = 0.01
#  SX,SY,tiles=get_tiles(L,ds)
#  
#  grid_freq=1
#  angle=0
#  
#  phases= get_all_phases(grid_freq,angle)
#  #phases = get_phases_on_axes(grid_freq,angle,20,axes=(1,))
#  #phases = get_phases_on_pgram(grid_freq,angle,num_phases=100)
#  num_phases = phases.shape[0]
#  
#  # reference grid with zero phase  
#  x0=grid_fun(tiles,angle=angle,freq=grid_freq,phase=[0.,0.])
#  x0=x0.reshape(SX.shape)
#
#  est_phases = np.zeros((num_phases,2))
#
#  tot=np.zeros_like(x0)
#  est_tot=np.zeros_like(x0)
#  
#  for idx in range(num_phases):
#    x=grid_fun(tiles,angle=angle,freq=grid_freq,phase=phases[idx,:])
#    x=x.reshape(SX.shape)
#    est_phase=get_grid_phase(x,x0,ds,doPlot=False,use_crosscorr=False)
#    est_x=grid_fun(tiles,angle=angle,freq=grid_freq,phase=phases[idx,:])
#    est_x=est_x.reshape(SX.shape)
#    est_phases[idx,:]=est_phase
#    tot=tot+x
#    est_tot=est_tot+est_x
#      
#  pl.figure(figsize=(10,10))
# 
#  pl.subplot(111,aspect='equal')
#  #plot_grid(SX,SY,tot,change_ticks=False)
#  
#  pl.xlim([-1.2,1.2])
#  pl.ylim([-1.2,1.2])
#
#  plot_hexagon(np.sqrt(3)/(3*grid_freq),angle)
#  
#  pl.scatter(est_phases[:,0],est_phases[:,1],color='b')
#  pl.scatter(phases[:,0],phases[:,1],color='r')
