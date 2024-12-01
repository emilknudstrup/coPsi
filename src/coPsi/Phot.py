#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

Module to read in and manipulate photometric data.

'''

import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class Data(object):
	'''Photometric data.
	
	Read and prepare photometric data for which to calculate the stellar rotation period.

	Data constructor initialized with either filename for data in a .txt file, or by setting x (time) and y (flux) directly (and dy, error).


	:param file: Filename. Optional, default ``None``.
	:type file: str

	:param x: Time.
	:type x: array

	:param y: Flux.
	:type y: array

	:param dy: Flux error.
	:type dy: array

	:param cadence: Cadence for observations. Optional, default ``None``. If ``None`` calculated from :py:class:`getCadence()`
	:param cadence: float
	
	'''

	def __init__(self,file=None,x=np.array([]),y=np.array([]),
				dy=None,cadence=None,figsize=(10,6)):
		'''Constructor

		'''
		if file != None:
			self.readData(file)
			self.getCadence()
			# self.maxTime(tolGap=tolGap)
		else:
			self.x = x
			self.y = y
			self.dy = dy
			if cadence is None:
				self.getCadence()
			else:
				self.cadence = cadence
			# if maxT is None:
			# 	self.maxTime(tolGap=tolGap)
			# else:
			# 	self.maxT = maxT

		self.ii = 0
		self.figsize = figsize

	def readData(self,file,atzero=True):
		'''Data from .txt file

		======  ====  =====
		# time  flux  error
		======  ====  =====
		t1      fl1   efl1
		t2      fl2   efl2
		...     ...   ...
		tn      fln   efln
		======  ====  =====

		:param file: Filename.
		:type file: str

		:param atzero: Whether to set the first time stamp to zero. Optional, default ``True``.
		:type atzero: bool

		'''
		arr = np.loadtxt(file)
		ss = np.argsort(arr[:,0])

		self.x = arr[ss,0] - min(arr[ss,0]) if atzero else arr[ss,0]
		self.y = arr[ss,1]
		ncols = arr.shape[1]
		if ncols > 2:
			self.dy = arr[ss,2]
		else:
			self.dy = None

	def filterData(self,window=5001,poly=3):
		'''Savitsky-Golay filter

		Using :py:class:`scipy.signal.savgol_filter()`.

		:param window: Length of window for filter. Optional, default 5001.
		:type window: int

		:param poly: Degree for polynomial. Optional, default 3.
		:type poly: int

		'''
		self.yhat = savgol_filter(self.y,window,poly)
		self.y = self.y/self.yhat


	def getCadence(self):
		'''Calculate cadence

		Calculated as the median of the difference between timestamps.

		'''
		self.cadence = np.median(np.diff(self.x))

	def bin2cadence(self,cadence):
		'''Bin to cadence

		Bin the data to a given cadence.

		:param cadence: Cadence for observations. Unit same as ``self.x``.
		:type cadence: float

		'''

		## Bin the data
		bins = np.arange(min(self.x),max(self.x),cadence)
		digi = np.digitize(self.x,bins)
		xx = np.array([self.x[digi == ii].mean() for ii in range(1,len(bins))])
		yy = np.array([self.y[digi == ii].mean() for ii in range(1,len(bins))])
		# if len(self.dy):
		# 	dd = np.array([self.dy[digi == ii].mean() for ii in range(1,len(bins))])

		## Remove nans
		finite = np.isfinite(yy) & np.isfinite(xx)
		xx = xx[finite]
		yy = yy[finite]

		## Save the unbinned data
		self.ubx = self.x
		self.uby = self.y
		## Save the binned data
		self.x = xx
		self.y = yy
		# if len(self.dy):
		# 	self.uby = self.dy
		# 	self.dy = dd

	def appendData(self,x,y,dy=None,atzero=True):
		'''Append data

		Append data to the existing data.

		:param x: Time.
		:type x: array

		:param y: Flux.
		:type y: array

		:param dy: Flux error. Optional, default ``None``.
		:type dy: array

		'''
		x = np.append(self.x,x)
		y = np.append(self.y,y)
		ss = np.argsort(x)
		self.x = x[ss]
		self.y = y[ss]
		if dy:
			dy = np.append(self.dy,dy)
			self.dy = dy[ss]
		
		if atzero:
			self.x -= min(self.x)
		
	def maxTime(self,tolGap=5):
		'''Maximum timeseries length

		Find the maximum timeseries length between gaps exceeding ``tolGap`` days.

		:param tolGap: Value to consider a gap in days. Optional, default 5.
		:type tolGap: float

		'''

		diff = np.diff(self.x)
		gaps = diff[diff > tolGap]
		tooBig = [np.where(gap == diff)[0][0] for gap in gaps]
		lengths = []
		idx = 0
		for gap in tooBig:
			lengths.append(max(self.x[idx:gap])-min(self.x[idx:gap]))
			idx = gap + 1
		lengths.append(max(self.x[idx:])-min(self.x[idx:]))
		print('Found {} chunks with gaps exceeding {} days:'.format(len(lengths),tolGap))
		for i in range(len(lengths)):
			print('Chunk {}: {:.2f} days'.format(i,lengths[i]))
		# print('Longest timeseries between gaps: {:.2f} days'.format(max(lengths)))
		print('Longest continous timeseries: {:.2f} days'.format(max(lengths)))
		self.maxT = max(lengths)


	def fillGaps(self,gap=0.5,yfill=None,cadence=None):
		'''Fill gaps

		Fill the gaps in the data by simply putting in the median of the observations by default.

		:param gap: Value to consider a gap in days. Optional, default 0.5.
		:type gap: float

		:param yfill: Values to put in gaps. Optional, default ``None``. If ``None`` median is used.
		:type yfill: float

		:param cadence: Cadence for observations. Optional, default ``None``. If ``None`` calculated from :py:class:`getCadence()`
		:param cadence: float		

		'''
		gaps = np.where(np.diff(self.x) > gap)[0]
		
		if yfill == None:
			yfill = np.median(self.y)
		
		if not cadence:
			self.getCadence()
			cadence = self.cadence
		
		nx = self.x.copy()#np.array([])
		ny = self.y.copy()#np.array([])
		for g in gaps:
			new_x = np.arange(self.x[g],self.x[g+1],cadence)
			new_y = np.zeros_like(new_x) + yfill
			
			nx = np.append(nx,new_x[1:])
			ny = np.append(ny,new_y[1:])
		
		## Save the data with gaps
		self.xg = self.x
		self.yg = self.y

		## Sort the arrays
		ss = np.argsort(nx)
		nx = nx[ss]
		ny = ny[ss]
		#nx -= min(self.x)
		if int(min(self.x)):
			nx -= min(self.x)
		self.x = nx
		self.y = ny


	def prepData(self,file=None,window=8001,poly=3,gap=0.5,yfill=None,cadence=None):
		'''Prepare data.

		Collection of calls to :py:class:`readData()`, :py:class:`filterData()`, and :py:class:`fillGaps()`

		'''
		if file != None:
			self.readData(file)
		self.filterData(window=window,poly=poly)
		self.fillGaps(gap=gap,yfill=yfill,cadence=cadence)
		#self.filterData(**kwargs)

	def plotData(self,ax=None,dots=1,return_ax=0,font=12,usetex=False):
		'''Plot the data

		:param ax: Axis in which to plot the KDE. Optional, default ``None``, figure and axis will be created.
		:type ax: :py:class:`matplotlib.axes._subplots.AxesSubplot`

		:param dots: Whether to use dots instead of lines in plot. Optional, default 1.
		:type dots: bool

		:param return_ax: Whether to return `ax`. Optional, default 0.
		:type return_ax: bool
	
		:param font: Fontsize for labels. Optional, default 12.
		:type font: float

		:param usetex: Whether to use LaTeX in plots. Optional, default ``False``.
		:type usetex: bool

		'''
		if not ax:
			plt.rc('text',usetex=usetex)
			fig = plt.figure(figsize=self.figsize)
			ax = fig.add_subplot(111)
		if dots:
			ax.plot(self.x-min(self.x),self.y,marker='o',markersize=2,color='k',ls='none')
			ax.plot(self.x-min(self.x),self.y,marker='o',markersize=1,color='C{}'.format(self.ii),ls='none')
		else:
			ax.plot(self.x,self.y,color='k',lw=0.5)

		ax.set_xlabel(r'$\rm Time \ (days)$',fontsize=font)
		ax.set_ylabel(r'$\rm Relative \ brightness$',fontsize=font)
		ax.tick_params(axis='both',labelsize=font)

		self.ii += 1
		if return_ax: return ax

	# def to_rotator(self):
	# 	pass
		