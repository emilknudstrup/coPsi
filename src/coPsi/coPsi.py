#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# external modules
# =============================================================================
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd
import emcee

# =============================================================================
# conversions
# =============================================================================
dayfac = 86400
rsunfac = 695700.0

# =============================================================================
# coPsi modules
# =============================================================================

from .priors import *

# =============================================================================
# fancy legend
# =============================================================================

from matplotlib.legend_handler import HandlerBase
class AnyObjectHandler(HandlerBase):
	'''Fancy legend
	
	Adds black borders to the lines in the legend.
	

	'''
	def create_artists(self, legend, handles,
					x0, y0, width, height, fontsize, trans,twocolor=True):

		l1 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height],
					linestyle='-', color=handles[2],lw=3.0,zorder=-1)
		l2 = plt.Line2D([x0,y0+width], [0.5*height,0.5*height],
					linestyle=handles[1], color=handles[0],lw=1.5,zorder=-1)          
		return [l1, l2] 

# =============================================================================
# Stellar inclination/obliquity class
# =============================================================================


class iStar(object):
	'''Stellar inclination.


	The variables in the constructor are tuples like (:math:`\mu,\sigma,a,b`,'distribution') used to create distributions,
	where :math:`\mu` is the mean/median, :math:`\sigma` is the standard deviation, :math:`a` is the lower boundary, and :math:`b` is the upper boundary.
	'distribution' is the type of distribution, which can be 'gauss', 'uniform', and 'tgauss' (truncated gaussian).

	:param inco: The orbital inclination (deg).
	:type inco: tuple

	:param incs: The stellar inclination (deg).
	:type incs: tuple

	:param lam: The projected obliquity (deg).
	:type lam: tuple
	
	:param Prot: Stellar rotation period (days).
	:type Prot: tuple

	:param vsini: Projected stellar rotation speed (km/s).
	:type vsini: tuple

	:param cosi: Cosine of stellar inclination. Used in :py:class:`stellarInclination`.
	:type cosi: tuple

	:param Rs: Stellar radius (:math:`R_\odot`).
	:type Rs: tuple

	
	:param parameters: List of parameters for which we can create distributions.
	:type parameters: list

	:param stepParameters: List of parameters to step in for MCMC.
	:type stepParameters: list

	:param labels: Dictionary to map labels for parameters.
	:type labels: dict

	:param cLouden: Dictionary with coefficients for :py:class:`stellarInclinationLouden`.
	:type cLouden: dict


	'''
	parameters = [#'incs', 
				'inco', 
				'lam', 
				'Prot', 
				'Rs', 
				'vsini',
				'Teff',
				'psi'
            ]

	stepParameters = [
				'Rs',
				'Prot',
				'cosi'
                ]

	labels = {
			'Rs'    : r'$R_\star \ \ (R_\odot)$',
			'Prot'  : r'$P_{\rm rot}\ \rm (days)$',
			'cosi'  : r'$\cos i_\star$',
			'incs'  : r'$i_\star \ \rm (deg)$',
			'v'     : r'$v \ \rm (km/s)$',
			'vsini' : r'$v \sin i_\star \ \rm (km/s)$',
			'psi'  : r'$\psi \ \rm (deg)$',
			}

	def __init__(self,
				inco = (85.,0.5,0.0,90.,'gauss'),
				lam = (0,0.5,-180.,180.,'gauss'),
				Prot = (3.5,0.5,0.0,10.,'gauss'),
				Rs = (1.0,0.1,0.5,2.0,'gauss'),
				vsini = (4.5,0.5,0.0,10.,'gauss'),
				cosi = (0,0.1,-1.0,1.0,'uniform'),
				Teff = (6250,100,3000,9000,'gauss'),
				figsize=(10,6)
				):
		'''Constructor

		'''
		self.inco = inco
		self.lam = lam
		self.Prot = Prot
		self.Rs = Rs
		self.vsini = vsini
		self.cosi = cosi
		self.Teff = Teff
		self.cLouden = {
					'single' : {
						'c0' : [9.57,0.29],
						'c1' : [8.01,0.54],
						'c2' : [3.30,0.62],
						'sini' : [0.856,0.036]
						},
					'two' : {
						'c0' : [9.44,0.28],
						'c1' : [8.87,0.61],
						'c2' : [4.05,0.62],
						'sini' : [0.794,0.052],
						'sini_down' : [0.928,0.042]
						}
					}
		self.figsize = figsize

	#def coPsi(self,inco,incs,lam,return_psi=True):
	def coPsi(self):
		'''Calculate :math:`\psi`.
		
		:math:`\psi` and :math:`\cos \psi` are calculated from

		.. math::
			\cos \psi = \sin i_\star \sin i_{\\rm o} \cos \lambda + \cos i_\star \cos i_{\\rm o} \, ,
		
		where :math:`\lambda` is the projected obliquity, :math:`i_\star` is the stellar inclination, and :math:`i_{\\rm o}` is the orbital incliation.

		
		Distributions for :math:`\psi` and :math:`\cos \psi` can be accessed through coPsi.iStar.dist['psi'] respectively coPsi.iStar.dist['cosp'].

		'''
		inco = np.deg2rad(self.dist['inco'])
		incs = np.deg2rad(self.dist['incs'])
		lam = np.deg2rad(self.dist['lam'])
		
		self.dist['cosp'] = np.sin(incs)*np.sin(inco)*np.cos(lam) + np.cos(incs)*np.cos(inco)
		self.dist['psi'] = np.rad2deg(np.arccos(self.dist['cosp']))
    

	def stellarInclinationDirectly(self,convert=True):
		'''Stellar inclination directly.
		
		Calculate the stellar inclination from the simple relation

		.. math::
			i_\star = \\frac{P_{\\rm rot} v \sin i_\star}{2 \pi R_\star} \, ,

		where :math:`P_{\\rm rot}` is the stellar rotation period obliquity, :math:`R_\star` is the stellar radius, and :math:`v \sin i_\star` is the projected stellar rotation speed.
		
		.. note::
			This assumes that rotation speed at the equator, :math:`v`, and the projected rotation speed, :math:`v \\sin i_\\star`, are independent, which they are not.

		
		Distribution for :math:`i_\star` can be accessed through coPsi.iStar.dist['incs'].

		:param convert: Whether to convert stellar radius from :math:`R_\odot` to km and :math:`P_{\\rm rot}` from days to seconds.
		:type convert: bool, optional. Default ``True``.

		'''
		try:
			self.dist
		except AttributeError:
			print('Distributions not initilialized.\nCalling iStar.createDistributions.')
			self.createDistributions()
		
		if convert:
			self.dist['Rs'] = self.dist['Rs']*u.R_sun
			self.dist['Rs'] = self.dist['Rs'].to_value('km')
			self.dist['Prot'] = self.dist['Prot']*u.d
			self.dist['Prot'] = self.dist['Prot'].to_value('s')


		si = self.dist['Prot']*self.dist['vsini']/(2*np.pi*self.dist['Rs'])
		incs = np.arcsin(si)
		self.dist['incs'] = np.rad2deg(incs)

	def stellarInclinationLouden(self,oblDist='two'):
		'''Stellar inclination relation

		Calculate stellar inclination using relation from :cite:t:`Louden2021`.

		:param oblDist: The assumed obliquity distribution - 'single' or 'two'. Optional, default 'two'.
		:type oblDist: str
		
		'''
		try:
			self.dist
		except AttributeError:
			print('Distributions not initilialized.\nCalling iStar.createDistributions.')
			self.createDistributions()
		
		N = len(self.dist['Teff'])
		c0 = np.random.normal(self.cLouden[oblDist]['c0'][0],self.cLouden[oblDist]['c0'][1],N)
		c1 = np.random.normal(self.cLouden[oblDist]['c1'][0],self.cLouden[oblDist]['c1'][1],N)
		c2 = np.random.normal(self.cLouden[oblDist]['c2'][0],self.cLouden[oblDist]['c2'][1],N)
		tau = (self.dist['Teff']-6250)/300
		v_avg = c0 + c1*tau + c2*tau**2
		vs = self.dist['vsini']
		si = vs/v_avg
		incs = np.rad2deg(np.arcsin(si))
		incs = incs[np.isfinite(incs)]
		self.dist['incs'] = incs


	def plotLouden(self,teffs=[5700,6700,1000],inclinations=[90,45,30,15],
					ax=None,oblDist='single',Teff=None,sTeff=0,vsini=None,svsini=0.,
					usetex=False,font=12,ymax=25,xmax=6700):
		'''Plot Louden relations.

		Plot the :math:`T_{\\rm eff}`, :math:`v \sin i_\star` from :cite:t:`Louden2021`. Compare by providing values for :math:`T_{\\rm eff}`,:math:`v \sin i_\star`.
		
		:param teffs: Grid values for :math:`T_{\\rm eff}` - [start,end,npoints]. Optional, ``[5700,6700,1000]``.
		:type teffs: list

		:param inclinations: Values to plot for :math:`i_\star`. Optional, default ``[90,45,30,15]``.
		:type inclinations: list

		:param ax: Axis in which to plot the KDE. Optional, default ``None``, figure and axis will be created.
		:type ax: :py:class:`matplotlib.axes._subplots.AxesSubplot`

		:param oblDist: The assumed obliquity distribution - 'single' or 'two'. Optional, default 'single'.
		:type oblDist: str

		:param Teff: :math:`T_{\\rm eff}`. Optional, default ``None``.
		:type Teff: float

		:param sTeff: Error on :math:`T_{\\rm eff}`. Optional, default 0.
		:type sTeff: float

		:param vini: :math:`v \sin i_\star`. Optional, default ``None``.
		:type vini: float

		:param svini: Error on :math:`v \sin i_\star`. Optional, default 0.
		:type svini: float

		:param usetex: Whether to use LaTeX in plots. Optional, default ``False``.
		:type usetex: bool

		:param font: Fontsize for labels. Optional, default 12.
		:type font: float

		:param ymax: Maximum value for :math:`v \sin i_\star`. Optional, default 25.
		:type ymax: float

		:param xmax: Maximum value for :math:`T_{\\rm eff}`. Optional, default 6700.
		:type xmax: float



		'''
		teffs = np.linspace(teffs[0],teffs[1],teffs[2])

		if not ax:
			plt.rc('text',usetex=usetex)
			fig = plt.figure(figsize=self.figsize)
			ax = fig.add_subplot(111)

		c0 = self.cLouden[oblDist]['c0'][0]
		c1 = self.cLouden[oblDist]['c1'][0]
		c2 = self.cLouden[oblDist]['c2'][0]

		tau = (teffs - 6250)/300
		y = c0 + c1*tau + c2*tau**2
		labs, hands = [], []
		for ii, inc in enumerate(inclinations):

			lab = r'$i_\star=' + str(inc) + '^\circ$'
			labs.append(lab)
			hands.append(('C{}'.format(ii),'-','k'))
			ax.plot(teffs,y*np.sin(np.deg2rad(inc)),color='k',lw=3.0)
			ax.plot(teffs,y*np.sin(np.deg2rad(inc)),color='C{}'.format(ii),lw=2.0)


		ax.set_xlabel(r'$T_{\rm eff} \ \rm (K)$',fontsize=font)
		ax.set_ylabel(r'$v \sin i_\star \ \rm (km/s)$',fontsize=font)
		ax.tick_params(axis='both', labelsize=font)

		ax.set_ylim(0,ymax)
		ax.set_xlim(5700,6700)

		if (vsini != None) & (Teff != None):
			ax.errorbar(Teff,vsini,yerr=svsini,xerr=sTeff,marker='o',mec='k',mfc='C7',ecolor='k')
		elif vsini != None:
			ax.axhline(vsini,color='C7')
		elif Teff != None:
			ax.axvline(Teff,color='C7')

		ax.legend(hands, labs,
				handler_map={tuple: AnyObjectHandler()},
				fancybox=True,shadow=True,
				fontsize=font,
				loc='upper left')


	def stellarInclination(self,ndraws=10000,nwalkers=100,nproc=1,
                        moves=None,path='./',
                        plot_corner=True,plot_vsini=True,save_df=True,
                        save_corner=True,save_vsini=True,
						save_convergence=False,plot_convergence=True):
		'''Stellar inclination (properly)
		
		Here the stellar inclination is calculated following :cite:t:`Masuda2020`, where we perform a Monte Carlo Markov Chain (MCMC) sampling of the posterior for :math:`i_\star`.
		:py:class:`emcee` :cite:p:`emcee` is used for the sampling.
		

		:param ndraws: Number of draws for the MCMC. Optional, default 10000.
		:type ndraws: int

		:param nwalkers: Number of walkers for MCMC. Optional, default 10000.
		:type nwalkers: int

		:param nproc: Number of CPUs for multiprocessing. Optional, default 1.
		:type nproc: int

		:param moves: :py:class:`emcee.moves` object. Optional, default ``None``.
		:type moves: int

		:param path: Path to store results. Optional, default './'.
		:type path: str

		:param plot_corner: Whether to create a :py:class:`corner` :cite:p:`corner` plot. Optional, default ``True``.
		:type plot_corner: bool

		:param plot_vsini: Whether to plot the distribution of the resulting :math:`v` against the input value for :math:`v \sin i_\star`. Optional, default ``True``.
		:type plot_vsini: bool

		:param save_df: Whether to save the results in a .csv file. Optional, default ``True``.
		:type save_df: bool

		:param save_corner: Whether to save the corner plot. Optional, default ``True``.
		:type save_corner: bool
		
		:param save_vsini: Whether to save the vsini plot. Optional, default ``True``.
		:type save_vsini: bool

		:param save_convergence: Whether to save the convergence plot. Optional, default ``False``.
		:type save_convergence: bool

		:param plot_covergence: Whether to plot the autocorrelation for the MCMC. Optional, default ``True``.
		:type plot_covergence: bool
		



		'''

		try:
			self.dist
		except AttributeError:
			print('Distributions not initilialized.\nCalling iStar.createDistributions.')
			self.createDistributions()

		def start_coords(nwalkers,ndim):
			fps = self.stepParameters
			pos = []
			pars = vars(self)
			for ii in range(nwalkers):
				start = np.ndarray(ndim)
				for idx, par in enumerate(fps):
					pri = pars[par][:4]
					dist = pars[par][-1]
					assert dist in ['gauss','tgauss','uniform','jeff'], print('{} is not a valid option for the starting distribution.'.format(dist))
					if dist == 'tgauss':
						start[idx] = tgauss_prior_dis(pri[0],pri[1],pri[2],pri[3])
					elif dist == 'gauss':
						start[idx] = gauss_prior_dis(pri[0],pri[1])
					elif dist == 'uniform':
						start[idx] = flat_prior_dis(np.random.uniform(),pri[2],pri[3])
					elif dist == 'jeff':
						start[idx] = jeff_prior_dis(np.random.uniform(),pri[2],pri[3])
				pos.append(start)
			return pos




		ndim = len(self.stepParameters)
		## Starting coordinates for walkers
		coords = start_coords(nwalkers,ndim)
		pp = vars(self)
		pars = { 'FPs' : ['Rs','Prot','cosi'],
				'Rs' : pp['Rs'],
				'Prot' : pp['Prot'],
				'cosi' : pp['cosi'],
				'vsini' : pp['vsini'],
            }
		
		with Pool(nproc) as pool:
			sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,
					pool=pool,moves=moves,kwargs=pars)

			print('Number of draws is {}.\n'.format(ndraws))
						
			## We'll track how the average autocorrelation time estimate changes
			index = 0
			autocorr = np.empty(ndraws)
			## This will be useful to testing convergence
			old_tau = np.inf

			for sample in sampler.sample(coords,iterations=ndraws,progress=True):
				## Only check convergence every 100 steps
				if sampler.iteration % 100: continue

				## Compute the autocorrelation time so far
				## Using tol=0 means that we'll always get an estimate even
				## if it isn't trustworthy
				tau = sampler.get_autocorr_time(tol=0)
				autocorr[index] = np.mean(tau)
				index += 1
				## Check convergence
				converged = np.all(tau * 50 < sampler.iteration)
				converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
				if converged: 
					print('\nMCMC converged after {} iterations.'.format(sampler.iteration))
					break
				
				old_tau = tau

		
		## Plot autocorrelation time
		## Converged if: 
		## - chain is longer than 100 times the estimated autocorrelation time 
		## - & this estimate changed by less than 1%.
		if plot_convergence:
			figc = plt.figure(figsize=self.figsize)
			axc = figc.add_subplot(111)
			nn, yy = 100*np.arange(1,index+1), autocorr[:index]
			axc.plot(nn,nn/50.,'k--')
			axc.plot(nn,yy,'k-',lw=3.0)
			axc.plot(nn,yy,'-',color='C0',lw=2.0)
			axc.set_xlabel(r'$\rm Step \ number$')
			axc.set_ylabel(r'$\rm \mu(\hat{\tau})$')
			if save_convergence:
				plt.savefig(path+'/autocorr.pdf')


		tau = sampler.get_autocorr_time(tol=0)
		n_auto = 2
		thin = int(0.5 * np.min(tau))
		burnin = int(n_auto * np.max(tau))
		samples = sampler.get_chain(discard=burnin)	
		flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)	
		flat_samples[:,2] = abs(flat_samples[:,2])

		self.dist['cosi'] = flat_samples[:,2]
		incs = np.rad2deg(np.arccos(flat_samples[:,2]))
		self.dist['incs'] = incs
		flat_samples = np.concatenate(
							(flat_samples, incs[:,None]), axis=1)

		nom = 2*np.pi*flat_samples[:,0]*rsunfac
		den = flat_samples[:,1]*dayfac
		v = nom/den
		flat_samples = np.concatenate(
							(flat_samples, v[:,None]), axis=1)

		flat_samples = np.concatenate(
							(flat_samples, v[:,None]*np.sin(np.deg2rad(incs[:,None]))), axis=1)

		log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

		results = {}
		results['Parameter'] = ['Median','Lower','Upper']
		medians = []
		pps = pars['FPs'] + ['incs','v','vsini']

		for ii, fp in enumerate(pps):
			val,upper,lower = self.getConfidence(flat_samples[:,ii])
			results[fp] = [val,lower,upper]
			medians.append(val)
		labs = [self.labels[par] for par in pps]
		labs.append(r'$\ln \mathcal{L}$')
		res_df = pd.DataFrame(results)
		if save_df:
			res_df.to_csv(path+'results.csv')
		if plot_corner:
			import corner
			all_samples = np.concatenate(
				(flat_samples, log_prob_samples[:, None]), axis=1)
			medians.append(np.amax(log_prob_samples[:, None]))
			corner.corner(all_samples, labels=labs, truths=None)

			if save_corner:
				plt.savefig(path+'corner.pdf')

		if plot_vsini:
			fig = plt.figure(figsize=self.figsize)
			ax = fig.add_subplot(111)
			ax.hist(v,color='C0',
					bins=50, density=True, histtype='step', 
					stacked=True, fill=False,
					label=r'$v, \ \rm calculated$')
			veq = 2*np.pi*self.Rs[0]*rsunfac/(self.Prot[0]*dayfac)
			ax.axvline(veq,ls='-',lw=1.0,label=r'$v_{\rm eq}=2 \pi R_\star/P_{\rm rot}$',color='C0')
			ax.hist(v*np.sin(np.deg2rad(incs)),color='C1',
					bins=50, density=True, histtype='step', 
					stacked=True, fill=False,
					label=r'$v \sin i_\star, \ \rm calculated$')
			ax.axvline(self.vsini[0],color='C1',ls='-',lw=1.0,label=r'$v \sin i_\star, \ \rm input$')
			ax.axvline(self.vsini[0]-self.vsini[1],color='C1',ls='--',lw=1.0)
			ax.axvline(self.vsini[0]+self.vsini[1],color='C1',ls='--',lw=1.0)
			ax.set_xlabel(r'$v \sin i_\star \ \rm (km/s)$')
			ax.set_ylabel(r'$\rm PDF$')
			ax.legend()
			if save_vsini:
				plt.savefig(path+'vsini.pdf')

		return res_df



	

	def createDistributions(self,N=2000):
		'''Create distributions for parameters

		Function that creates distributions for the parameters given the values in the tuples of `iStar()`. If a distribution (from an MCMC for instance) is set in place of the tuple, this (these) distribution will be used instead. The length of these distributions will be used for those, where we do create a distribution.

		:param N: Number of draws for distributions.
		:type N: int

		'''

		self.dist = {}
		pars = vars(self)
		generateDist = []

		for par in pars:
			if par in self.parameters:
				if type(pars[par]) == tuple:
					generateDist.append(par)
				elif type(pars[par]) == np.ndarray:
					self.dist[par] = pars[par]
					N = len(pars[par])
				else:
					print('{} should be either tuple (see __init__) or numpy.ndarray'.format(par))

		for par in generateDist:
			if pars[par][-1] == 'gauss':
				self.dist[par] = np.random.normal(pars[par][0],pars[par][1],N)
			elif pars[par][-1] == 'uniform':
				self.dist[par] = np.random.uniform(pars[par][2],pars[par][3],N)


	def getKDE(self,z,**kwargs):
		'''KDE for distribution

		Calculates the kernel density estimation (KDE) using :py:class:`statsmodel.nonparametric.KDEUnivariate()`.


		:param z: Distribution for which to calculate KDE.
		:type z: array

		:returns: KDE support, KDE density
		:rtype: array, array

		'''
		kde = sm.nonparametric.KDEUnivariate(z)
		kde.fit(**kwargs)  # Estimate the densities
		x, y = kde.support, kde.density
		return x, y


		
	def hpd(self,data,lev=0.68) :
		''' The Highest Posterior Density.
		
		The Highest Posterior Density (credible) interval of data at level lev.


		:param data: Sequence of real values.
		:type data: array

		:param lev: Confidence level (0 < lev < 1), optional. The default is 0.68.
		:type lev: float

		:returns: ()
		:rtype: tuple	
		

		'''
		
		d = list(data)
		d.sort()

		nData = len(data)
		nIn = int(round(lev * nData))
		if nIn < 2 :
			raise RuntimeError("not enough data")
		
		i = 0
		r = d[i+nIn-1] - d[i]
		for k in range(len(d) - (nIn - 1)) :
			rk = d[k+nIn-1] - d[k]
			if rk < r :
				r = rk
				i = k

		assert 0 <= i <= i+nIn-1 < len(d)
		
		return (d[i], d[i+nIn-1], i, i+nIn-1)

	def getConfidence(self,z,lev=0.68):
		'''Calculate confidence level.
		

		:param z: Sequence of real values.
		:type z: array

		:param lev: Confidence level (0 < lev < 1), optional. The default is 0.68.
		:type lev: float

		:returns: Median value, upper uncertainty, lower uncertainty.
		:rtype: float, float, float



		'''
		val = np.median(z)
		conf = self.hpd(z,lev=lev)
		up = conf[1]-val
		low = val-conf[0]

		return val, up, low


	def diagnostics(self,z,par='Parameter',ax=None,lev=0.68):
		'''Diagnostics and KDE plot

		This function calculates and prints the median and confidence intervals (from :py:class:`hpd`). It also creates a plot of the KDE distribution with the confidence levels highlighted.


		Provide either the name of the parameter from :py:class:`parameters` or give a distribution directly. In the latter case the desired label can be provdided in `par`.

		:param z: The name of the parameter or sequence of real values/distribution.
		:type z: str, array

		:param par: Parameter to plot and calculate confidence levels for. Optional, default 'Parameter'.
		:type par: str

		:param ax: Axis in which to plot the KDE. Optional, default ``None``, figure and axis will be created.
		:type ax: :py:class:`matplotlib.axes._subplots.AxesSubplot`


		:param lev: Confidence level (0 < lev < 1), optional. The default is 0.68.
		:type lev: float
		


		'''
		if type(z) == str:
			par = z
			z = self.dist[z]

		val, up, low = self.getConfidence(z,lev=lev)
		print('Median and confidence level ({} credibility):'.format(lev))
		print('{}={:0.3f}+{:0.3f}-{:0.3f}'.format(par,val,up,low))
		xkde, ykde = self.getKDE(z)
		if ax == None:
			fig = plt.figure(figsize=self.figsize)
			ax = fig.add_subplot(111)
		ax.plot(xkde,ykde,color='k')


		vals = (xkde > (val-low)) & (xkde < (up+val))
		xs = xkde[vals]
		ys = ykde[vals]
		ax.fill_between(xs,ys,color='C0',alpha=0.5, label=r"$\rm HPD$")


		ax.set_ylim(ymin=0.0)
		ax.set_xlim(min(xkde),max(xkde))

		ax.set_ylabel(r'$\rm KDE$')
		try:
			ax.set_xlabel(self.labels[par])
		except KeyError:
			ax.set_xlabel(par)

		idx = np.argmin(abs(xkde-val))
		plt.vlines(val,ymin=0,ymax=ykde[idx],linestyle='-',color='k')
		idx_up = np.argmin(abs(xkde-(up+val)))
		plt.vlines(up+val,ymin=0,ymax=ykde[idx_up],linestyle='--',color='k')
		idx_low = np.argmin(abs(xkde-(val-low)))
		plt.vlines(val-low,ymin=0,ymax=ykde[idx_low],linestyle='--',color='k')

	def savePosterior(self,post='cosp',path='./',name='name',format='.csv',
				xcol=None,ycol=None):
		'''Save posteriors

		Save the posteriors for the stellar inclination, obliquity, and projected obliquity in a .csv file.


		:param post: The posterior to save. Optional, default 'cosp'. Options are 'cosp', 'incs', 'lam'.
		:type post: str

		:param path: Path to save the file. Optional, default './'.
		:type path: str

		:param name: Name of the file. Optional, default 'name'.
		:type name: str

		:param format: Format of the file. Optional, default '.csv'. Options are '.csv', '.npy'.
		:type format: str

		'''
		assert post in ['cosp','incs','lam'], print('{} is not a valid option.'.format(post))
		assert format in ['.csv','.npy'], print('{} is not a valid format.'.format(format))
		
		xkde, ykde = self.getKDE(self.dist[post])
		if xcol == None:
			xcol = post
		if ycol == None:
			ycol = 'KDE'
		if format == '.csv':
			df = pd.DataFrame({xcol : xkde, ycol : ykde})
			df.to_csv(path+name+format,index=False)
		elif format == '.npy':
			np.save(path+name+format,[xkde,ykde])


def lnprob(positions,**pars):
	'''Likelihood 

	The likelihood function for :py:class:`iStar.stellarInclination()`. Defined similar to the one in :cite:t:`Hjorth2021`:

	.. math::
		\mathcal{L} = \mathcal{N}(x_{R_\star};\mu_{R_\star},\sigma_{R_\star}) + \mathcal{N}(x_{P_{\\rm rot}};\mu_{P_{\\rm rot}},\sigma_{P_{\\rm rot}}) + \mathcal{N}(v u;\mu_{v \sin i_\star},\sigma_{v \sin i_\star}) \, ,

	with :math:`u=\sqrt{ 1 - \mathcal{U}(x_{\cos i_\star};a=-1,b=1) }` and where :math:`x_i` is the drawn value for parameter :math:`i= R_\star,P_{\\rm rot},\cos i_\star`. :math:`\mathcal{L}` and :math:`\mathcal{U}` denote a gaussian respectively uniform prior. Other options are also available see :py:class:`coPsi.priors`.

	'''
	log_prob = 0.0
	fps = pars['FPs']
	for idx, par in enumerate(fps):
		val = positions[idx]
		pri = pars[par][:4]
		ptype = pars[par][-1]
		pval, sigma, lower, upper = pri[0], pri[1], pri[2], pri[3]
		if ptype == 'uniform':
			prob = flat_prior(val,lower,upper)
		elif ptype == 'jeff':
			prob = jeff_prior(val,lower,upper)
		elif ptype == 'gauss':
			prob = gauss_prior(val,pval,sigma)
		elif ptype == 'tgauss':
			prob = tgauss_prior(val,pval,sigma,lower,upper)
		if prob != 0.:
			log_prob += np.log(prob)
		else:
			return -np.inf

	nom = 2*np.pi*positions[0]*rsunfac
	den = positions[1]*dayfac
	v = nom/den
	vsini = v*np.sqrt(1-positions[2]**2)

	prob = gauss_prior(vsini,pars['vsini'][0],pars['vsini'][1])
	if prob != 0.:
		return np.log(prob) + log_prob
	else:
		return -np.inf

if __name__ == '__main__':
	N = 2000
	#incs = createDistribution(N,85.3,0.02)
	inco = createDistribution(N,85.3,0.02)#*deg2rad
	
	r = 2.69*u.R_sun
	r = r.to_value('km')
	sr = 0.04*u.R_sun
	sr = sr.to_value('km')
	
	rs = createDistribution(N,r,sr)
	vs = createDistribution(N,1.99,0.07)
	p = 3.36316383*u.d
	p = p.to_value('s')
	sp = 0.32774516*u.d
	sp = sp.to_value('s')

	Prot = createDistribution(N,p,sp)
	
	si = stellarInclination(Prot,rs,vs)#*deg2rad
	lam = createDistribution(N,-77.86,2.3)#*deg2rad
	
	
	psi, _ = coPsi(inco,si,lam)
	
	upsi, _ = coPsi(180-inco,si,lam)
	
	
	v, u, l = getConfidence(psi)
	
	
	xk, yk = getKDE(psi)
	xk2, yk2 = getKDE(upsi)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xk,yk)
	ax.plot(xk2,yk2)
	uv, uu, ul = getConfidence(upsi)
	
	#r = Rs#.to_value('km')*u.km
	#p = Prot.to_value('s')*u.s
	#vs = vsini[ii]*u.km/u.s

