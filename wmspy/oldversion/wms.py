import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy.signal import find_peaks
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

import simulation, filters
from utilities import plotData, Newton_Raphson
from datatypes import Measurement, Background

def assignMethodsToDatatype(dtype):
    """
    assignMethodsToDatatype(dtype)
    Assigns functions from wms to methods of dtype, including:

    Also creates attributes in dtype, including:
        filters
    """
    dtype.filters = []
    dtype.t_decimated = []
    dtype.RthetaXY = []
    dtype.ratios = []
    equations = [assignFilters,
                 applyLockIns,
                 calcnf1fRatio,
                 calcnf1fRatios,
                 plotFilterResults,
                 plotRatios,
                 plotFilters,
                 updateSims,
                 nearestIndex_decimated,
                 ]
    for equation in equations:
        setattr(dtype, equation.__name__, equation)

def assignFilters(self, *filterlists):
    """
    wms.assignFilters(*filterlists)
    *filterlists = variable number of filter lists each of the form [1f, 2f, 3f, ...]
    assigns filters to attribute self.filters as list of lists
    """
    self.filters = list(filterlists)

def applyLockIns(self, plotflag=False, plotflag_FFT=False):
    """
    wms.applyLockIns()
    plotflag = bool
    plotflag_FFT = bool
    Applies all lock-in filters in self.filters to the data of self (self.t, self.I)
    Stores results to self.t_decimated and self.RthetaXY, each having the same shape as self.filters
    """
    self.t_decimated = []
    self.RthetaXY = []
    for filterlist in self.filters:
        t_ = []
        R_ = []
        for filt in filterlist:
            t_decimated, RthetaXY = filt.applyLockIn(self.t, self.I, plotflag_FFT=plotflag_FFT)
            t_.append(t_decimated)
            R_.append(RthetaXY)
        self.t_decimated.append(t_)
        self.RthetaXY.append(R_)
    if plotflag:
        self.plotFilterResults()

def plotFilters(self, fmax=None, npoints=10000, linestyle='-', dB=False, fig=None):
    """
    wms.plotFilters
    fmin = min frequency (Hz)
    fmax = max frequency (Hz)
    npoints = number of points, default 10E3
    fig = figure handle to display on
    """
    if fig is not None:
        ax = fig.get_axes()[0]
        xlims_previous = ax.get_xlim()
        xscale_previous = ax.get_xscale()
    for filterlist in self.filters:
        for filt in filterlist:
            filt.plot(fmax=fmax, npoints=npoints, linestyle=linestyle, dB=dB, fig=fig)
    if fig is not None:
        ax.set_xlim(xlims_previous)
        ax.set_xscale(xscale_previous)

def calcnf1fRatio(self, R1, Xn, Yn, R1_bkg=1, Xn_bkg=0, Yn_bkg=0):
    """
    wms.calcnf1fRatio(R1, Xn, Yn, R1_bkg=1, Xn_bkg=0, Yn_bkg=0)
    Calculate the nf/1f ratio with background subtraction
    R1 = First harmonic lock-in amplitude array
    Xn = n-th harmonic lock-in X-amplitude array
    Yn = n-th harmonic lock-in Y-amplitude array
    R1_bkg = First harmonic lock-in amplitude array of background
    Xn_bkg = n-th harmonic lock-in X-amplitude array of background
    Yn_bkg = n-th harmonic lock-in Y-amplitude array of background
    Returns nf/1f ratio array
    """
    return np.sqrt( ((Xn/R1) - (Xn_bkg/R1_bkg))**2 +
                    ((Yn/R1) - (Yn_bkg/R1_bkg))**2 )

def calcnf1fRatios(self, bkg=None, replaceNaNs=0.0, applylockins=True, plotflag=False):
    """
    wms.calcnf1fRatios(bkg=None, replaceNaNs=0.0)
    Calculate all nf/1f ratios from results of each filter list
    bkg = filtered RthetaXY results from background measurement having same shape as self.RthetaXY
    replaceNaNs = what to replace NaN filter results with, default 0.0, if None then no replacement
    plotflag = bool, plots filter results and ratios
    Stores results to self.ratios
    """
    if applylockins:
        self.applyLockIns()
    # If background is None, set bkg values for no background subtraction
    if bkg is None:
        R1_bkg, _, _,      _      = 1, 0, 0, 0
        _,      _, Xn_bkg, Yn_bkg = 1, 0, 0, 0
    # Initialize the storage and loop through each list of filtered results
    self.ratios = []
    for i, RthetaXYlist in enumerate(self.RthetaXY):
        ratios_ = []
        # Grab the 1f parameters and background values if necessary
        R1, theta1, X1, Y1 = RthetaXYlist[0]
        if bkg is not None:
            R1_bkg, _, _, _ = bkg[i][0]
        # Loop through the higher order harmonics
        for j, (Rn, thetan, Xn, Yn) in enumerate(RthetaXYlist[1:]):
            if bkg is not None:
                _, _, Xn_bkg, Yn_bkg = bkg[i][j+1]
            # Calculate the nf/1f ratio with background subtraction
            r_ = self.calcnf1fRatio(R1, Xn, Yn,
                                    R1_bkg=R1_bkg, Xn_bkg=Xn_bkg, Yn_bkg=Yn_bkg)
            if replaceNaNs is not None:
                r_[np.where(np.isnan(r_))] = replaceNaNs
            ratios_.append(r_)
        self.ratios.append(ratios_)
    if plotflag:
        self.plotFilterResults(bkg=bkg)
        self.plotRatios()

def plotFilterResults(self, ind=(0,None), bkg=None, figs=None):
    """
    wms.plotFilterResults(bkg=None, figs=None)
    Plots each set of filter's results in a separate figure (R1f, R2f, R3f, ...)
    ind = indices of range to plot
    bkg = filtered background results RthetaXY
    figs = figure handles to overlay plots on
    Returns list of figures
    """
    i1, i2 = ind
    createfigslist = False
    if figs is None:
        figs = []
        createfigslist = True
    for i, (RthetaXYlist, t_declist) in enumerate(zip(self.RthetaXY, self.t_decimated)):
        xy, labels, styles = [], [], []
        for j, ((R, theta, X, Y), t_dec) in enumerate(zip(RthetaXYlist, t_declist)):
            xy.append((t_dec[i1:i2], R[i1:i2]))
            labels.append('{}f'.format(j+1))
            styles.append('')
            if bkg is not None:
                xy.append((t_dec[i1:i2], bkg[i][j][0][i1:i2]))
                labels.append('{}f_bkg'.format(j+1))
                styles.append('--')
        fig, _ = plotData(xy,
                 labels = labels,
                 linestyles = styles,
                 xlabel = 'Time (s)',
                 ylabel = 'Lock-In Amplitude',
                 title = 'wms.plotFilterResults: Filterlist {}'.format(i),
                 fig = None if createfigslist else figs[i])
        if createfigslist:
            figs.append(fig)
    return figs

def plotRatios(self, ind=(0,None), figs=None):
    """
    wms.plotRatios()
    Plot each set of filter's nf/1f ratios in a separate figure (2f/1f, 3f/1f, 4f/1f, ...)
    ind = indices of range to plot
    figs = figure handles to overlay plots on
    Returns list of figures
    """
    i1, i2 = ind
    createfigslist = False
    if figs is None:
        figs = []
        createfigslist = True
    for i, (ratioslist, t_declist) in enumerate(zip(self.ratios, self.t_decimated)):
        xy, labels, styles = [], [], []
        for j, (ratio, t_dec) in enumerate(zip(ratioslist, t_declist)):
            xy.append((t_dec[i1:i2], ratio[i1:i2]))
            labels.append('{}f/1f'.format(j+2))
            styles.append('')
        fig, _ = plotData(xy,
                 labels = labels,
                 linestyles = styles,
                 xlabel = 'Time (s)',
                 ylabel = 'nf/1f Ratio',
                 title = 'wms.plotRatios: Filterlist {}'.format(i),
                 fig = None if createfigslist else figs[i])
        if createfigslist:
            figs.append(fig)
    return figs

def updateSims(self, sims=None, T=None, P=None, X=None, L=None, vshift=None, FSR=None, i_sim=None):
    """
    wms.updateSims(sims=None, T=None, P=None, X=None, vshift=None)
    Updates any of given values (sims, T, P, X) and recalculates simulated intensity
    sims = simulation.Simulation or list of Simulations
    T = temperature (K)
    P = pressure (atm)
    X = mole fraction of absorbing species or list of broadening species beginning with self
    vshift = absolute wavelength (cm-1) float or array-like of same size as self.sims
    FSR = free spectral range (cm-1) float or array-like of same saize as self.sims
    i_sim = index of sim in self.sims to update vshift & FSR if single-valued
    Returns newly calculated self.I
    """
    # Update simulations & parameters
    if sims is not None:
        if isinstance(sims, list):
            self.sims = sims
        else:
            self.sims = [sims]
    if T is not None:
        for sim in self.sims:
            sim.T = T
    if P is not None:
        for sim in self.sims:
            sim.P = P
    if X is not None:
        for sim in self.sims:
            if isinstance(X, list) or (not isinstance(sim.X, list)):
                sim.X = X
            else:
                sim.X[0] = X
    if L is not None:
        for sim in self.sims:
            sim.L = L
    if vshift is not None:
        if isinstance(vshift, (list, tuple, np.ndarray)):
            for sim, vshift_ in zip(self.sims, vshift):
                sim.vshift = vshift_
        else:
            if i_sim is None:
                for sim in self.sims:
                    sim.vshift = vshift
            else:
                self.sims[i_sim].vshift = vshift
    if FSR is not None:
        if isinstance(FSR, (list, tuple, np.ndarray)):
            for sim, FSR_ in zip(self.sims, FSR):
                sim.FSR = FSR_
        else:
            if i_sim is None:
                for sim in self.sims:
                    sim.FSR = FSR
            else:
                self.sims[i_sim].FSR = FSR
    # Calculate simulated I
    self.I = 0
    for sim in self.sims:
        self.I += sim.calcI()
    return self.I

def nearestIndex_decimated(self, t, i_filterlist=0, i_harmonic=0):
    """
    wms.nearestIndex_decimated(t, i_filterlist=0, i_harmonic=0):
    Finds nearest index or indices to the given time(s) in t_decimated[i_filterlist][i_harmonic]
    t = float or array of times
    Returns int index or int array of indices
    """
    def nearest_i(t):
        return np.argmin(np.abs(self.t_decimated[i_filterlist][i_harmonic] - t))
    if isinstance(t, np.ndarray) or isinstance(t, list):
        i = np.array([nearest_i(t_) for t_ in t])
    else:
        i = nearest_i(t)
    return i

assignMethodsToDatatype(Measurement)
assignMethodsToDatatype(Background)

def find_ipeak2f(measObject, bkg=None, ind=(0,None),
                 i_filterlist=0, i_1f=0, i_2f=1, i_R=0,
                 subtractbkg=False, singleoutput=True, plotflag=False):
    """
    wms.find_ipeak2f(measObject, bkg=bkgObject,
                     ind=(0,None), i_filterlist=0, i_1f=0, i_2f=1, i_R=0,
                     subtractbkg=False, singleoutput=True, plotflag=False)
    measObject = datatypes.Measurement object (measurement or simulation)
    bkg = datatypes.Background object
    ind = (i_min, i_max) indices of range to find peaks within
    i_filterlist = index of filter-list in measObject.filters list to use for finding peak2f's
    i_1f = index of 1f filter in measObject.filters[i_filterlist]
    i_2f = index of 2f filter in measObject.filters[i_filterlist]
    i_R  = index of R in measObject.RthetaXY[i_filterlist][i_*f]
    subtractbkg = bool, whether to subtract bkg signal before peak finding
    singleoutput = bool, whether to only output a single peak
    plotflag = bool
    Returns i_global, i_local
    """
    i_min, i_max = ind
    changes1f = False
    # If bkg given, find where the 1f measurement crosses the 1f background in the ind range
    if bkg is not None:
        delta1f = measObject.RthetaXY[i_filterlist][i_1f][i_R][i_min:i_max] - \
                         bkg.RthetaXY[i_filterlist][i_1f][i_R][i_min:i_max]
        i_1fchanges = np.where(np.sign(delta1f[:-1]) != np.sign(delta1f[1:]))[0] + 1 #local indices
        # If any bkg crosses found, use those instead of maximum 2f peak
        if len(i_1fchanges) != 0:
            changes1f = True
    # Find all the 2f peaks of the measurement with optional bkg subtraction
    signal2f = measObject.RthetaXY[i_filterlist][i_2f][i_R][i_min:i_max] - \
                     (bkg.RthetaXY[i_filterlist][i_2f][i_R][i_min:i_max] if ((bkg is not None) and subtractbkg) else 0)
    i_2fpeaks_all, _ = find_peaks(signal2f) #local indices
    # If no peaks found in measurement, error out
    if len(i_2fpeaks_all) == 0:
        raise ValueError('No 2f signal peaks found in range of ind in wms.find_ipeak2f')
    # Default to using all peaks if necessary
    if not changes1f:
        i_2fpeaks = i_2fpeaks_all
        # If single output desired, find the maximum 2f peak
        if singleoutput:
            i_2fpeak  = i_2fpeaks[np.argmax(signal2f[i_2fpeaks])]
    # Otherwise find peaks nearest to meas-bkg crosses
    else:
        # Find the nearest 2f peak to each meas-bkg 1f crossing
        i_2fpeaks = i_2fpeaks_all[[np.argmin(np.abs(i_2fpeaks_all - i_1fchange)) for i_1fchange in i_1fchanges]] #local indices
        # If single output desired, find the maximum 2f peak
        if singleoutput:
            i_2fpeak = i_2fpeaks[np.argmax(signal2f[i_2fpeaks])]
    # Plot the results
    if plotflag:
        fig = plt.figure()
        ax1, ax2, ax3 = fig.add_subplot(311), fig.add_subplot(312), fig.add_subplot(313)
        # 1f Signals
        ax1.plot(measObject.t_decimated[i_filterlist][i_1f][i_min:i_max],
                 measObject.RthetaXY[i_filterlist][i_1f][i_R][i_min:i_max],
                 '-', label='measurement')
        if bkg is not None:
            ax1.plot(bkg.t_decimated[i_filterlist][i_1f][i_min:i_max],
                     bkg.RthetaXY[i_filterlist][i_1f][i_R][i_min:i_max],
                     '--', label='background')
        if changes1f:
            ax1.plot(measObject.t_decimated[i_filterlist][i_1f][i_min:i_max][i_1fchanges],
                     measObject.RthetaXY[i_filterlist][i_1f][i_R][i_min:i_max][i_1fchanges],
                     'o', label='delta_1f sign changes')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('1f Signals')
        ax1.legend(loc='best')
        # Delta1f
        if bkg is not None:
            ax2.plot(measObject.t_decimated[i_filterlist][i_1f][i_min:i_max],
                     delta1f,
                     '-', label='measurement - background')
        if changes1f:
            ax2.plot(measObject.t_decimated[i_filterlist][i_1f][i_min:i_max][i_1fchanges],
                     delta1f[i_1fchanges],
                     'o', label='delta_1f sign changes')
        ax2.plot(measObject.t_decimated[i_filterlist][i_1f][[i_min,i_max]],
                 [0,0],
                 'k--')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Delta_1f Signal')
        ax2.legend(loc='best')
        # 2f Signal
        ax3.plot(measObject.t_decimated[i_filterlist][i_1f][i_min:i_max],
                 signal2f,
                 '-', label='measurement')
        ax3.plot(measObject.t_decimated[i_filterlist][i_1f][i_min:i_max][i_2fpeak if singleoutput else i_2fpeaks],
                 signal2f[i_2fpeak if singleoutput else i_2fpeaks],
                 'o', label='2f peaks')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('2f Signal')
        ax3.legend(loc='best')
    return (i_2fpeak+i_min, i_2fpeak) if singleoutput else (i_2fpeaks+i_min, i_2fpeaks)

def fit_Xvshift(simObject, measObject, bkg=None, ind=(0,None),
                fit_X=True, X0=None, X_bounds=(0.0,1.0),
                fit_vshift=True, vshift0=None, dv_bound=None,
                fit_FSR=False, FSR0=None, FSR_perc_bound=5,
                fit_T=False, T0=None, T_bounds=(300, 3000),
                i_filterlist=0, i_harmonic=0, i_sim=0,
                i_spectra=0, i_line=0,
                calc_meas_ratios=True, subtractbkg=True, plotflag=False, printupdates=False,
                **kwargs):
    """
    wms.fit_Xvshift(simObject, measObject, bkg=None, ind=(0,None),
                    fit_X=True, X0=None, X_bounds=(0.0,1.0),
                    fit_vshift=True, vshift0=None, dv_bound=None,
                    fit_FSR=False, FSR0=None, FSR_perc_bound=5,
                    fit_T=False, T0=None, T_bounds=(300, 3000),
                    i_filterlist=0, i_harmonic=0, i_sim=0,
                    i_spectra=0, i_line=0,
                    rtol=0.01, atol=0, maxiter=1000,
                    calc_meas_ratios=True, subtractbkg=True, plotflag=False, printupdates=False)
    Fits any or all of X, vshift, FSR, T in simObject selected by boolean flag inputs fit_X, fit_vshift, fit_FSR, fit_T
    simObject  = datatypes.Measurement object of simulation
    measObject = datatypes.Measurement object of measurement
    bkg = datatypes.Background object
    ind = (i_min, i_max) indices of range to find peaks within
    fit_X = bool, whether to fit the mole fraction, default True
    X0 = initial mole fraction estimate, default simObject.sims[i_sim].X
    X_bounds = (min, max) mole fraction bounds, default (0,1)
    fit_vshift = bool, whether to fit the wavenumber shift, default True
    vshift0 = initial wavenumber shift estimate, default is to estimate it from:
        simObject.sims[i_sim].spectras[i_spectra].lines[i_line].calcAbsorbanceParameters(T,P,X,L)[0]
        i_spectra default 0
        i_line default 0
    dv_bound = +/- amplitude of bounds from vshift0 for fitting vshift, default is to estimate it from:
        simObject.sims[i_sim].vrel
    fit_FSR = bool, whether to fit the FSR
    FSR0 = initial FSR estimate, default is simObject.sims[i_sim].FSR
    FSR_perc_bound = +/- percentage bound on FSR fitting, default 5 percent (input as 5 not 0.05)
    fit_T = bool, whether to fit the Temperature
    T0 = initial Temperature estimate (K), default is simObject.sims[i_sim].T
    T_bounds = (min, max) temperature bounds, default (300, 3000) K
    i_filterlist = index of filter-list in measObject and simObject .ratios to fit
    i_harmonic   = index of harmonic ratio in measObject and simObject .ratios[i_filterlist] to fit
    i_sim = index of simObject.sims to fit .vshift and .FSR
    i_spectra = index of spectra in
    i_line =
    calc_meas_ratios = bool, whether to force recalculation of measObject.ratios
    subtractbkg = bool, whether to subtract bkg signal for fitting
    plotflag = bool
    printupdates = bool
    kwargs = keyword arguments to scipy.optimize.differential_evolution solving
        (tol, atol, maxiter, strategy, popsize, mutation, recombination, polish)
    Returns X, vshift, FSR (optional). Updates simObject parameters.
    """
    if (not fit_X) and (not fit_vshift) and (not fit_FSR) and (not fit_T):
        raise RuntimeError('One of fit_X, fit_vshift, fit_FSR, fit_T inputs to wms.fit_Xvshift must be True')
    i_min, i_max = ind
    # Apply lock-ins and calculate measurement ratios if necessary
    if calc_meas_ratios or ('ratios' not in dir(measObject)):
        if (bkg is not None) and subtractbkg and ('RthetaXY' not in dir(bkg)):
            bkg.applyLockIns()
        measObject.calcnf1fRatios(bkg = bkg.RthetaXY if ((bkg is not None) and subtractbkg) else None,
                                  applylockins = False if 'RthetaXY' in dir(measObject) else True)
    # Update simulation with initial guesses
    simObject.updateSims(T = T0,
                         X = X0,
                         vshift = vshift0,
                         FSR = FSR0,
                         i_sim = i_sim)
    # Define function of array of used inputs (X, vshift, FSR, T) to update simulation,
    # recalculate the simulated transmitted intensity, lock-in signals, & ratios,
    # and return the rms error of the simulation relative to the measurement
    # TODO: Make args i_sim, meas data, etc... constant parameters, if necessarty to make pickle-able for multiprocessing pool
    def calcSimnf1fRMS(X_vshift_FSR_T, *args):
        X_vshift_FSR_T = list(X_vshift_FSR_T)
        T = X_vshift_FSR_T.pop() if fit_T else None
        FSR = X_vshift_FSR_T.pop() if fit_FSR else None
        vshift = X_vshift_FSR_T.pop() if fit_vshift else None
        X = X_vshift_FSR_T.pop() if fit_X else None
        if printupdates:
            print('Trying X = {}, vshift={}, FSR={}, T={}'.format(X if fit_X else 'Excluded',
                                                            vshift if fit_vshift else 'Excluded',
                                                            FSR if fit_FSR else 'Excluded',
                                                            T if fit_T else 'Excluded'))
        # Update to new values
        simObject.updateSims(T = T,
                             X = X,
                             vshift = vshift,
                             FSR = FSR,
                             i_sim = i_sim)
        # Recalculate wms ratios
        simObject.calcnf1fRatios(bkg = bkg.RthetaXY if ((bkg is not None) and subtractbkg) else None,
                                 applylockins = True)
        return np.sqrt(np.sum((simObject.ratios[i_filterlist][i_harmonic][i_min:i_max] - measObject.ratios[i_filterlist][i_harmonic][i_min:i_max])**2))
    # Handle guesses and bounds
    bounds = []
    if fit_X:
        bounds.append(X_bounds)
    if fit_vshift:
        if vshift0 is None:
            sim = simObject.sims[i_sim]
            spectra = sim.spectras[i_spectra] if isinstance(sim.spectras, list) else sim.spectras
            line = spectra.lines[i_line] if isinstance(spectra.lines, list) else spectra.lines
            vshift0, _, _, _ = line.calcAbsorbanceParameters(sim.T, sim.P, sim.X, sim.L)
        if dv_bound is None:
            # TODO: Find a better way to handle estimating the v bound.  What if mod > scan or no scan?
            vrel = simObject.sims[i_sim].vrel[i_min:i_max] * simObject.sims[i_sim].FSR
            dv_bound = (np.max(vrel) - np.min(vrel)) / 2
        vshift_bounds = vshift0-dv_bound, vshift0+dv_bound
        bounds.append(vshift_bounds)
    if fit_FSR:
        if FSR0 is None:
            FSR0  = simObject.sims[i_sim].FSR
        FSR_bounds = FSR0 * np.array([1-(FSR_perc_bound/100), 1+(FSR_perc_bound/100)])
        bounds.append(FSR_bounds)
    if fit_T:
        bounds.append(T_bounds)
    # Setup initialization matrix for differential_evolution
    if 'init' not in kwargs.keys():
        len_x = np.int(np.sum(np.ones(4)[[fit_X, fit_vshift, fit_FSR, fit_T]]))
        M = np.int(len_x * (kwargs['popsize'] if 'popsize' in kwargs.keys() else 15))
        points = np.zeros((M, len_x))
        def get_truncated_normal(guess, bounds):
            sd = np.abs(np.diff(bounds))/4
            dist = truncnorm( (bounds[0]-guess)/sd, (bounds[1]-guess)/sd, loc=guess, scale=sd )
            return np.sort(dist.rvs(M).clip(bounds[0], bounds[1]))
        ix = 0
        if fit_X:
            points[:,ix] = get_truncated_normal(simObject.sims[i_sim].X, X_bounds)
            ix += 1
        if fit_vshift:
            points[:,ix] = get_truncated_normal(simObject.sims[i_sim].vshift, vshift_bounds)
            ix += 1
        if fit_FSR:
            points[:,ix] = get_truncated_normal(simObject.sims[i_sim].FSR, FSR_bounds)
            ix += 1
        if fit_T:
            points[:,ix] = get_truncated_normal(simObject.sims[i_sim].T, T_bounds)
            ix += 1
        kwargs['init'] = points
    # Perform curve fitting
    result = differential_evolution(calcSimnf1fRMS, bounds, **kwargs)
    # Print / plot results
    if printupdates:
        update = np.zeros(4)
        update[:] = np.NaN
        update[[fit_X, fit_vshift, fit_FSR, fit_T]] = result.x
        print('Optimal X, vshift, FSR, T = {}'.format(update))
    if plotflag:
        plotData([(measObject.t_decimated[i_filterlist][i_harmonic][i_min:i_max],
                   measObject.ratios[i_filterlist][i_harmonic][i_min:i_max]),
                  (simObject.t_decimated[i_filterlist][i_harmonic][i_min:i_max],
                   simObject.ratios[i_filterlist][i_harmonic][i_min:i_max])],
                 labels = ['measurement', 'simulation'],
                 linestyles = ['k-', 'r-'],
                 xlabel = 'Time (s)',
                 ylabel = 'WMS nf/1f Ratio',
                 title = 'wms.fit_Xvshift')
    return result.x # X, vshift, FSR, T (values not fit are excluded)

def fit_T(simObject, measObject, bkg=None, ind=(0,None),
          T0=(1000, 2000), X0=(0.1, 0.2), vshift0=(None,None), FSR0=(None,None), fit_FSR=False,
          ratio_perc_tol=1, Ttol=1.0, X_bounds=(0,1), dv_bounds=(None,None), FSR_perc_bounds=(5,5), maxTiter=50,
          i_filterlists=(0,1), i_harmonics=(0,0), i_sims=(0,1),
          i_spectras=(0,0), i_lines=(0,0), i_1fs=(0,0), i_2fs=(1,1), i_Rs=(0,0),
          calc_meas_ratios=True, subtractbkg_peak2f=False, subtractbkg_ratios=True,
          plotflag=False, plotflag_X=False, plotflag_findpeaks=False, printupdates=False,
          **kwargs):
    """

    """
    i_min, i_max = ind
    # Apply lock-ins and calculate measurement ratios if necessary
    if calc_meas_ratios or ('ratios' not in dir(measObject)):
        if (bkg is not None) and subtractbkg_ratios and ('RthetaXY' not in dir(bkg)):
            bkg.applyLockIns()
        measObject.calcnf1fRatios(bkg = bkg.RthetaXY if ((bkg is not None) and subtractbkg_ratios) else None,
                                  applylockins = False if 'RthetaXY' in dir(measObject) else True)
    # Calculate the target index and ratio of 2f/1f's
    i_peak2f_meas_line1, _ = find_ipeak2f(measObject, bkg=bkg, ind=ind,
                                          i_filterlist=i_filterlists[0], i_1f=i_1fs[0], i_2f=i_2fs[0], i_R=i_Rs[0],
                                          subtractbkg=subtractbkg_peak2f, singleoutput=True, plotflag=plotflag_findpeaks)
    i_peak2f_meas_line2, _ = find_ipeak2f(measObject, bkg=bkg, ind=ind,
                                          i_filterlist=i_filterlists[1], i_1f=i_1fs[1], i_2f=i_2fs[1], i_R=i_Rs[1],
                                          subtractbkg=subtractbkg_peak2f, singleoutput=True, plotflag=plotflag_findpeaks)
    line1_2f1f_meas = measObject.ratios[i_filterlists[0]][i_harmonics[0]][i_peak2f_meas_line1]
    line2_2f1f_meas = measObject.ratios[i_filterlists[1]][i_harmonics[1]][i_peak2f_meas_line2]
    target_ratio = line1_2f1f_meas / line2_2f1f_meas
    # Define function of T to update simulations with new T,
    # refit first line for X and vshift (& FSR optionally) with new T,
    # then refit second line for vshift with new T and X from first line.
    # Finally find the indices of peak2f for each line and return the ratio of peak 2f/1f's.
    def calcSimRatioOf2f1fs(T, Xguess, vshiftguesses, FSRguesses):
        print('Trying T = {}'.format(T))
        # Update simObject with new T and X, vshift, FSR, guesses
        simObject.updateSims(T=T, X=Xguess, vshift=vshiftguesses[0], FSR=FSRguesses[0], i_sim=i_sims[0])
        simObject.updateSims(vshift=vshiftguesses[1], FSR=FSRguesses[1], i_sim=i_sims[1])
        # Fit first line for X & vshift (& FSR) at new T
        line1fit = fit_Xvshift(simObject, measObject, bkg=bkg, ind=ind,
                               fit_X=True, X0=Xguess, X_bounds=X_bounds,
                               fit_vshift=True, vshift0=vshiftguesses[0], dv_bound=dv_bounds[0],
                               fit_FSR=fit_FSR, FSR0=FSRguesses[0], FSR_perc_bound=FSR_perc_bounds[0],
                               fit_T=False, T0=T,
                               i_filterlist=i_filterlists[0], i_harmonic=i_harmonics[0], i_sim=i_sims[0],
                               i_spectra=i_spectras[0], i_line=i_lines[0],
                               calc_meas_ratios=False, subtractbkg=subtractbkg_ratios, plotflag=plotflag_X, printupdates=printupdates,
                               **kwargs)
        Xopt, vshift_line1 = line1fit[0:2]
        FSR_line1 = line1fit[2] if fit_FSR else None
        # Fit second line for vshift with new T & X
        line2fit = fit_Xvshift(simObject, measObject, bkg=bkg, ind=ind,
                               fit_X=False, X0=Xopt, X_bounds=X_bounds,
                               fit_vshift=True, vshift0=vshiftguesses[1], dv_bound=dv_bounds[1],
                               fit_FSR=fit_FSR, FSR0=FSRguesses[1], FSR_perc_bound=FSR_perc_bounds[1],
                               fit_T=False, T0=T,
                               i_filterlist=i_filterlists[1], i_harmonic=i_harmonics[1], i_sim=i_sims[1],
                               i_spectra=i_spectras[1], i_line=i_lines[1],
                               calc_meas_ratios=False, subtractbkg=subtractbkg_ratios, plotflag=plotflag_X, printupdates=printupdates,
                               **kwargs)
        vshift_line2 = line2fit[0]
        FSR_line2 = line2fit[1] if fit_FSR else None
        # Find indices of 2f peaks
        i_line1, _ = find_ipeak2f(simObject, bkg=bkg, ind=ind,
                                  i_filterlist=i_filterlists[0], i_1f=i_1fs[0], i_2f=i_2fs[0], i_R=i_Rs[0],
                                  subtractbkg=subtractbkg_peak2f, singleoutput=True, plotflag=plotflag_findpeaks)
        i_line2, _ = find_ipeak2f(simObject, bkg=bkg, ind=ind,
                                  i_filterlist=i_filterlists[1], i_1f=i_1fs[1], i_2f=i_2fs[1], i_R=i_Rs[1],
                                  subtractbkg=subtractbkg_peak2f, singleoutput=True, plotflag=plotflag_findpeaks)
        # Calculate ratio of 2f-1f's
        line1_2f1f_sim = simObject.ratios[i_filterlists[0]][i_harmonics[0]][i_line1]
        line2_2f1f_sim = simObject.ratios[i_filterlists[1]][i_harmonics[1]][i_line2]
        return Xopt, (i_line1, i_line2), (vshift_line1, vshift_line2), (FSR_line1, FSR_line2), (line1_2f1f_sim, line2_2f1f_sim), line1_2f1f_sim/line2_2f1f_sim
    # Setup Newton-Raphson container
    nr_dtype = [('T', float), ('X', float), ('i', int, 2), ('vshift', float, 2), ('FSR', float, 2), ('2f1f', float, 2), ('ratio', float)]
    nr_values = np.zeros(2, dtype=nr_dtype)
    # Calculate initial guesses
    for i, (T0_, X0_) in enumerate(zip(T0,X0)):
        nr_values[i] = tuple([T0_] + list(calcSimRatioOf2f1fs(T0_, X0_, vshift0, FSR0)))
    for iteration in np.arange(maxTiter):
        T_new, i_replace = Newton_Raphson(nr_values['T'], nr_values['ratio'], target_ratio)
        X_new, _         = Newton_Raphson(nr_values['X'], nr_values['T'], T_new)
        X_new = np.array([X_new]).clip(*X_bounds)[0]
        vshift_new_line1 = np.mean(nr_values['vshift'][:,0])  #Newton_Raphson(nr_values['vshift'][:,0], nr_values['T'], T_new)
        vshift_new_line2 = np.mean(nr_values['vshift'][:,1])  #Newton_Raphson(nr_values['vshift'][:,1], nr_values['T'], T_new)
        FSR_new_line1 = np.mean(nr_values['FSR'][:,0]) if fit_FSR else None
        FSR_new_line2 = np.mean(nr_values['FSR'][:,1]) if fit_FSR else None
        nr_values[i_replace] = tuple([T_new] + list(calcSimRatioOf2f1fs(T_new,
                                                                        X_new,
                                                                        (vshift_new_line1, vshift_new_line2),
                                                                        (FSR_new_line1, FSR_new_line2))))
        if np.abs((nr_values[i_replace]['ratio'] - target_ratio)/target_ratio)*100 < ratio_perc_tol:
            if printupdates:
                print('Converged with simulated ratio within {}% of measured ratio.'.format(ratio_perc_tol))
            break
        if np.abs(np.diff(nr_values['T']))[0] < Ttol:
            if printupdates:
                print('Converged with dT < {}K.'.format(Ttol))
            break
        if iteration == maxTiter - 1:
            raise RuntimeError('Maximum iterations reached without convergence in wms.fitT. Final values of (T, X, i, vshift, FSR, 2f1f, ratio-of-2f1fs) = {}'.format(nr_values))
    output = nr_values[i_replace]
    print('Optimal T, X, i, vshift, FSR, 2f1f, ratio-of-2f1fs = {}'.format(output))
    if plotflag:
        plotData([(measObject.t_decimated[i_filterlists[0]][i_harmonics[0]][i_min:i_max],
                   measObject.ratios[i_filterlists[0]][i_harmonics[0]][i_min:i_max]),
                  (simObject.t_decimated[i_filterlists[0]][i_harmonics[0]][i_min:i_max],
                   simObject.ratios[i_filterlists[0]][i_harmonics[0]][i_min:i_max])],
                 labels = ['measurement', 'simulation'],
                 linestyles = ['k-', 'r-'],
                 xlabel = 'Time (s)',
                 ylabel = 'WMS nf/1f Ratio',
                 title = 'wms.fitT: Line 1')
        plotData([(measObject.t_decimated[i_filterlists[1]][i_harmonics[1]][i_min:i_max],
                   measObject.ratios[i_filterlists[1]][i_harmonics[1]][i_min:i_max]),
                  (simObject.t_decimated[i_filterlists[1]][i_harmonics[1]][i_min:i_max],
                   simObject.ratios[i_filterlists[1]][i_harmonics[1]][i_min:i_max])],
                 labels = ['measurement', 'simulation'],
                 linestyles = ['k-', 'r-'],
                 xlabel = 'Time (s)',
                 ylabel = 'WMS nf/1f Ratio',
                 title = 'wms.fitT: Line 2')
    return output # T, X, i, vshift, FSR, 2f1f, ratio-of-2f1fs





def vScanEquation(t,fs,vs,phivs,v0):
    """  """
    return vs*np.sin(2*np.pi*fs*t + phivs) + v0

def fitvScan(etseg,ind=None):
    """  """
    fitEq = lambda t,vs,phivs,v0 : vScanEquation(t,etseg.fs,vs,phivs,v0)
    if ind is None:
        ind = [0,len(etseg.t)-1]
    i1,i2 = ind
    popt, pcov = curve_fit(fitEq, etseg.t[i1:i2], etseg.v[i1:i2], p0=((np.max(etseg.v)-np.min(etseg.v))/2,0,np.mean(etseg.v)), bounds=([0,-2*np.pi,np.min(etseg.v)],[np.max(etseg.v)-np.min(etseg.v),2*np.pi,np.max(etseg.v)]))
    return popt, pcov

def findpeak2f(data_filtered,bkg1f,ind=None):
    """  """
    (t1f, (R1f,_,_,_)), (t2f, (R2f,_,_,_)) = data_filtered
    _, (R1f_bkg,_,_,_) = bkg1f
    if ind is None:
        ind = (0,len(t2f)-1)
    R1frel = R1f[ind[0]:ind[1]] - R1f_bkg[ind[0]:ind[1]]
    R2f    = R2f[ind[0]:ind[1]]
    multiplier = 1 - (np.abs(R1frel) / np.max(np.abs(R1frel)))
    i_local = np.nanargmax(R2f * multiplier)
    while (R2f[i_local-1]>R2f[i_local]) or (R2f[i_local+1]>R2f[i_local]):
        i_local += np.nanargmax(R2f[i_local-1:i_local+2]) - 1
    return ind[0]+i_local, i_local, t2f[ind[0]:ind[1]][i_local], R2f[i_local]

def calcvshift(T,P,X,sim,vScanParams,filts,meas_filtered,bkg1f_filtered,ind=None,line=None,previous=None,maxiter=None):
    """  """
    line_default = 0
    if line is None:
        line = line_default
    ipeak2f_meas, _, tpeak2f_meas, _ = findpeak2f(meas_filtered,bkg1f_filtered,ind=ind)
    v_meas = vScanEquation(tpeak2f_meas, sim.fs, *vScanParams)
    v_line = simulation.calcAbsorbanceParameters(T, P, X, sim.L, sim.spectras, line=line)[0]
    if previous is None:
        vshift = [v_line - v_meas*sim.FSR/(2*np.pi)]
    else:
        vshift_prev, conditions_prev = previous
        v_line_prev = simulation.calcAbsorbanceParameters(*conditions_prev[:-1],line=line_default if conditions_prev[-1] is None else conditions_prev[-1])[0]
        vshift = [vshift_prev + v_line - v_line_prev]
    sim_filtered = [filt.applyLockIn(sim.t,sim.Simulate(T,P,X,vshift[-1])) for filt in filts]
    ipeak2f_sim_global, ipeak2f_sim_local, tpeak2f_sim, _ = findpeak2f(sim_filtered,bkg1f_filtered,ind=ind)
    ipeak2f_sim = [ipeak2f_sim_global]
    iteration = 0
    while ipeak2f_meas != ipeak2f_sim[-1]:
        iteration += 1
        if iteration <= 1:
            vshift_new = vshift[-1] - (v_meas - vScanEquation(tpeak2f_sim, sim.fs, *vScanParams)) * sim.FSR/(2*np.pi)
            sim_filtered = [filt.applyLockIn(sim.t,sim.Simulate(T,P,X,vshift_new)) for filt in filts]
            ipeak2f_sim_global, ipeak2f_sim_local, tpeak2f_sim, _ = findpeak2f(sim_filtered,bkg1f_filtered,ind=ind)
            ipeak2f_sim = [ipeak2f_sim[-1], ipeak2f_sim_global]
            vshift  = [vshift[-1], vshift_new]
        else:
            if vshift[0] == vshift[1]:
                raise ValueError('calcvshift found same vshift initial guesses. Cannot run Newton-Raphson.')
            elif ipeak2f_sim[0] == ipeak2f_sim[1]:
                while ipeak2f_sim[0] == ipeak2f_sim[1]:
                    vshift_new = vshift[0] + 2*(vshift[1]-vshift[0])
                    sim_filtered = [filt.applyLockIn(sim.t,sim.Simulate(T,P,X,vshift_new)) for filt in filts]
                    ipeak2f_sim_global, ipeak2f_sim_local, tpeak2f_sim, _ = findpeak2f(sim_filtered,bkg1f_filtered,ind=ind)
                    if (ipeak2f_sim[0] != ipeak2f_sim_global) and (np.abs(ipeak2f_sim_global-ipeak2f_meas) > np.abs(ipeak2f_sim[0]-ipeak2f_meas)):
                        vshift_new = vshift[0] - (vshift_new - vshift[0])
                        sim_filtered = [filt.applyLockIn(sim.t,sim.Simulate(T,P,X,vshift_new)) for filt in filts]
                        ipeak2f_sim_global, ipeak2f_sim_local, tpeak2f_sim, _ = findpeak2f(sim_filtered,bkg1f_filtered,ind=ind)
                    print('Same peak value found: {}. Changing vshift[1] from {} to {} gives: {}'.format(ipeak2f_sim[0],vshift[1],vshift_new,ipeak2f_sim_global))
                    ipeak2f_sim[1] = ipeak2f_sim_global
                    vshift[1] = vshift_new
            else:
                vshift_new = vshift[1] + (np.diff(vshift)[0]/np.diff(ipeak2f_sim)[0])*(ipeak2f_meas-ipeak2f_sim[1])
                sim_filtered = [filt.applyLockIn(sim.t,sim.Simulate(T,P,X,vshift_new)) for filt in filts]
                ipeak2f_sim_global, ipeak2f_sim_local, tpeak2f_sim, _ = findpeak2f(sim_filtered,bkg1f_filtered,ind=ind)
                print('Newton-Raphson: {} gives {}'.format(vshift_new,ipeak2f_sim_global))
                i_keep = np.argmin(np.abs(np.array(ipeak2f_sim)-ipeak2f_sim_global))
                ipeak2f_sim = [ipeak2f_sim[i_keep], ipeak2f_sim_global]
                vshift = [vshift[i_keep], vshift_new]
        print(vshift, ipeak2f_sim, ipeak2f_meas)
        if iteration >= maxiter:
            raise RuntimeError('Maximum iterations reached in wms.calcvshift')
    else:
        if iteration == 0:
            print('First try... {}, {}, {}'.format(vshift,ipeak2f_sim, ipeak2f_meas))
    return vshift[-1], ipeak2f_sim[-1], ipeak2f_sim_local

def calcTwoLineRatioFromFilteredResults(datafiltered,bkg=None,ipeaks=None,ind=None):
    """  """
    if ipeaks is None:
        if bkg is None:
            raise RuntimeError('Background kwarg required if ipeaks is None in wms.calcTwoLineRatioFromFilteredResults')
        ipeakLD1,_,_,_ = findpeak2f(datafiltered[0],bkg[0][0],ind=ind)
        ipeakLD2,_,_,_ = findpeak2f(datafiltered[1],bkg[1][0],ind=ind)
    else:
        ipeakLD1, ipeakLD2 = ipeaks
    R2f1fLD1, R2f1fLD2 = calcnf1fRatios() #FromFilteredResults(datafiltered, bkg=bkg)
    R2f1fLD1_peak = R2f1fLD1[0][ipeakLD1]
    R2f1fLD2_peak = R2f1fLD2[0][ipeakLD2]
    return R2f1fLD1_peak/R2f1fLD2_peak, R2f1fLD1_peak, R2f1fLD2_peak

def calcTwoLineRatio(T,P,X,sims,vScanParams,filts,measfiltered,bkg=None,ipeaks=None,ind=None,maxiter=None,lines=None,previousshifts=None):
    """  """
    lineLD1 = None if lines is None else lines[0]
    lineLD2 = None if lines is None else lines[1]
    previousLD1 = None if previousshifts is None else previousshifts[0]
    previousLD2 = None if previousshifts is None else previousshifts[1]
    simLD1, simLD2 = sims
    vScanParamsLD1, vScanParamsLD2 = vScanParams
    measfilteredLD1, measfilteredLD2 = measfiltered
    vshiftLD1, ipeakLD1, _ = calcvshift(T, P, X, simLD1, vScanParamsLD1, filts[0], measfilteredLD1, bkg[0][0], ind=ind, maxiter=maxiter, line=lineLD1, previous=previousLD1)
    vshiftLD2, ipeakLD2, _ = calcvshift(T, P, X, simLD2, vScanParamsLD2, filts[1], measfilteredLD2, bkg[1][0], ind=ind, maxiter=maxiter, line=lineLD2, previous=previousLD2)
    simfiltered = filters.applyLockIns(filts, simLD1.t, simLD1.Simulate(T, P, X, vshiftLD1)+simLD2.Simulate(T, P, X, vshiftLD2))
    return calcTwoLineRatioFromFilteredResults(simfiltered,bkg=bkg,ipeaks=ipeaks,ind=ind), (vshiftLD1, vshiftLD2), (ipeakLD1, ipeakLD2)

def fitT_Ratio(T0,measuredRatio,P,X,sims,vScanParams,filts,measfiltered,bkg=None,ipeaks=None,ind=None,lines=None,previousshifts=None,tol=1,maxiter=50):
    """  """
    print('T_guess = {}'.format(T0))
    fitEq = lambda T,prev: calcTwoLineRatio(T,P,X,sims,vScanParams,filts,measfiltered,bkg=bkg,ipeaks=ipeaks,ind=ind,maxiter=maxiter,lines=lines,previousshifts=prev)
    Ratios, Shifts, Peaks = fitEq(T0[0],previousshifts)
    prevLD1 = Shifts[0], [T0[0],P,X,sims[0].L,sims[0].spectras,None if lines is None else lines[0]]
    prevLD2 = Shifts[1], [T0[0],P,X,sims[1].L,sims[1].spectras,None if lines is None else lines[1]]
    prev = [prevLD1,prevLD2]
    Ratios = [Ratios, fitEq(T0[1],prev)[0]]
    iteration = 0
    while np.abs(np.diff(T0)[0])>tol:
        iteration += 1
        T_new = (T0[1]-T0[0])/(Ratios[1][0]-Ratios[0][0]) * (measuredRatio-Ratios[0][0]) + T0[0]
        Ratio_new, vshifts, ipeaks = fitEq(T_new,prev)
        prevLD1 = vshifts[0], [T_new,P,X,sims[0].L,sims[0].spectras,None if lines is None else lines[0]]
        prevLD2 = vshifts[1], [T_new,P,X,sims[1].L,sims[1].spectras,None if lines is None else lines[1]]
        prev = [prevLD1,prevLD2]
        i_keep = np.argmin(np.abs(np.array([Ratios[0][0],Ratios[1][0]])-Ratio_new[0]))
        T0 = [T0[i_keep], T_new]
        Ratios = [Ratios[i_keep], Ratio_new]
        print('T0 = {}'.format(T0))
        if iteration >= maxiter:
            raise RuntimeError('Maximum iterations reached in wms.fitT_Ratio')
    return T_new, Ratio_new, vshifts, ipeaks

def fitTX_Ratios(X0,T0,measuredRatios,P,sims,vScanParams,filts,measfiltered,bkg=None,ipeaks=None,ind=None,lines=None,previousshifts=None,Xtol=0.005,Ttol=1,maxiter=50,LD1=True):
    """  """
    TwoLineRatio, nf1fRatio = measuredRatios
    i_laser = 1 if LD1 else 2
    print('X_guess = {}'.format(X0))
    fitEq = lambda X,prev,Tguesses: fitT_Ratio(Tguesses,TwoLineRatio,P,X,sims,vScanParams,filts,measfiltered,bkg=bkg,ipeaks=ipeaks,ind=ind,lines=lines,previousshifts=prev,tol=Ttol,maxiter=maxiter)
    Ts, Ratios, Shifts, Peaks = fitEq(X0[0],previousshifts,T0)
    prevLD1 = Shifts[0], [Ts,P,X0[0],sims[0].L,sims[0].spectras,None if lines is None else lines[0]]
    prevLD2 = Shifts[1], [Ts,P,X0[1],sims[1].L,sims[1].spectras,None if lines is None else lines[1]]
    prev = [prevLD1,prevLD2]
    Ts2, Ratios2, _, _ = fitEq(X0[1],prev,T0)
    nf1fs = [Ratios, Ratios2]
    T0 = [np.min([Ts,Ts2])-Ttol, np.max([Ts,Ts2])+Ttol] if np.abs(Ts2-Ts) <= Ttol else [Ts,Ts2]
    iteration = 0
    while np.abs(np.diff(X0)[0])>Xtol:
        iteration += 1
        X_new = (X0[1]-X0[0])/(nf1fs[1][i_laser]-nf1fs[0][i_laser]) * (nf1fRatio-nf1fs[0][i_laser]) + X0[0]
        Tfit, nf1f_new, vshifts, ipeaks = fitEq(X_new,prev,T0)
        prevLD1 = vshifts[0], [Tfit,P,X_new,sims[0].L,sims[0].spectras,None if lines is None else lines[0]]
        prevLD2 = vshifts[1], [Tfit,P,X_new,sims[1].L,sims[1].spectras,None if lines is None else lines[1]]
        prev = [prevLD1,prevLD2]
        i_keep = np.argmin(np.abs(np.array([nf1fs[0][i_laser],nf1fs[1][i_laser]])-nf1f_new[i_laser]))
        X0 = [X0[i_keep], X_new]
        nf1fs = [nf1fs[i_keep], nf1f_new]
        print('X_new = {}'.format(X_new))
        if iteration >= maxiter:
            raise RuntimeError('Maximum iterations reached in wms.fitTX_Ratios')
    return Tfit, X_new, nf1f_new, vshifts, ipeaks
