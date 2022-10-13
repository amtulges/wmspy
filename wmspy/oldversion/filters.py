import numpy as np
from scipy.signal import firwin, iirfilter, filtfilt, freqz, resample_poly, kaiserord, cheb2ord, buttord

from utilities import plotData, FFT, dB2gain, gain2dB

class WMSFilters(object):
    """
    Filters for Wavelength Modulation Spectroscopy
    """

    def __init__(self, configObject, harmonics=(1,2), f_lowpass=None, f_stopband=None, SR=None, passbandripple=2, stopbandripple=60, fir=True, ftype=None, multirate=True, npoints=None, lowpassfactor=10, stopbandfactor=8, plotflag=False):
        """
        Creates a lock-in filters at each harmonic of each modulation frequency in configObject.fm
        stored to the attribute self.filters which is a list of lists indexed as
        self.filters[i][j] where i is the i-th modulation frequency and j is the j-th harmonic
        configObject = utilities.Configuration object with measurement parameters
        harmonics = tuple of ints, default (1,2)
        f_lowpass = list of lists of lowpass frequencies with same shape as self.filters
        f_stopband = list of lists of stopband frequencies with same shape as self.filters
        SR = sample rate in Hertz, default is configObject.SR
        passbandripple = maximum passband ripple in db
        stopbandripple = minimum stopband attenuation in dB
        fir = bool, whether to use fir or iir filters
        multirate = bool, whether to use multirate-multistage filtering
        npoints = number of points in data to be filtered, for optimizing order and nstages
        lowpassfactor = float used to scale modulation frequency to create f_lowpass if None given
        stopbandfactor = float used to scale modulation frequency to create f_stopband if None given
        plotflag = bool
        """
        # Create default ftype if necessary
        if ftype is None:
            ftype = 'kais' if fir else 'cheb2'
        # Create default f_lowpass if necessary
        if f_lowpass is None:
            f_lowpass = []
            for i, fm in enumerate(configObject.fm):
                f_low = []
                for j, harmonic in enumerate(harmonics):
                    f_low.append(fm/lowpassfactor)
                f_lowpass.append(f_low)
        # Create default f_stopband if necessary
        if f_stopband is None:
            f_stopband = []
            for i, fm in enumerate(configObject.fm):
                f_stop = []
                for j, harmonic in enumerate(harmonics):
                    f_stop.append(fm/stopbandfactor)
                f_stopband.append(f_stop)
        # Get default sample rate from configObject
        if SR is None:
            SR = configObject.SR
        # Create filters
        self.filters = []
        for i, fm in enumerate(configObject.fm):
            filters_ = []
            for j, harmonic in enumerate(harmonics):
                filters_.append(Filter(f_lowpass[i][j],
                                       f_stopband[i][j],
                                       SR,
                                       f_center = harmonic*fm,
                                       passbandripple = passbandripple,
                                       stopbandripple = stopbandripple,
                                       fir = fir,
                                       ftype = ftype,
                                       multirate = multirate,
                                       npoints = npoints,
                                       plotflag = plotflag))
            self.filters.append(filters_)

    def applyLockIns(self, t, data, plotflag=False, plotflag_FFT=False):
        """
        Apply Lock-in filters to data
        t = time array
        data = data array
        plotflag = bool
        plotflag_FFT = bool
        Output list of lists of (t_decimated, (R,theta,X,Y)) for each filter's results.
        """
        results = []
        for filters_ in self.filters:
            results_ = []
            for filt in filters_:
                results_.append(filt.applyLockIn(t,
                                                 data,
                                                 plotflag = plotflag,
                                                 plotflag_FFT = plotflag_FFT))
            results.append(results_)
        return results

class Filter(object):
    """
    FIR or IIR, single or multistage-multirate filter
    """

    def __init__(self, f_lowpass, f_stopband, f_sample, f_center=0, passbandripple=2, stopbandripple=60, fir=True, ftype='kais', multirate=True, decimationfactor=None, nstages=None, npoints=None, plotflag=False):
        """
        Create multistage-multirate FIR or IIR low-pass filter (to be used as a lock-in) with attributes:
            coeffs = dict of coefficients with keys 'b' and 'a'
            order, ntaps = size of filter
            theta_pass, theta_stop = normalized pass and stop band frequencies
            nstages, decimationfactor = multirate parameters
            f_lowpass, f_stopband, f_center = frequencies in Hz
            f_Nyquist, f_sample = frequencies in Hz
            stopbandripple, passbandripple, stopripplerperstage, passrippleperstage = band parameters
            fir, multirate = filter bools
            decimationfactor, nstages = corresponding parameters of multirate filter are forced to match these inputs
            ftype = abbreviated name of filter type, ('kais') for fir or ('cheb2', 'butt') for iir
            npoints = size of data to be filtered, for optimizing nstages and decimationfactor
        """
        # Collect inputs
        self.f_sample  = f_sample
        self.f_Nyquist = self.f_sample/2
        self.f_lowpass = f_lowpass
        self.f_stopband= f_stopband
        self.f_center  = f_center
        self.stopbandripple = stopbandripple
        self.passbandripple = passbandripple
        self.fir       = fir
        self.type      = ftype.lower()
        self.multirate = multirate
        self.targetdecimation = decimationfactor
        self.targetstages     = nstages
        self.npoints   = npoints
        # Create Filter
        self.designFilter()
        # Calculate Coefficients
        self.calcCoefficients()
        # Plot
        if plotflag:
            self.plot()

    def designFilter(self):
        """
        Calculates optimal filter parameters
        """
        oversample_factor = self.f_Nyquist / self.f_stopband
        if (self.multirate) and (oversample_factor > 4):
            decimation_list = np.arange(2, int(np.round(np.min([np.sqrt(oversample_factor), 10])))) if self.targetdecimation is None else [self.targetdecimation]
            stages_list  = [int(np.floor(np.log(oversample_factor)/np.log(dec))) for dec in decimation_list] if self.targetstages is None else [self.targetstages]
            downsampling = [dec**stages for dec,stages in zip(decimation_list,stages_list)]
            Nyqref_list  = [self.f_Nyquist/(dec**(stages-1)) for dec,stages in zip(decimation_list,stages_list)]
            passripple_list  = [self.passbandripple/stages for stages in stages_list]
            stopripple_list = [-gain2dB(1-(1-dB2gain(-self.stopbandripple))**stages) for stages in stages_list]
            if self.fir:
                if self.type == 'kais':
                    order_list = [kaiserord(stopripp, (self.f_stopband-self.f_lowpass)/Nyqref)[0] for stopripp,Nyqref in zip(stopripple_list,Nyqref_list)]
            else:
                if self.type == 'cheb2':
                    order_list = [cheb2ord(self.f_lowpass/Nyqref, self.f_stopband/Nyqref, passripp, stopripp)[0] for stopripp,passripp,Nyqref in zip(stopripple_list,passripple_list,Nyqref_list)]
                elif self.type == 'butt':
                    order_list = [ buttord(self.f_lowpass/Nyqref, self.f_stopband/Nyqref, passripp, stopripp)[0] for stopripp,passripp,Nyqref in zip(stopripple_list,passripple_list,Nyqref_list)]
            if self.npoints is None:
                i_best = np.argmin(downsampling)
            else:
                nremaining = [(self.npoints/downsample)-2*np.ceil(order/dec) for downsample,order,dec in zip(downsampling,order_list,decimation_list)]
                i_best = np.argmax(nremaining)
            self.decimationfactor = decimation_list[i_best]
            self.nstages   = stages_list[i_best]
            Nyq_ref        = Nyqref_list[i_best]
            self.stoprippleperstage = stopripple_list[i_best]
            self.passrippleperstage = passripple_list[i_best]
        else:
            self.decimationfactor = 1
            self.nstages   = 1
            Nyq_ref        = self.f_Nyquist
            self.stoprippleperstage = self.stopbandripple
            self.passrippleperstage = self.passbandripple
        self.theta_pass    = self.f_lowpass/Nyq_ref
        self.theta_stop    = self.f_stopband/Nyq_ref
        if self.fir:
            if self.type == 'kais':
                taps, self.beta = kaiserord(self.stoprippleperstage, self.theta_stop-self.theta_pass)
            self.ntaps = taps if np.mod(taps,2)==1 else taps+1
            self.order = self.ntaps - 1
        else:
            if self.type == 'cheb2':
                self.order, self.wn = cheb2ord(self.theta_pass, self.theta_stop, self.passrippleperstage, self.stoprippleperstage)
            elif self.type == 'butt':
                self.order, self.wn =  buttord(self.theta_pass, self.theta_stop, self.passrippleperstage, self.stoprippleperstage)
            self.ntaps = self.order + 1

    def calcCoefficients(self):
        """
        Calculates filter coefficients
        """
        self.coeffs = {}
        if self.fir:
            if self.type == 'kais':
                self.coeffs['b'] = firwin(self.ntaps, np.mean([self.theta_pass, self.theta_stop]), window=('kaiser',self.beta))
            self.coeffs['a'] = 1
        else:
            typestr = {'butt'  : 'butter',
                       'cheb1' : 'cheby2',
                       'cheb2' : 'cheby2',
                       'ellip' : 'ellip',
                       'bess'  : 'bessel'}
            self.coeffs['b'], self.coeffs['a'] = iirfilter(self.order, self.theta_stop,
                                                          rp=self.passrippleperstage, rs=self.stoprippleperstage,
                                                          btype='lowpass', analog=False, output='ba',
                                                          ftype=typestr[self.type])

    def calcGain(self, freq=None, npoints=1000, dB=False):
        """
        Calculate the gain of the filter at the absolute frequencies (Hz).
        freq = array, default to 0-Nyquist with 1000 evenly spaced points
        npoints = int, number of frequencies to use
        dB = bool, whether to output as amplitude gain or dB
        Returns freq, gain.
        """
        if freq is None:
            freq = np.linspace(0, self.f_Nyquist, num=npoints)
        gain = 1
        for stage in range(self.nstages):
            upsamplefactor = self.decimationfactor**stage
            stage_coeffs_b = np.zeros(self.order*upsamplefactor+1)
            stage_coeffs_b[::upsamplefactor] = self.coeffs['b']
            if not self.fir:
                stage_coeffs_a = np.zeros(self.order*upsamplefactor+1)
                stage_coeffs_a[::upsamplefactor] = self.coeffs['a']
            _,h = freqz(stage_coeffs_b,
                        a = 1 if self.fir else stage_coeffs_a,
                        worN = np.divide(freq, self.f_Nyquist) * np.pi)
            gain *= np.absolute(h)
        if dB:
            gain = gain2dB(gain)
        return freq, gain

    def applyLockIn(self, t, data, plotflag=False, plotflag_FFT=False):
        """
        Apply lock-in filter to the input data.
        t = time array
        data = data array
        plotflag = bool
        plotflag_FFT = bool
        Return t_decimated, (R,theta,X,Y).
        """
        X = np.cos( 2*np.pi * self.f_center * t) * data
        Y = np.sin( 2*np.pi * self.f_center * t) * data
        for i in range(self.nstages):
            if self.fir:
                X = resample_poly(X, 1, self.decimationfactor, window=self.coeffs['b'], padtype='smooth')
                Y = resample_poly(Y, 1, self.decimationfactor, window=self.coeffs['b'], padtype='smooth')
            else:
                X = filtfilt(self.coeffs['b'], self.coeffs['a'], X)
                Y = filtfilt(self.coeffs['b'], self.coeffs['a'], Y)
                X = X[::self.decimationfactor]
                Y = Y[::self.decimationfactor]
        nanlength = int(np.ceil(self.order/self.decimationfactor))
        X[:nanlength]  = np.NaN
        X[-nanlength:] = np.NaN
        Y[:nanlength]  = np.NaN
        Y[-nanlength:] = np.NaN
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan(np.divide(Y, X))
        t_decimated = t[::self.decimationfactor**self.nstages]
        if plotflag_FFT:
            # Plot FFT
            freq_filt = np.linspace(0,
                                    self.f_center+3.0*self.f_lowpass if self.f_center+3.0*self.f_lowpass<self.f_Nyquist else self.f_Nyquist,
                                    num=1000)
            _, gain_filt = self.calcGain(freq=freq_filt)
            freq_fft, gain_fft = FFT(t, data)
            plotData([(self.f_center+freq_filt, gain_filt),
                      (self.f_center-freq_filt, gain_filt),
                      (freq_fft, gain_fft)],
                     linestyles = ['r-', 'r-', 'k-'],
                     labels = ['filter', '', 'data'],
                     xlabel = 'Frequency (Hz)',
                     ylabel = 'Gain',
                     title = 'Filters.applyLockIn: FFT',
                     ylog = True,
                     xlim = (0, 2.0*self.f_center if self.f_center>0 else 3.0*self.f_lowpass),
                     ylim = (np.min(gain_fft), 1.05))
        if plotflag:
            # Plot Data
            xy = [(t, data)] if self.f_center == 0 else []
            linestyles = ['k-'] if self.f_center == 0 else []
            labels = ['raw'] if self.f_center == 0 else []
            xy.append((t_decimated, R))
            linestyles.append('r-')
            labels.append('filtered')
            plotData(xy,
                     linestyles = linestyles,
                     labels = labels,
                     xlabel = 'Time (s)',
                     ylabel = 'Signal (V)',
                     title = 'Filter.applyLockIn: Data')
        return t_decimated, (R,theta,X,Y)

    def plot(self, fmax=None, npoints=1000, linestyle='r-', dB=True, xlog=False, ylog=False, fig=None):
        """
        Plots the filter gain curve
        fmax = maximum frequency to display measured from f_center
        npoints = number of frequencies to show
        linestyle = default 'r-'
        fig = figure handle to display on
        """
        if fmax is None:
            fmax = np.min([10*self.f_stopband, self.f_Nyquist])
        freq = np.linspace(0, fmax, num=npoints)
        freq, gain = self.calcGain(freq=freq, dB=dB)
        if self.f_center != 0:
            freq = self.f_center + np.concatenate((-freq[::-1], freq))
            gain = np.concatenate((gain[::-1], gain))
        plotData((freq,gain),
                 labels = 'filter',
                 linestyles = linestyle,
                 xlabel = 'Frequency (Hz)',
                 ylabel = 'Gain (dB)',
                 title = 'Filter Gain',
                 xlog = xlog,
                 ylog = ylog,
                 fig = fig)
