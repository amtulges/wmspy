import numpy as np
from utilities import plotData

class Data(object):
    """
    Base data object
    """

    def __init__(self,I,t=None):
        """
        datatypes.Data(I, t=None)
        I = numpy array of light intensity
        t = numpy array of time, optional
        if time not specified it must be created by self.loadConfig()
        """
        if not isinstance(I,np.ndarray):
            raise TypeError('First input to datatypes.Data must be numpy array')
        if (t is not None) and not isinstance(t,np.ndarray):
            raise TypeError('t input to datatypes.Data must be numpy array')
            if I.size != t.size:
                raise ValueError('I and t inputs to datatypes.Data must be same size')
        self.I = I
        self.t = t
        self.npoints = len(I)

    def loadConfig(self,configObject,etalon=False):
        """
        Data.loadConfig(configObject, etalon=False)
        configObject = utilities.Configuration object
        etalon = bool, True for etalon settings, False for other measurement settings
        Loads configObject settings to attributes of self
        """
        settings = configObject.exportSettings(etalon=etalon)
        for key,value in settings.items():
            setattr(self,key,value)

    def checkTime(self):
        """
        Data.checkTime()
        If no time array exists, creates time array attribute self.t having length of self.I with dt = 1/self.SR
        """
        if self.t is None:
            self.t = np.arange(len(self.I))/self.SR

    def calcNumberOfScansAndModulations(self):
        """
        Data.calcNumberOfScansAndModulations()
        Creates attributes self.Ns and self.Nm for number of scan and modulation periods in measurement
        """
        self.Ns = self.fs * (self.t[-1]-self.t[0])
        self.Nm = [fm * (self.t[-1]-self.t[0]) for fm in self.fm]

    def trimSettings(self,ind):
        """
        Data.trimSettings(ind)
        ind = int of laser index, e.g. 0, 1, 2, ...
        Trims attributes from lists to floats for specific laser index
        Applies to: fm, Tm, modsperscan, nm, Nm
        """
        self.fm = self.fm[ind]
        self.Tm = self.Tm[ind]
        self.modsperscan = self.modsperscan[ind]
        self.nm = self.nm[ind]
        self.Nm = self.Nm[ind]

    def loadConfigAndProcessSettings(self,configObject, etalon=False, ind=None):
        """
        Data.loadConfigAndProcessSettings(configObject, etalon=False, ind=None)
        configObject = utilities.Configuration object
        etalon = bool, True for etalon settings, False for other measurement settings
        ind = int of laser index, e.g. 0, 1, 2, ...
        Loads configObject settings to attributes of self
        If no time array exists, creates time array attribute self.t having length of self.I with dt = 1/self.SR
        Creates attributes self.Ns and self.Nm for number of scan and modulation periods in measurement
        Trims attributes from lists to floats for specific laser index for: fm, Tm, modsperscan, nm, Nm
        """
        Data.loadConfig(self,configObject, etalon=etalon)
        self.checkTime()
        self.calcNumberOfScansAndModulations()
        if ind is not None:
            self.trimSettings(ind)

    def nearestIndex(self, t):
        """
        Data.nearestIndex(t):
        Finds nearest index or indices to the given time(s)
        t = float or array of times
        Returns int index or int array of indices
        """
        def nearest_i(t):
            return np.argmin(np.abs(self.t - t))
        if isinstance(t, np.ndarray) or isinstance(t, list):
            i = np.array([nearest_i(t_) for t_ in t])
        else:
            i = nearest_i(t)
        return i

    def plot(self, ind=(0,None)):
        """
        Data.plot(ind=(0,None)):
        Plots data in a new figure.
        ind = (i_min, i_max) index bounds to plot. Default is all data
        Returns None
        """
        i_min, i_max = ind
        tplot = self.t[i_min:i_max] if self.t is not None else np.arange(len(self.I))
        plotData((tplot, self.I[i_min:i_max]), linestyles='k-', labels='data', xlabel='Time (s)', ylabel='Signal (V)')

class Measurement(Data):
    """
    Measurement data object
    """

    def __init__(self, I, t=None, config=None):
        """
        datatypes.Measurement(I, t=None, config=None)
        I = numpy array of light intensity
        t = numpy array of time, optional, if not specified it will be created when using self.loadConfig()
        config = utilities.Configuration object, optional, if not specified must use self.loadConfig method
        """
        super(Measurement, self).__init__(I, t=t)
        if config is not None:
            self.loadConfig(config)

    def loadConfig(self,configObject):
        """
        Measurement.loadConfig(configObject)
        configObject = utilities.Configuration object
        Loads configObject settings to attributes of self
        Then creates time array if none exists, and calculates number of scans and modulations
        """
        super(Measurement, self).loadConfigAndProcessSettings(configObject, etalon=False, ind=None)

class Dark(Data):
    """
    Dark data object
    """

    def __init__(self, I):
        """
        datatypes.Dark(I, t=None)
        I = numpy array of light intensity
        calculates average intensity as attribute self.avg
        """
        super(Dark, self).__init__(I)
        self.avg = np.mean(self.I)

class Reference(Data):
    """
    Reference data object
    """

    def __init__(self,I, t=None, config=None, ind=0):
        """
        datatypes.Reference(I, t=None, config=None, ind=0)
        I = numpy array of light intensity
        t = numpy array of time, optional, if not specified it will be created when using self.loadConfig()
        config = utilities.Configuration object, optional, if not specified must use self.loadConfig method
        ind = int of laser index, default = 0
        """
        super(Reference, self).__init__(I, t=t)
        if config is not None:
            self.loadConfig(config, ind=ind)

    def loadConfig(self,configObject, ind=0):
        """
        Reference.loadConfig(configObject, ind=0)
        configObject = utilities.Configuration object
        ind = int of laser index, e.g. 0, 1, 2, ...
        Loads configObject settings to attributes of self
        Then creates time array if none exists, and calculates number of scans and modulations
        """
        super(Reference, self).loadConfigAndProcessSettings(configObject, etalon=False, ind=ind)

class Background(Data):
    """
    Background data object
    """

    def __init__(self, I, t=None, config=None):
        """
        datatypes.Background(I, t=None, config=None)
        I = numpy array of light intensity
        t = numpy array of time, optional, if not specified it will be created when using self.loadConfig()
        config = utilities.Configuration object, optional, if not specified must use self.loadConfig method
        """
        super(Background, self).__init__(I, t=t)
        if config is not None:
            self.loadConfig(config)

    def loadConfig(self,configObject):
        """
        Background.loadConfig(configObject)
        configObject = utilities.Configuration object
        Loads configObject settings to attributes of self
        Then creates time array if none exists, and calculates number of scans and modulations
        """
        super(Background, self).loadConfigAndProcessSettings(configObject, etalon=False, ind=None)

class Etalon(Data):
    """
    Etalon data object
    """

    def __init__(self,I, t=None, config=None, ind=0):
        """
        datatypes.Etalon(I, t=None, config=None, ind=0)
        I = numpy array of light intensity
        t = numpy array of time, optional, if not specified it will be created when using self.loadConfig()
        config = utilities.Configuration object, optional, if not specified must use self.loadConfig method
        ind = int of laser index, default = 0
        """
        super(Etalon, self).__init__(I, t=t)
        if config is not None:
            self.loadConfig(config, ind=ind)

    def loadConfig(self,configObject, ind=0):
        """
        Etalon.loadConfig(configObject, ind=0)
        configObject = utilities.Configuration object
        ind = int of laser index, e.g. 0, 1, 2, ...
        Loads configObject settings to attributes of self
        Then creates time array if none exists, and calculates number of scans and modulations
        """
        super(Etalon, self).loadConfigAndProcessSettings(configObject, etalon=True, ind=ind)
