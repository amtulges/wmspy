import numpy as np
import yaml
from scipy import constants
from pathlib import Path
from matplotlib import pyplot as plt

from utilities import Voigt, plotData

class Line(object):
    """
    Spectral line object
    """
    c2 = constants.physical_constants['second radiation constant'][0] * 100 # K/cm-1
    NA = constants.N_A # unitless
    P0 = constants.physical_constants['standard atmosphere'][0] # Pa
    R  = constants.R # J/(mol*K)

    def __init__(self, v_transition, linestrength, E_lowerstate, gammas, molarmass, TIPSparams, gamma_powers=None, deltaP=0, deltaP_power=0, Tref=296.0):
        """
        simulation.Line(v_transition, linestrength, A_Einstein, E_lowerstate, gammas, gamma_powers=None, deltaP=0, deltaP_power=0, Tref=296.0)
        v_transition = theoretical vacuum transition wavenumber (cm-1)
        linestrength = spectral line intensity (cm-1 / (molecule * cm-2))
        E_lowerstate = lower state energy (cm-1)
        gammas = HWHM's starting with gamma_self (cm-1/atm)
        molarmass = molar mass (g/mol)
        TIPSparams = total internal partition sum parameter dictionary with keys
            Tlow, Tmid, Thigh, a, b, c, d
            where a, b, c, d are arrays of low, mid, high coefficients
        gamma_powers = Temperature dependence power(s) of the linewidth(s), Default 0
        deltaP = Pressure shift coefficient of the transition wavenumber (cm-1/atm), Default 0
        deltaP_power = Temperature dependence power of the pressure shift, Default 0
        Tref = Reference temperature (K), Default 296.0
        """
        self.v = v_transition
        self.S = linestrength
        self.E = E_lowerstate
        self.gammas = gammas
        self.MM = molarmass
        self.TIPSparams = TIPSparams
        if gamma_powers is None:
            if isinstance(gammas, (int, float)):
                self.ns = 0
            else:
                self.ns = np.array([0] * len(gammas))
        else:
            self.ns = gamma_powers
        self.deltaP = deltaP
        self.nP = deltaP_power
        self.Tref = Tref

    def TIPS(self, T):
        """
        Total internal partition sum function
        T = temperature in Kelvin
        Returns total internal partition sum
        """
        if T > self.TIPSparams['Thigh']:
            i = 2
        elif T > self.TIPSparams['Tmid']:
            i = 1
        elif T > self.TIPSparams['Tlow']:
            i = 0
        else:
            raise ValueError('Input temperature {} K below lowest temperature range {} K of internal partition sum in simulation.Line.TIPS'.format(T,self.TIPSparams['Tlow']))
        return self.TIPSparams['a'][i]          + \
               self.TIPSparams['b'][i] * T      + \
               self.TIPSparams['c'][i] * (T**2) + \
               self.TIPSparams['d'][i] * (T**3)

    def calcAbsorbanceParameters(self, T, P, X, L):
        """
        Calculate v0, A0, dvD0, dvC0 from given conditions
        T = temperature (K)
        P = pressure (atm)
        X = mole fraction of line species, or list of mole fractions of broadening species including self
        L = path length (m)
        Returns:
            v0 = pressure shifted line center (cm-1)
            A0 = temperature adjusted line strength (cm-1)
            dvD0 = Doppler FWHM (cm-1)
            dvC0 = Lorentzian / collisional FWHM (cm-1)
        """
        v0 = self.v + (P * self.deltaP * (self.Tref/T)**self.nP) # cm-1
        A0 = P * (X if isinstance(X, (int, float)) else X[0]) * L * self.NA * self.P0 * self.S / (self.R * T * 100**2) * \
             (self.TIPS(self.Tref) / self.TIPS(T)) * \
             np.exp(-self.c2 * self.E * (1/T - 1/self.Tref)) * \
             (1 - np.exp(-self.c2 * self.v/T)) / (1 - np.exp(-self.c2 * self.v/self.Tref))
        dvD0 = 7.162E-7 * v0 * np.sqrt(T/self.MM)
        if isinstance(X, (int, float)):
            X = [X, 1-X]
        dvC0 = 0
        for gamma, n, X_i in zip(self.gammas, self.ns, X):
            dvC0 += 2 * P * gamma * (self.Tref/T)**n * X_i
        return v0, A0, dvD0, dvC0

    def calcAbsorbance(self, v, T, P, X, L, plotflag=False):
        """
        Calculate absorbance at each wavenumber v, for the given conditions
        v = absolute wavenumber (cm-1)
        T = temperature (K)
        P = pressure (atm)
        X = mole fraction of line species, or list of mole fractions of broadening species including self
        L = path length (m)
        plotflag = bool
        Returns array of absorbances
        """
        v0, A0, dvD0, dvC0 = self.calcAbsorbanceParameters(T, P, X, L)
        dv = v - v0
        a  = np.sqrt(np.log(2)) * dvC0/dvD0
        w  = dv*2*np.sqrt(np.log(2))/dvD0
        absorbance = Voigt(a,w) * A0 * 2*np.sqrt(np.log(2)/np.pi)/dvD0
        if plotflag:
            plotData((v,absorbance),
                     linestyles = 'k-',
                     labels = 'absorbance',
                     xlabel = 'Wavenumber(cm-1)',
                     ylabel = 'Absorbance')
        return absorbance

class Spectra(object):
    """
    Spectra containing single or multiple lines
    """
    linekeys = ['transitionwavenumber',
                'lineintensity',
                'lowerstateenergy',
                'selfbroadenedwidth',
                'airbroadenedwidth',
                'pressureshift',
                'temperaturedependenceofselfwidth',
                'temperaturedependenceofairwidth',
                'temperaturedependenceofpressureshift',
                'molarmass',
                'referencetemperature']

    def __init__(self, spectrafile, tipsfile, hitran=False, molarmass=None, Tref=296.0):
        """
        Read, import, and store the spectral information
        spectrafile = filepath of Hitran or custom yaml file
        tipsfile = filepath or list of filepaths of total internal partition sum parameter file(s)
        hitran = bool, whether spectrafile is hitran .par or custom .yaml file
        molarmass = numeric or array of molecular mass(es) in line(s); used when hitran=True
        Tref = numeric or array of reference temperature(s), default 296.0 K; used when hitran=True
        """
        if hitran:
            self.lineparams = self.importHitran(spectrafile, molarmass, Tref=Tref)
        else:
            self.lineparams = self.importCustom(spectrafile)
        self.tipsparams = self.importPartitionSumParameters(tipsfile)
        self.createLines()

    def importHitran(self, parfile, molarmass, Tref=296.0):
        """
        Import line parameters from Hitran 2004+ formatted .par text file
        parfile = filepath of Hitran .par text file
        molarmass = numeric or array of molecular mass(es) in line(s)
        Tref = numeric or array of reference temperature(s), default 296.0 K
        Return dictionary of line parameters
        """
        # Initialize dictionary
        lineparams = dict()
        for key in self.linekeys:
            lineparams[key] = []
        # Read the file
        with open(Path(parfile), 'r') as f:
            iline = 0
            for textline in f.readlines():
                ## Parse line parameters
                #moleculenumber = int(textline[0:2])
                #isotopologuenumber = int(textline[2:3])
                lineparams['transitionwavenumber'].append( float(textline[3:15]) )
                lineparams['lineintensity'].append( float(textline[15:25]) )
                #lineparams['einsteinacoefficient'].append( float(textline[25:35]) )
                lineparams['airbroadenedwidth'].append( float(textline[35:40]) )
                lineparams['selfbroadenedwidth'].append( float(textline[40:45]) )
                lineparams['lowerstateenergy'].append( float(textline[45:55]) )
                lineparams['temperaturedependenceofairwidth'].append( float(textline[55:59]) )
                lineparams['pressureshift'].append( float(textline[59:67]) )
                #uppervibrationalquanta = textline[67:82]
                #lowervibrationalquanta = textline[82:97]
                #upperlocalquanta = textline[97:112]
                #lowerlocalquanta = textline[112:127]
                #errorcodes = int(textline[127:133])
                #referencecodes = int(textline[133:145])
                #flagforlinemixing = textline[145:146]
                #upperstatisticalweight = float(textline[146:153])
                #lowerstatisticalweight = float(textline[153:160])
                ## Include additional required line parameters
                lineparams['molarmass'].append( molarmass if isinstance(molarmass, (int, float)) else molarmass[iline] )
                lineparams['referencetemperature'].append( Tref if isinstance(Tref, (int, float)) else Tref[iline] )
                lineparams['temperaturedependenceofselfwidth'].append( float(textline[55:59]) )
                lineparams['temperaturedependenceofpressureshift'].append( 0 )
                iline += 1
        # If only a single line, reduce lists to numerics
        if len(lineparams['transitionwavenumber']) == 1:
            for key, value in lineparams.items():
                lineparams[key] = value[0]
        return lineparams

    def importCustom(self, yamlfile):
        """
        Import custom line parameters from yaml file
        Return dictionary of line parameters
        """
        # Load data
        lineparams = yaml.safe_load(open(Path(yamlfile)))
        # Check for all necessary parameters
        for key in self.linekeys:
            if key not in lineparams.keys():
                raise ValueError('No {} entry found in yamlfile {}'.format(key,yamlfile))
            if lineparams[key] is None:
                raise ValueError('No {} values found in yamlfile {}'.format(key, yamlfile))
        return lineparams

    def importPartitionSumParameters(self, yamlfile):
        """
        Import total internal partition sum polynomial coefficients from yamlfile(s)
        yamlfile = filepath or list of filepaths of total internal partition sum parameter file(s)
        Return dictionary or list of dictionaries of parameters with keys:
            Tlow, Tmid, Thigh, a, b, c, d
        """
        if isinstance(yamlfile, list):
            tipsparams = []
            for yfile in yamlfile:
                tipsparams.append(yaml.safe_load(open(Path(yfile))))
        else:
            tipsparams = yaml.safe_load(open(Path(yamlfile)))
        return tipsparams

    def createLines(self, lineparams=None, tipsparams=None):
        """
        Create Line object or list of Line objects from given parameters
        lineparams = dict of spectral line parameters
        tipsparams = dict or list of dicts of partition sum parameters
        Default to use self.lineparams and self.tipsparams
        """
        if lineparams is None:
            lineparams = self.lineparams
        if tipsparams is None:
            tipsparams = self.tipsparams
        # If single line
        if isinstance(lineparams['transitionwavenumber'], (int, float)):
            self.lines = Line(lineparams['transitionwavenumber'],
                              lineparams['lineintensity'],
                              lineparams['lowerstateenergy'],
                              [lineparams['selfbroadenedwidth'],
                               lineparams['airbroadenedwidth']],
                              lineparams['molarmass'],
                              tipsparams,
                              gamma_powers = [lineparams['temperaturedependenceofselfwidth'],
                                              lineparams['temperaturedependenceofairwidth']],
                              deltaP = lineparams['pressureshift'],
                              deltaP_power = lineparams['temperaturedependenceofpressureshift'],
                              Tref = lineparams['referencetemperature'])
        # If multiple lines
        else:
            self.lines = []
            for iline in range(len(lineparams['transitionwavenumber'])):
                self.lines.append(Line(lineparams['transitionwavenumber'][iline],
                                       lineparams['lineintensity'][iline],
                                       lineparams['lowerstateenergy'][iline],
                                       [lineparams['selfbroadenedwidth'][iline],
                                        lineparams['airbroadenedwidth'][iline]],
                                       lineparams['molarmass'][iline],
                                       tipsparams if isinstance(tipsparams, dict) else tipsparams[iline],
                                       gamma_powers = [lineparams['temperaturedependenceofselfwidth'][iline],
                                                       lineparams['temperaturedependenceofairwidth'][iline]],
                                       deltaP = lineparams['pressureshift'][iline],
                                       deltaP_power = lineparams['temperaturedependenceofpressureshift'][iline],
                                       Tref = lineparams['referencetemperature'][iline]))

    def calcTotalAbsorbance(self, v, T, P, X, L, plotflag=False):
        """
        Calculates total absorbance of all lines in spectra
        v = absolute wavenumber (cm-1)
        T = temperature (K)
        P = pressure (atm)
        X = mole fraction of line species, or list of mole fractions of broadening species including self
        L = path length (m)
        plotflag = bool
        Returns array of absorbances
        """
        if isinstance(self.lines, Line):
            absorbance = self.lines.calcAbsorbance(v, T, P, X, L)
        else:
            absorbance = 0
            for line in self.lines:
                absorbance += line.calcAbsorbance(v, T, P, X, L)
        if plotflag:
            plotData((v,absorbance),
                     linestyles = 'k-',
                     labels = 'absorbance',
                     xlabel = 'Wavenumber(cm-1)',
                     ylabel = 'Total Absorbance')
        return absorbance

class Simulation(object):
    """
    Spectral simulation object
    """

    def __init__(self, T, P, X, L, t, vrel, Iref, spectras, FSR=1, vshift=0, plotflag=False):
        """

        """
        self.T = T
        self.P = P
        self.X = X
        self.L = L
        self.t = t
        self.vrel = vrel
        self.Iref = Iref
        self.spectras = spectras
        self.FSR = FSR
        self.vshift = vshift
        self.v = None
        self.absorbance = None
        self.I = None
        self.calcI(plotflag=plotflag)
        #TODO: Ensure X is normalized if array-like. Same in wms.updateSims()

    def calcV(self):
        """

        """
        self.v = self.vrel * self.FSR + self.vshift

    def calcAbsorbance(self, plotflag=False):
        """

        """
        self.calcV()
        if isinstance(self.spectras, list):
            self.absorbance = 0
            for spectra in self.spectras:
                self.absorbance += self.spectras.calcTotalAbsorbance(self.v,
                                                                     self.T,
                                                                     self.P,
                                                                     self.X,
                                                                     self.L)
        else:
            self.absorbance = self.spectras.calcTotalAbsorbance(self.v,
                                                                self.T,
                                                                self.P,
                                                                self.X,
                                                                self.L)
        if plotflag:
            plotData((self.v, self.absorbance),
                     linestyles = 'k-',
                     labels = 'absorbance',
                     xlabel = 'Wavelength (cm-1)',
                     ylabel = 'Absorbance',
                     title = 'Simulation.calcAbsorbance: vs wavelength')
            plotData((self.t, self.absorbance),
                     linestyles = 'k-',
                     labels = 'absorbance',
                     xlabel = 'Time',
                     ylabel = 'Absorbance',
                     title = 'Simulation.calcAbsorbance: vs time')

    def calcI(self, plotflag=False):
        """

        """
        self.calcAbsorbance()
        self.I = self.Iref * np.exp(-self.absorbance)
        if plotflag:
            plotData((self.v, self.I),
                     linestyles = 'k-',
                     labels = 'transmitted',
                     xlabel = 'Wavelength (cm-1)',
                     ylabel = 'Intensity',
                     title = 'Simulation.calcI: vs wavelength')
            plotData((self.t, self.I),
                     linestyles = 'k-',
                     labels = 'transmitted',
                     xlabel = 'Time',
                     ylabel = 'Intensity',
                     title = 'Simulation.calcI: vs time')
        return self.I

    def plotVandLine(self, ind=(0,None)):
        i1, i2 = ind
        spec = self.spectras[0] if isinstance(self.spectras, list) else self.spectras
        line = spec.lines[0] if isinstance(spec.lines, list) else spec.lines
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(self.t[i1:i2], self.v[i1:i2], 'k-')
        plt.plot([self.t[i1],self.t[-1 if i2 is None else i2]], np.array([1,1])*line.calcAbsorbanceParameters(self.T,self.P,self.X,self.L)[0], 'r--')
        plt.ylim((np.min(self.v), np.max(self.v)))
        plt.xlabel('Time')
        plt.ylabel('Wavelength (cm-1)')
        plt.subplot(1,2,2)
        plt.plot(self.absorbance[i1:i2], self.v[i1:i2], 'r+')
        plt.plot([0,np.max(self.absorbance)], np.array([1,1])*line.calcAbsorbanceParameters(self.T,self.P,self.X,self.L)[0], 'r--')
        plt.ylim((np.min(self.v), np.max(self.v)))
        plt.xlabel('Absorbance')
        plt.ylabel('Wavelength (cm-1)')
