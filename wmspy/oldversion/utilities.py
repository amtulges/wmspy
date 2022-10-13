import pickle
import yaml
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path, PurePath
from scipy import fftpack, special

def save(data,filepath):
    """
    utilities.save(data,filepath)
    Saves data to filepath.pickle
    """
    with open(Path(filepath),'wb') as f:
        pickle.dump(data,f)

def load(filepath):
    """
    utilities.load(filepath)
    Returns contents of filepath.pickle
    """
    with open(filepath,'rb') as f:
        return pickle.load(f)

def saveFigure(figure,filepath):
    """
    utilities.saveFigure(figure,filepath)
    Saves figure to filepath
    """
    figure.savefig(fname=Path(filepath),bbox_inches='tight')

def FFT(time,data):
    """
    utilities.FFT(time,data)
    Calculate the fft
    Return freq, gain
    """
    freq = np.linspace(0.0, 1.0/(2.0*np.mean(np.diff(time))), data.size//2)
    gain = np.abs(fftpack.fft(data))[:data.size//2]*(2.0/data.size)
    return freq, gain

def gain2dB(gain):
    """
    utilities.gain2dB(gain)
    Amplitude Gain To Decibels
    """
    return 20*np.log10(gain)

def dB2gain(dB):
    """
    utilities.dB2gain(dB)
    Decibels to Amplitude Gain
    """
    return 10**(dB/20)

def Voigt(a,w):
    """
    utilities.Voigt(a,w)
    Voigt function
    """
    z  = w + 1j*a
    return np.real(special.wofz(z))

def Newton_Raphson(x, y, ytarget):
    """
    utilities.Newton_Raphson(x, y, ytarget)
    Calculates next x guess and which entry of x to replace for Newton_Raphson / Secant method
    x = 2-list or tuple
    y = 2-list or tuple
    ytarget = numeric
    Returns x_next, i_replace
    """
    x_next = (np.diff(x)/np.diff(y))[0] * (ytarget - y[-1]) + x[-1]
    i_replace = np.argmax(np.abs(x - x_next))
    return x_next, i_replace

def plotArrays(arraylist, **kwargs):
    for arr in arraylist:
        plotData(arr, **kwargs)

def plotData(data, linestyles=None,
             xlabel=None, ylabel=None, title=None,
             labels=None, legendloc='best',
             xsci=True, ysci=False,
             xlog=False, ylog=False,
             xlim=None, ylim=None,
             fig=None, ax_index=-1,
             savefile=None, display=True, close=False):
    """
    utilities.plotData()
    Plots data in a new figure.
    Parameters:
        data tuple(s) eg. (x,y) or [(x,y),(x2,y2),...],
        linestyles = str or list,
        xlabel = str, ylabel = str, title = str,
        labels = str or list,  legendloc = str (default 'best'),
        xsci = bool (default True), ysci = bool (default False),
        xlog = bool (default False), ylog = bool (default False),
        xlim = 2-tuple, ylim = 2-tuple
        savefile = str or path, display = bool (default True),
        close = bool (default True) closes figure after plotting
    Returns figure and axis pointers
    """
    if fig is None:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
    else:
        if len(fig.axes) == 0:
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[ax_index]
    ax.set_title(title) if title is not None else None
    ax.set_xlabel(xlabel) if xlabel is not None else None
    ax.set_ylabel(ylabel) if ylabel is not None else None
    if isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0)) if xsci else None
    if isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter):
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) if ysci else None
    # Handle multiple line input
    if isinstance(data,list):
        for i, xy in enumerate(data):
            # Handle linestyle
            if linestyles is None:
                style = 'k-'
            elif isinstance(linestyles,str):
                style = linestyles
            else:
                style = linestyles[i]
            # Handle label
            if labels is None:
                label = None
            elif isinstance(labels,str):
                label = labels
            else:
                label = labels[i]
            x,y = xy
            ax.plot(x, y, style, label=label)
    elif isinstance(data,tuple):
        # Handle linestyle
        if linestyles is None:
            style = 'k-'
        elif isinstance(linestyles,str):
            style = linestyles
        else:
            raise ValueError('Too many linestyles in plotData')
        # Handle label
        if labels is None:
            label = None
        elif isinstance(labels,str):
            label = labels
        else:
            raise ValueError('Too many labels in plotData')
        x,y = data
        ax.plot(x, y, style, label=label)
    elif isinstance(data,np.ndarray):
        # Handle linestyle
        if linestyles is None:
            style = '-'
        elif isinstance(linestyles,str):
            style = linestyles
        else:
            raise ValueError('Too many linestyles in plotData')
        # Handle label
        if labels is None:
            label = None
        elif isinstance(labels,str):
            label = labels
        else:
            raise ValueError('Too many labels in plotData')
        ax.plot(data, style, label=label)
    else:
        raise TypeError('First input to utilities.plotData must be a tuple or list of tuples')
    ax.legend(loc=legendloc)
    ax.set_yscale('log') if ylog else None
    ax.set_xscale('log') if xlog else None
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None
    plt.show() if display else None
    saveFigure(fig,savefile) if savefile is not None else None
    plt.close(fig) if close else None
    return fig, ax

class Configuration(object):
    """
    Base object containing configuration settings
    """

    def __init__(self, folder=None, files=None, SR=None, SR_et=None, fs=None, fm=None):
        """
        utilities.Configuration( )
        Creates configuration object
        """
        self.configfile = None
        self.folder     = PurePath(Path('.' if folder is None else folder).resolve())
        self.files      = dict()
        if files is not None:
            for filetype, file in files.items():
                if isinstance(file, str):
                    self.datafiles[filetype] = PurePath().joinpath(self.folder, file)
                else:
                    self.datafiles[filetype] = [PurePath().joinpath(self.folder, file_) for file_ in file]
        self.SR     = SR
        self.SR_et  = SR if SR_et is None else SR_et
        self.fs     = fs
        self.fm     = fm
        self.processConfig()

    def importFile(self, configfile, filetype='yaml'):
        """
        Configuration.importFile(configfile, filetype='yaml')
        Imports settings from configfile
        filetype = 'yaml' or 'tdms'
        """
        self.configfile = configfile
        if filetype.lower() == 'yaml':
            self.data = self.readYaml()
        elif filetype.lower == 'tdms':
            self.data = self.readTdms()
        # Datafile Settings
        self.folder = PurePath(Path(self.data['folder'] if 'folder' in self.data.keys() else '.').resolve())
        if 'files' in self.data.keys():
            for filetype, file in self.data['files'].items():
                if isinstance(file, str):
                    self.files[filetype] = PurePath().joinpath(self.folder, file)
                else:
                    self.files[filetype] = [PurePath().joinpath(self.folder, file_) for file_ in file]
        if 'SR' in self.data.keys():
            self.SR = float(self.data['SR'])
        if 'SR_et' in self.data.keys():
            self.SR_et = float(self.data['SR_et'])
        elif self.SR_et is None:
            self.SR_et = self.SR
        if 'fs' in self.data.keys():
            self.fs = float(self.data['fs'])
        if 'fm' in self.data.keys():
            self.fm = [float(fm) for fm in self.data['fm']]
        self.processConfig()

    def readYaml(self):
        """
        Configuration.readYaml()
        Returns dictionary of data in yaml configuration file
        """
        return yaml.safe_load(open(Path(self.configfile)))

    def readTdms(self):
        """
        Configuration.readTdms()
        Returns dictionary of data in tdms configuration file
        """
        ##TODO: WRITE SCRIPT FOR READING TDMS CONFIG SETTINGS
        return None

    def processConfig(self):
        """
        Configuration.processConfig()
        Pulls parameters out of dictionary and defines them as attributes of self
        Adds calculated parameters to attributes of self
        """
        # Scan and Modulation Settings
        if self.fs is not None:
            self.Ts = 1/self.fs
        if self.fm is not None:
            self.Tm = [1/Tm for Tm in self.fm]
        if (self.fs is not None) and (self.fm is not None):
            self.modsperscan = [fm/self.fs for fm in self.fm]
        # Sampling Settings (non-etalon measurements)
        if (self.SR is not None) and (self.fs is not None):
            self.ns = np.int(np.round(self.SR/self.fs))
        if (self.SR is not None) and (self.fm is not None):
            self.nm = [np.int(np.round(self.SR/fm)) for fm in self.fm]
        # Sampling Settings (etalon measurements)
        if (self.SR_et is not None) and (self.fs is not None):
            self.ns_et = np.int(np.round(self.SR_et/self.fs))
        if (self.SR_et is not None) and (self.fm is not None):
            self.nm_et = [np.int(np.round(self.SR_et/fm)) for fm in self.fm]

    def exportMeasurementSettings(self):
        """
        Configuration.exportMeasurementSettings()
        Returns dictionary of attributes and values for non-etalon measurement settings
        """
        members = [attr for attr in dir(self) if (not callable(getattr(self,attr))) and (not attr.startswith("__")) and (not attr.endswith('_et'))]
        members = [attr for attr in members if attr not in {'t','I','files','folder','configfile'}]
        values  = [getattr(self,attr) for attr in members]
        settings = dict()
        for m,v in zip(members,values):
            settings[m] = v
        return settings

    def exportEtalonSettings(self):
        """
        Configuration.exportEtalonSettings()
        Returns dictionary of attributes and values for etalon measurement settings
        """
        settings = self.exportMeasurementSettings()
        members = [attr for attr in dir(self) if attr.endswith('_et')]
        values = [getattr(self,attr) for attr in members]
        for m,v in zip(members,values):
            settings[m.strip('_et')] = v
        return settings

    def exportSettings(self,etalon=False):
        """
        Configuration.exportSettings(etalon=False)
        etalon = bool (True for etalon settings, false for other measurement settings)
        Returns dictionary of attributes and values for measurement settings.
        """
        if etalon:
            return self.exportEtalonSettings()
        else:
            return self.exportMeasurementSettings()
