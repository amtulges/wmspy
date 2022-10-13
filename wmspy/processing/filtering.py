from __future__ import annotations
import dataclasses
import numpy as np
import numpy.typing as npt
import scipy
import matplotlib.pyplot as plt
import copy
from typing import List, Tuple, Dict, Optional

from wmspy.datatypes import config
from wmspy import utilities


fir_order_functions = {
    'kaiser' : scipy.signal.kaiserord,
    }

fir_order_args = ['ripple', 'width']

iir_order_functions = {
    'butter' : scipy.signal.buttord,
    'cheby1' : scipy.signal.cheb1ord,
    'cheby2' : scipy.signal.cheb2ord,
    'ellip'  : scipy.signal.ellipord,
    }

iir_order_args = ['wp', 'ws', 'gpass', 'gstop']

def gain2dB(gain):
    return 20*np.log10(gain)

def dB2gain(dB):
    return 10**(dB/20)


@dataclasses.dataclass
class _BaseLockIn:
    passband: float
    stopband: float
    samplerate: float
    center: float = 0.0
    ftype: str = None
    passripple: float = 0.1
    stopripple: float = 40.0
    _nyquist: float = None
    npoints: Optional[int] = None
    decimation: int = dataclasses.field(init=False, default=1)
    nstages: int = dataclasses.field(init=False, default=1)
    downsample: int = dataclasses.field(init=False, default=1)
    order: int = dataclasses.field(init=False, default=0)
    ntaps: int = dataclasses.field(init=False, default=1)
    a: npt.ArrayLike = dataclasses.field(init=False, default=1)
    b: npt.ArrayLike = dataclasses.field(init=False, default=1)

    @property
    def nyquist(self) -> float:
        return self.samplerate/2 if self._nyquist is None else self._nyquist

    @property
    def oversample(self) -> float:
        return (self.samplerate/2) / self.stopband

    @property
    def wp(self) -> float:
        return self.passband/self.nyquist

    @property
    def ws(self) -> float:
        return self.stopband/self.nyquist

    @property
    def gpass(self) -> float:
        return self.passripple

    @property
    def gstop(self) -> float:
        return self.stopripple

    @property
    def ripple(self) -> float:
        return self.stopripple

    @property
    def width(self) -> float:
        return self.ws - self.wp

    @property
    def nremaining(self) -> Optional[int]:
        if self.npoints is None:
            return None
        return (self.npoints/self.downsample) - 2*np.ceil(self.order/self.decimation)

    def apply(self, time: npt.ArrayLike, data: npt.ArrayLike) -> LockInResult:
        X = np.cos( 2*np.pi * self.center * time) * data
        Y = np.sin( 2*np.pi * self.center * time) * data
        for _ in range(self.nstages):
            X = scipy.signal.filtfilt(self.b, self.a, X)
            Y = scipy.signal.filtfilt(self.b, self.a, Y)
            X = X[::self.decimation]
            Y = Y[::self.decimation]
        nanlength = int(np.ceil(self.order/self.decimation))
        X[:nanlength]  = np.NaN
        X[-nanlength:] = np.NaN
        Y[:nanlength]  = np.NaN
        Y[-nanlength:] = np.NaN
        return LockInResult(
            t_original = time,
            raw_data = data,
            t_decimated = time[::self.decimation**self.nstages],
            magnitude = np.sqrt(X**2 + Y**2),
            phase = np.arctan(np.divide(Y, X)),
            in_phase = X,
            quadrature = Y,
            lockin = self,
            )

    def plot(self, f_stop=None, f_start=0, num=1000, dB=True, xlog=False, ylog=False, **kwargs) -> plt.Axes:
        if f_stop is None:
            f_stop = np.min([10*self.stopband, self.samplerate/2])
        freq, gain = self.calc_gain(f_stop=f_stop, f_start=f_start, num=num, dB=dB)
        if self.center != 0:
            freq = self.center + np.concatenate((-freq[::-1], freq))
            gain = np.concatenate((gain[::-1], gain))
        kwargs.update({'xlog':xlog, 'ylog':ylog})
        s = utilities.PlotSettings(
            xlabel = 'Frequency (Hz)',
            ylabel = 'Gain (dB)' if dB else 'Gain',
            ).update(**kwargs)
        return utilities.plot([freq,], [gain,], s)


@dataclasses.dataclass
class IirLockIn(_BaseLockIn):
    wn: float = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        if self.ftype is None:
            self.ftype = 'cheby2'
        self.calc_order()
        self.calc_coeffs()

    @property
    def order_args(self) -> Dict:
        return {arg:getattr(self, arg) for arg in iir_order_args}

    def calc_order(self) -> None:
        self.order, self.wn = iir_order_functions[self.ftype](**self.order_args)
        self.ntaps = self.order + 1

    def calc_coeffs(self) -> None:
        self.b, self.a = scipy.signal.iirfilter(
            self.order,
            self.wn,
            rp=self.gpass,
            rs=self.gstop,
            btype='lowpass',
            analog=False,
            output='ba',
            ftype=self.ftype,
            )

    def calc_gain(self, f_stop=None, f_start=0, num=1000, dB=False) -> Tuple:
        freq = np.linspace(f_start, (self.samplerate/2) if f_stop is None else f_stop, num=num)
        gain = 1
        for stage in range(self.nstages):
            upsample = self.decimation**stage
            stage_b = np.zeros(self.order*upsample+1)
            stage_b[::upsample] = self.b
            stage_a = np.zeros(self.order*upsample+1)
            stage_a[::upsample] = self.a
            _, h = scipy.signal.freqz(
                stage_b,
                a = stage_a,
                worN = np.divide(freq, self.samplerate/2) * np.pi)
            gain *= np.absolute(h)
        if dB:
            gain = gain2dB(gain)
        return freq, gain


@dataclasses.dataclass
class FirLockIn(_BaseLockIn):
    beta: float = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        if self.ftype is None:
            self.ftype = 'kaiser'
        self.calc_order()
        self.calc_coeffs()

    @property
    def order_args(self) -> Dict:
        return {arg:getattr(self, arg) for arg in fir_order_args}

    def calc_order(self) -> None:
        taps, self.beta = fir_order_functions[self.ftype](**self.order_args)
        self.ntaps = taps if (np.mod(taps,2)==1) else taps+1
        self.order = self.ntaps - 1

    def calc_coeffs(self) -> None:
        self.a = 1.0
        self.b = scipy.signal.firwin(
            self.ntaps,
            np.mean([self.wp, self.ws]),
            window = (self.ftype, self.beta),
            )

    def calc_gain(self, f_stop=None, f_start=0, num=1000, dB=False) -> Tuple:
        freq = np.linspace(f_start, (self.samplerate/2) if f_stop is None else f_stop, num=num)
        gain = 1
        for stage in range(self.nstages):
            upsample = self.decimation**stage
            stage_b = np.zeros(self.order*upsample+1)
            stage_b[::upsample] = self.b
            _, h = scipy.signal.freqz(
                stage_b,
                a = 1,
                worN = np.divide(freq, self.samplerate/2) * np.pi)
            gain *= np.absolute(h)
        if dB:
            gain = gain2dB(gain)
        return freq, gain


@dataclasses.dataclass
class MultiRateIirLockIn(IirLockIn):
    decimation: int = 2
    totalpassripple: float = 0.1
    totalstopripple: float = 40.0

    def __post_init__(self):
        self._nyquist = (self.samplerate/2)/(self.decimation**(self.nstages-1))
        self.passripple = self.totalpassripple/self.nstages
        self.stopripple = -gain2dB(1-(1-dB2gain(-self.totalstopripple))**self.nstages)
        super().__post_init__()

    @property
    def nstages(self) -> int:
        return int(np.floor(np.log(self.oversample)/np.log(self.decimation)))

    @property
    def downsample(self) -> int:
        return self.decimation**self.nstages


@dataclasses.dataclass
class MultiRateFirLockIn(FirLockIn):
    decimation: int = 2
    totalpassripple: float = 0.1
    totalstopripple: float = 40.0

    def __post_init__(self):
        self._nyquist = (self.samplerate/2)/(self.decimation**(self.nstages-1))
        self.passripple = self.totalpassripple/self.nstages
        self.stopripple = -gain2dB(1-(1-dB2gain(-self.totalstopripple))**self.nstages)
        super().__post_init__()

    @property
    def nstages(self) -> int:
        return int(np.floor(np.log(self.oversample)/np.log(self.decimation)))

    @property
    def downsample(self) -> int:
        return self.decimation**self.nstages


class LockInFactory:

    def __init__(self,
                 passband: float,
                 stopband: float,
                 samplerate: float,
                 npoints: int = None,
                 multirate: bool = False,
                 fir: bool = True,
                 **kwargs,
                 ):
        self.passband = passband
        self.stopband = stopband
        self.samplerate = samplerate
        self.npoints = npoints
        self.multirate = multirate
        self.fir = fir
        self.kwargs = kwargs
        self.kwargs.update({
            'passband' : passband,
            'stopband' : stopband,
            'samplerate' : samplerate,
            'npoints' : npoints,
            })

    @property
    def oversample(self) -> float:
        return (self.samplerate/2) / self.stopband

    @property
    def can_make_multirate(self) -> bool:
        return self.multirate and (self.oversample > 4)

    @property
    def max_decimation(self) -> int:
        return int(np.round(np.sqrt(self.oversample)))

    @property
    def is_decimation_specified(self) -> bool:
        return 'decimation' in self.kwargs

    def get_best_multirate(self, filter_type: MultiRateFirLockIn | MultiRateIirLockIn) -> MultiRateFirLockIn | MultiRateIirLockIn:
        if self.is_decimation_specified:
            return filter_type(**self.kwargs)
        filters = []
        for dec in np.arange(2, self.max_decimation):
            self.kwargs.update({'decimation':dec})
            filters.append(filter_type(**self.kwargs))
        if self.npoints is None:
            return filters[np.argmin([f.downsample for f in filters])]
        else:
            return filters[np.argmax([f.nremaining for f in filters])]

    def update(self, **kwargs) -> LockInFactory:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.kwargs.update({k:v})
        return self

    def create(self) -> IirLockIn | FirLockIn | MultiRateFirLockIn | MultiRateIirLockIn:
        if self.can_make_multirate:
            return self.get_best_multirate(MultiRateFirLockIn if self.fir else MultiRateIirLockIn)
        else:
            return FirLockIn(**self.kwargs) if self.fir else  IirLockIn(**self.kwargs)


@dataclasses.dataclass
class WmsFilters:
    fm: float
    filters: Dict   # lock-in filters keyed by harmonic str e.g. '1f', '2f', etc.

    @property
    def nharmonics(self) -> int:
        return len(self.filters.keys())

    @property
    def centers(self) -> npt.ArrayLike:
        return self.fm * (np.arange(self.nharmonics) + 1)

    def apply(self, time: npt.ArrayLike, data: npt.ArrayLike) -> Dict[LockInResult]:
        output = {}
        for harmonic, filt in self.filters.items():
            output[harmonic] = filt.apply(time, data)
        return WmsFilterResults(output, self)

    def plot(self, color='r', **kwargs) -> plt.Axes:
        if 'ax' not in kwargs.keys():
            fig, ax = plt.subplots(1)
            kwargs['ax'] = ax
        else:
            ax = kwargs['ax']
        if 'linestyles' in kwargs.keys():
            linestyles = kwargs.pop('linestyles')
            if len(linestyles) != self.nharmonics:
                raise ValueError(f'{len(linestyles)=} must be the same as {self.nharmonics=}')
        else:
            linestyles = utilities.linestyle_cycler(self.nharmonics, color=color)
        if 'labels' in kwargs.keys():
            labels = kwargs.pop('labels')
            if len(labels) != self.nharmonics:
                raise ValueError(f'{len(labels)=} must be the same as {self.nharmonics=}')
        else:
            labels = list(self.filters.keys())
        if 'legendloc' not in kwargs.keys():
            kwargs['legendloc'] = 'best'
        for i, (_, filt) in enumerate(self.filters.items()):
            kwargs.update({
                'linestyles' : [linestyles[i], ],
                'labels' : [labels[i], ],
                })
            filt.plot(**kwargs)
        return ax

    @classmethod
    def from_config(cls, cfg: config.ParentConfig, **kwargs) -> List[WmsFilters]:
        params = cfg.filters.to_dict()
        params.update({
            'samplerate' : cfg.daq.sr,
            'npoints' : cfg.daq.npoints,
            })
        params.update(kwargs)
        nharmonics = params.pop('nharmonics')
        factory = LockInFactory(**params)
        output = []
        for fm in cfg.laser.fm:
            filters = {}
            factory.update(center = fm)
            filt = factory.create()
            for harmonic in (np.arange(nharmonics) + 1):
                filters[f'{harmonic}f'] = copy.deepcopy(filt)
                filters[f'{harmonic}f'].center = harmonic * fm
            output.append(cls(fm, filters))
        return output


@dataclasses.dataclass
class LockInResult:
    t_original: npt.ArrayLike
    raw_data: npt.ArrayLike
    t_decimated: npt.ArrayLike
    magnitude: npt.ArrayLike
    phase: npt.ArrayLike
    in_phase: npt.ArrayLike
    quadrature: npt.ArrayLike
    lockin: IirLockIn | FirLockIn | MultiRateFirLockIn | MultiRateIirLockIn

    def nearest_index(self, t: float) -> int:
        return np.argmin(np.abs(self.t_original - t))

    def nearest_indices(self, ts: List) -> np.ndarray:
        return np.array([self.nearest_indices(t for t in ts)])

    def nearest_index_decimated(self, t: float) -> int:
        return np.argmin(np.abs(self.t_decimated - t))

    def nearest_indices_decimated(self, ts: List) -> np.ndarray:
        return np.array([self.nearest_indices_decimated(t for t in ts)])

    def plot(self, overlay_raw_data: bool = True, **kwargs) -> List[plt.Axes]:
        if 'ax' in kwargs.keys():
            axs = kwargs.pop('ax')
            if len(axs) != 4:
                raise ValueError('Length of ax in LockInResult.plot() must be 4. ')
        else:
            fig, axs = plt.subplots(4, 1)
        if overlay_raw_data:
            s = utilities.PlotSettings(linestyles=['ko',], labels=['raw data',], xlabel='Time', ax=axs[0]).update(**kwargs)
            utilities.plot([self.t_original, ], [self.raw_data, ], s)
        for ax, attr in zip(axs, ['magnitude', 'phase', 'in_phase', 'quadrature']):
            s = utilities.PlotSettings(linestyles=['ro',], labels=[attr,], xlabel='Time', ax=ax).update(**kwargs)
            utilities.plot([self.t_decimated, ], [getattr(self, attr), ], s)
        return axs

    def plot_fft(self, **kwargs) -> plt.Axes:
        f_original, gain_original = utilities.fft(self.t_original, self.raw_data)
        mask = np.isfinite(self.magnitude)
        f_filtered, gain_filtered = utilities.fft(self.t_decimated[mask], self.magnitude[mask])
        s = utilities.PlotSettings(linestyles=['k--','r-'],
                                   labels=['raw','filtered'],
                                   xlabel='Frequency (Hz)',
                                   ylabel='Gain',
                                   ylog=True,
                                   ).update(**kwargs)
        return utilities.plot([f_original, f_filtered], [gain_original, gain_filtered], s)


@dataclasses.dataclass
class WmsFilterResults:
    results: Dict[LockInResult]     # keyed by harmonic str e.g. '1f', '2f', etc.
    filters: WmsFilters

    @property
    def nharmonics(self) -> int:
        return len(self.results.keys())
