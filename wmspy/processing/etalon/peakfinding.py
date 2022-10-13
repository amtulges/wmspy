from __future__ import annotations
import hydra
import dataclasses
import numpy as np
import numpy.typing as npt
from scipy import signal
import lmfit
import functools
from typing import List, Tuple, Optional

from wmspy.datatypes import data
from wmspy.processing import indexing
from wmspy.processing.etalon import intensityfitting
from wmspy import utilities


peak_positions = ['upper', 'lower', 'upper_turn', 'lower_turn']


@dataclasses.dataclass
class PeakFindingSettings:
    min_prominence: float = 0.5
    min_width: float = 2.0
    min_height: float = 1.5
    min_npeaks: int = 3
    max_nfev: int = 5

    @classmethod
    def from_dictconfig(cls, dictconfig: hydra.core.utils.DictConfig) -> PeakFindingSettings:
        return cls(**dictconfig)

    @classmethod
    def from_yaml(cls, filepath: str) -> PeakFindingSettings:
        return cls.from_dictconfig(utilities.import_yaml(filepath))


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='peak_settings', node=PeakFindingSettings)


@dataclasses.dataclass
class Peak:
    index: int
    position: str
    t: float
    I: float
    prominence: float
    left_base: float
    right_base: float
    width: float
    width_height: float
    left_ip: float
    right_ip: float
    dt: float = np.nan
    height: float = np.nan


class Peaks(list):

    def __post_init__(self):
        self.sort_by_attr(attr='index')

    def _get_attrs(self, attr: str) -> np.ndarray:
        return np.array([getattr(peak, attr) for peak in self])

    def _get_min_peak(self, attr: str) -> Peak:
        return self[np.nanargmin(self._get_attrs(attr))]

    def _get_max_peak(self, attr: str) -> Peak:
        return self[np.nanargmax(self._get_attrs(attr))]

    def _get_loc_of_nearest_peak(self, attr: str, value) -> int:
        return np.nanargmin(np.abs(self._get_attrs(attr) - value))

    def _get_loc_of_matching_peaks(self, attr: str, value) -> np.ndarray:
        return np.where(self._get_attrs(attr) == value)[0]

    def _get_loc_of_successive_positions(self, position) -> List:
        indices = np.where(np.diff(self._get_loc_of_matching_peaks(attr='position', value=position)) == 1)[0]
        pos_peaks = self._get_matching_peaks(attr='position', value=position)
        return [self.loc_of_index(pos_peaks[i].index) for i in indices]

    def _get_matching_peaks(self, attr: str, value) -> Peaks:
        return Peaks([self[i] for i in np.where(self._get_attrs(attr) == value)[0]])

    def _get_peaks_in_range(self, lowerbound: float, upperbound: float, attr: str) -> Peaks:
        attrs = self._get_attrs(attr=attr)
        return Peaks([self[i] for i in np.where((lowerbound < attrs) & (attrs < upperbound))[0]])

    def sort_by_attr(self, attr: str) -> None:
        self.sort(key = lambda x : getattr(x, attr))

    @property
    def upper(self) -> Peaks:
        return self._get_matching_peaks(attr='position', value='upper')

    @property
    def lower(self) -> Peaks:
        return self._get_matching_peaks(attr='position', value='lower')

    @property
    def upper_turn(self) -> Peaks:
        return self._get_matching_peaks(attr='position', value='upper_turn')

    @property
    def lower_turn(self) -> Peaks:
        return self._get_matching_peaks(attr='position', value='lower_turn')

    def turns(self) -> Peaks:
        turns = self.upper_turn.extend(self.lower_turn)
        turns.sort_by_attr('index')
        return turns

    @property
    def indices(self) -> np.ndarray:
        return self._get_attrs(attr='index')

    @property
    def times(self) -> np.ndarray:
        return self._get_attrs(attr='t')

    @property
    def positions(self) -> np.ndarray:
        return self._get_attrs(attr='position')

    @property
    def min_height(self) -> Peak:
        return self._get_min_peak(attr='height')

    @property
    def max_dt(self) -> Peak:
        return self._get_max_peak(attr='dt')

    @property
    def max_width(self) -> Peak:
        return self._get_max_peak(attr='width')

    def peaks_between_times(self, t0: float, t1: float) -> Peaks:
        return self._get_peaks_in_range(t0, t1, attr='t')

    def loc_of_index(self, index: int) -> int:
        return self._get_loc_of_matching_peaks(attr='index', value=index)[0]

    def set_heights(self, Inorm: npt.ArrayLike) -> None:
        for peak in self:
            if (peak.position == 'upper') or (peak.position == 'upper_turn'):
                peak.height = Inorm[peak.index] - -1
            else:
                peak.height = 1 - Inorm[peak.index]

    def set_dts(self):
        upper_dts = np.diff(self.upper.times[1:]) + np.diff(self.upper.times[:-1])
        lower_dts = np.diff(self.lower.times[1:]) + np.diff(self.lower.times[:-1])
        iter_upper = iter(upper_dts)
        iter_lower = iter(lower_dts)
        for peak in self[2:-2]:
            peak.dt = next(iter_upper) if peak.position == 'upper' else next(iter_lower)


@dataclasses.dataclass
class PeakFinder:
    _dat: data._BaseData
    intensity: intensityfitting.IntensityFitter
    slicer: Optional[indexing.TimeRange] = None
    peaks: Peaks = dataclasses.field(default_factory = lambda : Peaks())
    settings: PeakFindingSettings = dataclasses.field(default_factory = lambda : PeakFindingSettings())

    @property
    def dat(self) -> data._BaseData:
        if self.slicer is None:
            return self._dat
        else:
            return self._get_dat_slice(self.slicer)

    @functools.lru_cache(maxsize=2, typed=False)
    def _get_dat_slice(self, slicer) -> npt.ArrayLike:
        return slicer.slice_data(self._dat)[0]

    @property
    def turn_period(self) -> float:
        return self._dat.Tm/2

    @property
    def offset(self) -> float:
        return self._dat.Tm/4

    @property
    def _initial_turn(self) -> Peak:

        if self.peaks.min_height.height < self.settings.min_height:
            return self.peaks.min_height
        else:
            return self.peaks.max_dt

    @property
    def _first_separator(self) -> float:
        return self.peaks.all[0].t + np.mod(self._initial_turn.t + self.offset - self.peaks.all[0].t, self.turn_period)

    @property
    def _last_separator(self) -> float:
        return self.peaks.all[-1].t - np.mod(self.peaks.all[-1].t - self._initial_turn.t + self.offset, self.turn_period)

    @property
    def _is_turn_before_first_separator(self) -> bool:
        return np.mod(self._initial_turn.t - self.dat.t[0], self.turn_period) < (self._first_separator - self.dat.t[0])

    @property
    def _is_turn_after_last_separator(self) -> bool:
        return np.mod(self.dat.t[-1] - self._initial_turn.t, self.turn_period) < (self.dat.t[-1] - self._last_separator)

    @property
    def _is_min_npeaks_before_first_separator(self) -> bool:
        return len(self.peaks.peaks_between_times(self.dat.t[0], self._first_separator)) > self.settings.min_npeaks

    @property
    def _is_min_npeaks_after_last_separator(self) -> bool:
        return len(self.peaks.peaks_between_times(self._last_separator, self.dat.t[-1])) > self.settings.min_npeaks

    @property
    def separators(self) -> np.ndarray:
        output = np.arange(self._first_separator, self._last_separator + self.offset, self.turn_period)
        if self._is_turn_before_first_separator and self._is_min_npeaks_before_first_separator:
            output = np.concatenate(([self.dat.t[0],], output)) # include first point as additional separator
        else:
            output[0] = self.dat.t[0] # move first separatorto first point
        if self._is_turn_after_last_separator and self._is_min_npeaks_after_last_separator:
            output = np.concatenate((output, [self.dat.t[-1],]))
        else:
            output[-1] = self.dat.t[-1]
        return output

    def _next_slice_times(self, forward: bool = True) -> Tuple:
        if forward:
            return (self._last_separator - self.offset/2,
                    2*self._last_separator - self._first_separator + self.offset/2)
        else:
            return (2*self._first_separator - self._last_separator - self.offset/2,
                    self._first_separator + self.offset/2)

    def move_to_next_slice(self, forward=True) -> None:
        next_times = self._next_slice_times(forward=forward)
        if forward and (next_times[-1] > self._dat.t[-1]):
            return
        if not forward and (next_times[0] < self._dat.t[0]):
            return
        else:
            self.slicer = indexing.TimeRange(*next_times)


    def _fit_position_intensity(self, position: str) -> None:
        indices = np.arange(len(self.dat.t)) if (position == 'center') else getattr(self.peaks, position).indices
        self.intensity.fit(position=position, dat=self.dat, indices=indices)

    def init_fit_params(self, position: str, **kwargs) -> None:
        self._fit_position_intensity(position)
        self.intensity.set_fit_params(position, **kwargs)

    def fit_intensity(self) -> None:
        if self.peaks.all:
            for position in ['upper', 'lower']:
                self._fit_position_intensity(position)
        else:
            self._fit_position_intensity('center')

    def _find_position_peaks(self, position: str) -> None:
        indices, properties = signal.find_peaks(
            self.intensity.Inorm * (-1 if position == 'lower' else 1),
            prominence = self.settings.min_prominence,
            width = self.settings.min_width,
            )
        peaks = []
        for i, index in enumerate(indices):
            peaks.append(
                Peak(
                    index = index,
                    position = position,
                    t = self.dat.t[index],
                    I = self.dat.I[index],
                    prominence   = properties['prominences'][i],
                    left_base    = properties['left_bases'][i],
                    right_base   = properties['right_bases'][i],
                    width        = properties['widths'][i],
                    width_height = properties['width_heights'][i],
                    left_ip      = properties['left_ips'][i],
                    right_ip     = properties['right_ips'][i],
                )
            )
        self.peaks.all.extend(peaks)

    def _filter_successive_positions(self) -> None:
        indices = set()
        for position in ['upper', 'lower']:
            indices = indices.union(self.peaks._get_loc_of_successive_positions(position))
        for index in reversed(list(indices)):
            if self.peaks.all[index].width < self.peaks.all[index + 1].width:
                self.peaks.all.pop(index)
            else:
                self.peaks.all.pop(index + 1)


    def find_peaks(self) -> None:
        self.peaks.clear()
        self._find_position_peaks('upper')
        self._find_position_peaks('lower')
        self.peaks.sort_by_attr(attr='index')
        self.peaks.set_heights(self.intensity.Inorm)
        self.peaks.set_dts()
        self._filter_successive_positions()

    def eval(self) -> None:
        for ifev in range(self.settings.max_nfev):
            prev_indices = self.peaks.indices
            prev_positions = self.peaks.positions
            self.fit_intensity()
            self.intensity.set_Inorm(self.dat)
            self.find_peaks()
            if np.array_equal(prev_indices, self.peaks.indices) and np.array_equal(prev_positions, self.peaks.positions):
                break
        if ifev == self.settings.max_nfev - 1:
            raise RuntimeWarning('Max iterations reached in peakfinder. ')
        return PeakFindingResult(self.dat, self.peaks, self.separators, self.intensity, self.settings)
