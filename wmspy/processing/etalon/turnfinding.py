from __future__ import annotations
import hydra
import dataclasses
import numpy as np

from wmspy.models import etalon
from wmspy.processing.etalon import peakfinding
from wmspy import utilities


@dataclasses.dataclass
class TurnFindingSettings:
    min_prominence: float = 0.5
    min_width: float = 2.0
    check_neighbors: int = 3
    max_nfev: int = 5

    @classmethod
    def from_dictconfig(cls, dictconfig: hydra.core.utils.DictConfig) -> TurnFindingSettings:
        return cls(**dictconfig)

    @classmethod
    def from_yaml(cls, filepath: str) -> TurnFindingSettings:
        return cls.from_dictconfig(utilities.import_yaml(filepath))


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='turn_settings', node=TurnFindingSettings)


@dataclasses.dataclass
class TurnFinder:
    peakresult: peakfinding.PeakFindingResult
    settings: TurnFindingSettings = dataclasses.field(default_factory = lambda : TurnFindingSettings())

    def _get_max_width_peaks(self, sep0: float, sep1: float) -> peakfinding.Peaks:
        subpeaks = self.peakresult.peaks.peaks_between_times(sep0, sep1)
        subpeaks.sort_by_attr('width')
        if len(subpeaks) > self.settings.check_neighbors:
            subpeaks = peakfinding.Peaks(subpeaks.all[-self.settings.check_neighbors:])
        subpeaks.sort_by_attr('t')
        return subpeaks

    def _find_turn_by_interference_fit(self, sep0: float, sep1: float) -> None:
        subpeaks = self._get_max_width_peaks(sep0, sep1)
        r = range(*self.peakresult.dat.nearest_indices([subpeaks.all[i].t for i in [0, -1]]))
        linear_model = etalon.LinearInterferenceModel()
        guess = linear_model.guess(
            self.peakresult.intensity.Inorm[r],
            self.peakresult.dat.t[r],
            min_prominence = self.settings.min_prominence,
            min_width = self.settings.min_width,
            n_fev = self.settings.max_nfev,
            )
        fitresult = linear_model.fit(
            self.peakresult.intensity.Inorm[r],
            x = self.peakresult.dat.t[r],
            params = guess,
            )
        i_turn = self.peakresult.peaks._get_nearest_index('t', fitresult.params['b'].value)
        self.peakresult.peaks.all[i_turn].position += '_turn'

    def find_turns(self) -> None:
        for sep0, sep1 in zip(self.peakresult.separators[:-1], self.peakresult.separators[1:]):
            self._find_turn_by_interference_fit(sep0, sep1)


# _turn_finding_methods = {
#     'fit' : _find_turn_by_interference_fit,
#     'dt' : _find_turn_by_height_and_dt,
#     }

# def _find_turn_by_height_and_dt(self, sep0: float, sep1: float) -> None:
#     subpeaks = self.peaks.peaks_between_times(sep0, sep1)
#     if subpeaks.min_height.height < self.settings.min_height:
#         i_turn = self.peaks.loc_of_index(subpeaks.min_height.index)
#     else:
#         i_turn = self._check_around_max_dt(subpeaks)
#     self.peaks.all[i_turn].position += '_turn'

# def _check_around_max_dt(self, peaks: peakdata.Peaks) -> int:
#     i_init = peaks.loc_of_index(peaks.max_dt.index)
#     i_check = np.arange(i_init - self.settings.check_neighbors, i_init + self.settings.check_neighbors + 1)
#     i_check = i_check[(i_check>=0) & (i_check<len(peaks))]
#     subpeaks = peakdata.Peaks([peaks.all[i] for i in i_check])
#     subpeaks.sort_by_attr('dt')
#     subpeaks = peakdata.Peaks(subpeaks.all[-3:])
#     subpeaks.sort_by_attr('t')
#     i_top3dt = [peaks.loc_of_index(subpeak.index) for subpeak in subpeaks.all]
#     di = np.abs(np.diff(i_top3dt))
#     ii = -3 if (di[0]==1 and di[1]!=1) else -2
#     return self.peaks.loc_of_index(peaks.all[i_top3dt[ii]].index)
