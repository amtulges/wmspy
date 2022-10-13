import dataclasses
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Dict, Optional

from wmspy.processing import filtering
from wmspy import utilities


@dataclasses.dataclass
class LockInRatios:
    measurement: filtering.WmsFilterResults
    background: Optional[filtering.WmsFilterResults] = None
    subtract_background: bool = True
    ratios: Dict = dataclasses.field(init=False, default_factory=lambda:{})
    fundamental_key: str = dataclasses.field(init=False, default='1f')

    def __post_init__(self):
        self.calc_ratios()

    @property
    def nratios(self) -> int:
        return len(self.ratios.keys())

    def _normalize_in_phase(self, harmonic_key: str) -> npt.ArrayLike:  # Calculates: (in_phase_nf/magnitude_1f)
        return self.measurement.results[harmonic_key].in_phase / self.measurement.results[self.fundamental_key].magnitude

    def _normalize_in_phase_bkg(self, harmonic_key: str) -> npt.ArrayLike:  # Calculates: (in_phase_nf_bkg/magnitude_1f_bkg)
        if self.background is None:
            return 0
        else:
            return self.background.results[harmonic_key].in_phase / self.background.results[self.fundamental_key].magnitude

    def _normalize_quadrature(self, harmonic_key: str) -> npt.ArrayLike:  # Calculates: (quadrature_nf/magnitude_1f)
        return self.measurement.results[harmonic_key].quadrature / self.measurement.results[self.fundamental_key].magnitude

    def _normalize_quadrature_bkg(self, harmonic_key: str) -> npt.ArrayLike:  # Calculates: (quadrature_nf_bkg/magnitude_1f_bkg)
        if self.background is None:
            return 0
        else:
            return self.background.results[harmonic_key].quadrature / self.background.results[self.fundamental_key].magnitude

    def calc_ratios(self):
        for harmonic_key, result in self.measurement.results.items():
            if harmonic_key == self.fundamental_key:
                continue
            self.ratios[harmonic_key + '_' + self.fundamental_key] = np.sqrt(
                (self._normalize_in_phase(harmonic_key) - self._normalize_in_phase_bkg(harmonic_key))**2 + \
                (self._normalize_quadrature(harmonic_key) - self._normalize_quadrature_bkg(harmonic_key))**2 )

    def plot(self, **kwargs):
        if 'ax' in kwargs.keys():
            axs = kwargs.pop('ax')
            if len(axs) != self.nratios:
                raise ValueError(f'Length of ax in LockInRatios.plot() must be same as {self.nratios=}. ')
        else:
            fig, axs = plt.subplots(self.nratios, 1)
        if isinstance(axs, plt.Axes):
            axs = [axs, ]
        for ax, (harmonic_key, ratio) in zip(axs, self.ratios.items()):
            s = utilities.PlotSettings(linestyles=['k-',], labels=[harmonic_key,], xlabel='Time', ax=ax).update(**kwargs)
            utilities.plot([self.measurement.results[self.fundamental_key].t_decimated, ], [ratio, ], s)
        return axs
