from __future__ import annotations
import hydra
import dataclasses
import numpy as np
import numpy.typing as npt
import lmfit
from typing import Dict, Optional

from wmspy.datatypes import data
from wmspy import utilities


intensity_positions = ['upper', 'lower', 'center']


@dataclasses.dataclass
class IntensityFittingSettings:
    overwrite_guesses: bool = True    # True to overwrite manually entered guesses after first fitting

    @classmethod
    def from_dictconfig(cls, dictconfig: hydra.core.utils.DictConfig) -> IntensityFittingSettings:
        return cls(**dictconfig)

    @classmethod
    def from_yaml(cls, filepath: str) -> IntensityFittingSettings:
        return cls.from_dictconfig(utilities.import_yaml(filepath))


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='intensity_settings', node=IntensityFittingSettings)


@dataclasses.dataclass
class IntensityFitResult:
    upper: Optional[lmfit.model.ModelResult] = None
    lower: Optional[lmfit.model.ModelResult] = None
    center: Optional[lmfit.model.ModelResult] = None
    Inorm: Optional[np.ndarray] = dataclasses.field(init=False, default=None)

    def set_params(self, position: str = None, **kwargs) -> None:
        positions = intensity_positions if position is None else [position, ]
        for position in positions:
            if getattr(self, position) is None:
                continue
            for key, val in kwargs.items():
                getattr(self, position).params[key].value = val

@dataclasses.dataclass
class IntensityParamGuesses:
    upper: Dict = dataclasses.field(default_factory = lambda : {})
    lower: Dict = dataclasses.field(default_factory = lambda : {})
    center: Dict = dataclasses.field(default_factory = lambda : {})

    def set_guesses(self, position: str = None, **kwargs) -> None:
        positions = intensity_positions if position is None else [position, ]
        for position in positions:
            getattr(self, position).update(kwargs)


@dataclasses.dataclass
class IntensityFitter:
    model: Optional[lmfit.Model] = None
    fits: IntensityFitResult = dataclasses.field(default_factory = lambda : IntensityFitResult())
    guesses: IntensityParamGuesses = dataclasses.field(default_factory = lambda : IntensityParamGuesses())
    settings: IntensityFittingSettings = dataclasses.field(default_factory = lambda : IntensityFittingSettings())

    def guess(self, position: str, dat: Optional[data._BaseData] = None) -> lmfit.Parameters:
        if getattr(self.fits, position) is not None:
            output = getattr(self.fits, position).params
        elif position == 'center':
            output = self.model.guess(dat.I, x=dat.t)
        elif (position != 'center') and (self.fits.center is None):
            self.fit('center', dat)
            output = self.guess(position, dat)
        else:
            dI = dat.I - self.eval(dat.t, 'center')
            dI_range = np.max(dI) - np.min(dI)
            shift = {'upper': dI_range/2, 'lower': -dI_range/2}
            output = self.model.guess(dat.I + shift[position], x=dat.t)
        if not self.settings.overwrite_guesses:
            for key, val in getattr(self.guesses, position).items():  # Keep manually entered guesses
                output[key].value = val
        return output

    def fit(self, position: str, dat: data._BaseData, indices: Optional[np.ndarray] = None) -> None:
        if position == 'center':
            t, I = dat.t, dat.I
        else:
            t, I = dat.t[indices], dat.I[indices]
        setattr(self.fits, position, self.model.fit(I, x=t, params=self.guess(position=position, dat=dat)))

    def eval(self, t: npt.ArrayLike, position: str) -> npt.ArrayLike:
        return self.model.eval(params=getattr(self.fits, position).params, x=t)

    def _set_Inorm_from_center(self, dat: data._BaseData) -> None:
        rough_norm = -1 + (dat.I / self.eval(t=dat.t, position='center'))
        self._Inorm = rough_norm / (2*np.median(np.abs(rough_norm)))

    def _set_Inorm_from_bounds(self, dat: data._BaseData) -> None:
        Iupper = self.eval(t=dat.t, position='upper')
        Ilower = self.eval(t=dat.t, position='lower')
        self._Inorm = -1 + 2*((dat.I - Ilower) / (Iupper - Ilower))

    def set_Inorm(self, dat: data._BaseData) -> None:
        if (self.fits.upper is not None) and (self.fits.lower is not None):
            self._set_Inorm_from_bounds(dat)
        elif self.fits.center is not None:
            self._set_Inorm_from_center(dat)
        else:
            raise ValueError('No intensity fits available in self.fits to use for normalization. ')
