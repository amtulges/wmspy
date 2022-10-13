import dataclasses
import numpy as np
from copy import deepcopy
from typing import Tuple, Optional

from wmspy.datatypes import data


@dataclasses.dataclass(frozen=True)
class TimeRange:
    _tmin: float = dataclasses.field(default=0, repr=False)
    _tmax: float = dataclasses.field(default=0, repr=False)
    tend: Optional[float] = dataclasses.field(default=None, repr=False)
    tmin: float = dataclasses.field(init=False, repr=True)
    tmax: float = dataclasses.field(init=False, repr=True)

    def __post_init__(self):
        object.__setattr__(self, 'tmin', self._handle_tmin(self._tmin))
        object.__setattr__(self, 'tmax', self._handle_tmax(self._tmax))

    @property
    def dt(self) -> float:
        return self.tmax - self.tmin

    @property
    def ts(self) -> Tuple[float]:
        return self.tmin, self.tmax

    def _handle_tmin(self, tmin: float) -> float:
        if tmin is None:
            tmin = 0
        elif tmin < 0:
            tmin += self.tend
        return tmin

    def _handle_tmax(self, tmax: float) -> float:
        if tmax is None:
            tmax = self.tend
        elif tmax < 0:
            tmax += self.tend
        return tmax

    def slice_data(self, dat: data._BaseData) -> Tuple:
        imin, imax = dat.nearest_indices(self.ts)
        index = IndexRange(imin, imax)
        dat_slice = index.slice_data(dat)
        return dat_slice, index


@dataclasses.dataclass(frozen=True)
class IndexRange:
    _imin: int = dataclasses.field(default=0, repr=False)
    _imax: int = dataclasses.field(default=0, repr=False)
    iend: Optional[int] = dataclasses.field(default=None, repr=False)
    imin: int = dataclasses.field(init=False, repr=True)
    imax: int = dataclasses.field(init=False, repr=True)

    def __post_init__(self):
        object.__setattr__(self, 'imin', self._handle_imin(self._imin))
        object.__setattr__(self, 'imax', self._handle_imax(self._imax))

    def __call__(self) -> slice:
        return self.slx

    def __len__(self) -> int:
        return self.imax - self.imin

    @property
    def slx(self) -> slice:
        return slice(self.imin, self.imax+1)

    @property
    def points(self) -> np.ndarray:
        return np.arange(self.imin, self.imax+1)

    @property
    def ind(self) -> Tuple[int]:
        return self.imin, self.imax

    def _handle_imin(self, imin: int) -> int:
        if imin is None:
            imin = 0
        elif imin < 0:
            imin += self.iend
        return imin

    def _handle_imax(self, imax: int) -> int:
        if imax is None:
            imax = self.iend-1
        elif imax < 0:
            imax += self.iend
        return imax

    def slice_data(self, dat: data._BaseData) -> data._BaseData:
        dat_copy = deepcopy(dat)
        dat_copy.t = dat.t[self.slx]
        dat_copy.I = dat.I[self.slx]
        return dat_copy
