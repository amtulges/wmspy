import numpy as np
import dataclasses
import nptdms
from typing import List, Tuple, Union, Optional

from wmspy.datatypes import config
from wmspy import utilities


@dataclasses.dataclass
class _BaseData:
    """ Base class for measured data. """
    I   : np.ndarray = dataclasses.field(repr=False)
    cfg : config.ParentConfig = dataclasses.field(repr=False)

    def __post_init__(self):
        self.I = np.array(self.I[self.pre:])
        self.t = np.arange(self.npoints)/self.sr

    @property
    def npoints(self) -> int:
        return len(self.I)

    @property
    def nscans(self) -> float:
        return self.cfg.daq.laser.fs * self.timerange

    @property
    def nmodulations(self) -> List:
        return [fm_ * self.timerange for fm_ in self.cfg.daq.laser.fm]

    @property
    def timerange(self) -> float:
        return self.t[-1] - self.t[0]

    @property
    def sr(self) -> float:
        return self.cfg.etalon.sr if self.is_etalon() else self.cfg.daq.sr

    @property
    def pre(self) -> int:
        return self.cfg.etalon.pre if self.is_etalon() else self.cfg.daq.pre

    @property
    def fs(self) -> float:
        return self.cfg.laser.fs

    @property
    def fm(self) -> List:
        return self.cfg.laser.fm

    @property
    def Ts(self) -> float:
        return self.cfg.laser.Ts

    @property
    def Tm(self) -> List:
        return self.cfg.laser.Tm

    @property
    def ns(self) -> int:
        return self.cfg.etalon.ns if self.is_etalon() else self.cfg.daq.ns

    @property
    def nm(self) -> List:
        return self.cfg.etalon.nm if self.is_etalon() else self.cfg.daq.nm

    @property
    def ns_float(self) -> float:
        return self.cfg.etalon.ns_float if self.is_etalon() else self.cfg.daq.ns_float

    @property
    def nm_float(self) -> List:
        return self.cfg.etalon.nm_float if self.is_etalon() else self.cfg.daq.nm_float

    @property
    def mods_per_scan(self) -> List:
        return self.cfg.laser.mods_per_scan

    def is_etalon(self) -> bool:
        return isinstance(self, Etalon)

    def nearest_index(self, t: float) -> int:
        return np.argmin(np.abs(self.t - t))

    def nearest_indices(self, ts: List) -> np.ndarray:
        return np.array([self.nearest_index(t) for t in ts])

    def plot(self, slx: slice = slice(0,None,None), **settings) -> utilities.plt.Axes:
        s = utilities.PlotSettings(
            xlabel = 'Time',
            labels = [self.name if (hasattr(self,'name') and (self.name is not None)) else self.__class__],
            ).update(**settings)
        return utilities.plot([self.t[slx],], [self.I[slx],], s)

    file_channel_types = {
        'Etalon' : 'et',
        'Reference' : 'ref',
        'Background' : 'bkg',
        'Measurement' : 'meas',
        'Dark' : 'dark',
    }

    @classmethod
    def get_file_channel_pairs(cls, cfg: config.ParentConfig) -> List[Tuple]:
        handler = config.FileAndChannelHandler(cfg, cls.file_channel_types[cls.__name__])
        return handler.get_file_channel_pairs()

    @classmethod
    def from_tdms(cls, cfg: config.ParentConfig, group: str = None) -> List:
        classes = []
        for i, (file, channel) in enumerate(cls.get_file_channel_pairs(cfg)):
            ch_data, ch_name = read_tdms_channel(file, channel, group)
            if issubclass(cls, _IndexedData):
                klass = cls(ch_data, cfg, i, ch_name)
            else:
                klass = cls(ch_data, cfg, ch_name)
            classes.append(klass)
        return classes


@dataclasses.dataclass
class _IndexedData(_BaseData):
    """ Base class for data measured with a corresponding index. """
    index : int = 0

    @property
    def nmodulations(self) -> float:
        return super().nmodulations[self.index]

    @property
    def fm(self) -> float:
        return super().fm[self.index]

    @property
    def Tm(self) -> float:
        return super().Tm[self.index]

    @property
    def nm(self) -> int:
        return super().nm[self.index]

    @property
    def nm_float(self) -> float:
        return super().nm_float[self.index]

    @property
    def mods_per_scan(self) -> float:
        return super().mods_per_scan[self.index]


@dataclasses.dataclass
class Measurement(_BaseData):
    name : Optional[str] = None


@dataclasses.dataclass
class Background(_BaseData):
    name : Optional[str] = None


@dataclasses.dataclass
class Dark(_BaseData):
    name : Optional[str] = None

    @property
    def Imean(self) -> np.ndarray:
        return np.mean(self.I)


@dataclasses.dataclass
class Reference(_IndexedData):
    name : Optional[str] = None


@dataclasses.dataclass
class Etalon(_IndexedData):
    name : Optional[str] = None


@dataclasses.dataclass
class Importer:
    cfg : config.ParentConfig

    def __post_init__(self):
        self.fileattrs = [attr for attr in dir(self.cfg.files) if not attr.startswith('_')]

    def all_from_tdms(self, group: str = None) -> dict:
        instances = {}
        for data_type, class_type in self._get_reversed_file_channel_types().items():
            if not self.is_fileattr(data_type):
                continue
            if getattr(self.cfg.files, data_type) is None:
                continue
            instances[data_type] = globals()[class_type].from_tdms(self.cfg, group)
        return instances

    @staticmethod
    def _get_reversed_file_channel_types() -> dict:
        return {val:key for (key,val) in _BaseData.file_channel_types.items()}

    def is_fileattr(self, data_type: str) -> bool:
        return data_type in self.fileattrs


def read_tdms_channel(path: str,
                      channel: Union[str, int],
                      group: Optional[str] = None
                      ) -> Tuple[np.ndarray, str]:
    ''' Read and return just the specified channel from the tdms file. '''
    with nptdms.TdmsFile.open(path) as tdms_file:
        if group is None:
            group = tdms_file.groups()[0].name
        if isinstance(channel, str):
            channel = tdms_file[group][channel]
        elif isinstance(channel, int):
            channel = tdms_file[group].channels()[channel]
        else:
            raise TypeError(f'Channel must be either str or int. {type(channel)} given.')
        return channel[:], channel.name
