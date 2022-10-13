"""
This module allows for the creation of configuration settings objects.
These objects can be created manually or imported from yaml files.

The following _ChildConfig classes hold settings of a specific type:

- `Laser` - Holds the laser settings (e.g. scan and modulation frequencies)
- `Daq` - Data acquisition settings (e.g. sample rate and depth)
- `Etalon` - Etalon acquisition settings (e.g. sample rate and depth) 
- `Filters` - Filter settings (e.g. passband, stopband, passripple, ...)
- 'Files' - Files to import 
- 'Channels' - Channels of files to import

The following _ParentConfig classes group together _ChildConfig objects:
    
- 'TotalConfig' - Contains all _ChildConfig classes
- 'EtalonConfig' - Contains _ChildConfigs necessary for etalon processing
- 'MeasurementConfig' - Contains _ChildConfigs necessary for measurement importing
- 'WMSConfig' - Contains _ChildConfigs necessary for wms processing

The FileAndChannelHandler class pairs files and channels from _ChildConfigs for importing

"""

from __future__ import annotations
import hydra
import pathlib
import dataclasses
import numpy as np
from typing import Optional, List, Tuple

from wmspy import utilities


@dataclasses.dataclass
class _ChildConfig:
    """ Base object for settings of a specific type. """
    parentconfig : Optional[_ParentConfig] = dataclasses.field(init=False, default=None, repr=False)

    @property
    def as_dict(self):
        """ 
        Returns:
            _ChildConfig as a dictionary. 
        """ 
        output = {}
        for attr in dir(self):
            if attr.startswith('_') or (attr in self.excluded_attrs):
                continue
            value = getattr(self, attr)
            if callable(value):
                continue
            output[attr] = value
        return output

    excluded_attrs = set([
        'parentconfig',
    ])

    @classmethod
    def from_dictconfig(cls, dictconfig: hydra.core.utils.DictConfig) -> _ChildConfig:
        """ Create _ChildConfig object from dictconfig. 

        Args:
            dictconfig: Settings in a dictconfig object

        Returns:
            _ChildConfig: An object holding settings
        """
        return cls(**dictconfig)

    @classmethod
    def from_yaml(cls, filepath: str) -> _ChildConfig:
        """ Create _ChildConfig object from yaml file.

        Args:
            filepath: The location of the yaml file to import

        Returns:
            _ChildConfig: An object holding settings
        """
        return cls.from_dictconfig(utilities.import_yaml(filepath))


@dataclasses.dataclass
class Laser(_ChildConfig):
    """ Holds laser settings. 
    
    Args:
        fs: Scan frequency (Hz)
        fm: Modulation frequencies (Hz)        
    """
    fs  : float
    fm  : List

    @property
    def Ts(self) -> float:
        """ Scan period (s) """
        return 1.0/self.fs

    @property
    def Tm(self) -> List:
        """ Modulation period (s) """
        return [1.0/fm_ for fm_ in self.fm]

    @property
    def mods_per_scan(self) -> List:
        """ Modulations per scan (float). """
        return [fm_/self.fs for fm_ in self.fm]


@dataclasses.dataclass
class Daq(_ChildConfig):
    """ Holds data acquisition settings. 
    
    Args:
        sr: Sample Rate (Hz)
        pre: Pretrigger points to remove
        npoints: Number of post-trigger points (used for filter design)
    """
    sr  : float
    pre : Optional[int] = 0
    npoints : Optional[int] = None

    @property
    def ns_float(self) -> float:
        """ Number of scans (float). """
        return self.sr / self.parentconfig.laser.fs

    @property
    def ns(self) -> int:
        """ Number of scans rounded to nearest int. """
        return np.int(np.round(self.ns_float))

    @property
    def nm_float(self) -> List:
        """ Number of modulations (float). """
        return [self.sr / fm_ for fm_ in self.parentconfig.laser.fm]

    @property
    def nm(self) -> List:
        """ Number of modulations rounded to nearest int. """ 
        return [np.int(np.round(nm_)) for nm_ in self.nm_float]


@dataclasses.dataclass
class Etalon(_ChildConfig):
    """ Holds etalon acquisition settings. 
    
    Args:
        sr: Sample Rate (Hz)
        fsr: Free-Spectral-Range of the etalon (cm-1)
        pre: Pretrigger points to remove
        npoints: Number of post-trigger points (used for filter design)
    """
    sr  : float
    fsr : float
    pre : Optional[int] = 0
    npoints : Optional[int] = None

    @property
    def ns_float(self) -> float:
        """ Number of scans (float). """
        return self.sr / self.parentconfig.laser.fs

    @property
    def ns(self) -> int:
        """ Number of scans rounded to nearest int. """
        return np.int(np.round(self.ns_float))

    @property
    def nm_float(self) -> List:
        """ Number of modulations (float). """
        return [self.sr / fm_ for fm_ in self.parentconfig.laser.fm]

    @property
    def nm(self) -> List:
        """ Number of modulations rounded to nearest int. """
        return [np.int(np.round(nm_)) for nm_ in self.nm_float]


@dataclasses.dataclass
class Filters(_ChildConfig):
    """ Hold the filter settings. 
    
    Args:
        passband: pass frequency (Hz)
        stopband: stop frequency (Hz)
        passripple: ripple in the passband (dB)
        stopripple: ripple in the stopband (dB)
        nharmonics: number of harmonic filters to generate
        multirate: whether to use multistage-multirate (True) or single-stage (False) filters
        fir: whether to use fir (True) or iir (False) filters
        ftype: name of filter type (e.g. 'kais', 'cheb2')
    """
    passband    : float
    stopband    : float
    passripple  : float
    stopripple  : float
    nharmonics  : int
    multirate   : bool
    fir         : bool
    ftype       : Optional[str]


@dataclasses.dataclass
class Files(_ChildConfig):
    """ Hold the files to import.
    
    Args:
        et: Etalon files
        ref: Reference files
        bkg: Background file
        meas: Measurement file
        dark: Dark file
    """
    et      : List
    ref     : List
    bkg     : str
    meas    : str
    dark    : Optional[str] = None

    def get_files(self, meas_type: str) -> List:
        """ Get files of a certain type returned as a list. 
        
        Args:
            meas_type: attribute of Files (e.g. 'et', 'ref', 'bkg', 'meas', or 'dark')
            
        Returns:
            files list
        """ 
        files = getattr(self, meas_type)
        return [files,] if isinstance(files, str) else files


@dataclasses.dataclass
class Channels(_ChildConfig):
    """ Holds the channel identifiers or indices to import.
    
    Args:
        et: Etalon channels
        ref: Reference channels
        bkg: Background channels
        meas: Measurement channels
        dark: Dark channels
    """
    et      : List
    ref     : List
    bkg     : List
    meas    : List
    dark    : Optional[List] = None

    def get_channels(self, meas_type: str) -> List:
        """ Get channels of a certain type returned as a list.
        
        Args:
            meas_type: attribute of Files (e.g. 'et', 'ref', 'bkg', 'meas', or 'dark')
            
        Returns:
            channels list            
        """
        channels = getattr(self, meas_type)
        return [channels,] if isinstance(channels, int) else channels


@dataclasses.dataclass
class _ParentConfig:
    """ Base object for collection of _ChildConfigs. """

    @classmethod
    def from_dictconfig(cls, dictconfig: hydra.core.utils.DictConfig) -> _ParentConfig:
        """ Create _ParentConfig object from dictconfig.

        Args:
            dictconfig: The dictconfig object to convert

        Returns:
            _ParentConfig: An object holding settings
        """
        children = {}
        for key, val in dictconfig.items():
            if isinstance(val, (dict, hydra.core.utils.DictConfig)):
                val = {k:v for (k,v) in val.items() if k not in cls.uninitialized_attrs}
                children[key] = cls.child_mapping[key](**val)
            else:
                children[key] = val
        parent = cls(**children)
        for child in cls.child_mapping.keys():
            getattr(parent, child).parentconfig =  parent
        return parent

    @classmethod
    def from_yaml(cls, filepath: str) -> _ParentConfig:
        """ Create _ParentConfig object from yaml file.

        Args:
            filepath: The location of the yaml file to import

        Returns:
            _ParentConfig: An object holding settings
        """
        return cls.from_dictconfig(utilities.import_yaml(filepath))

    def get_channels(self, meas_type: str) -> List:
        return self.channels.get_channels(meas_type)

    def get_files(self, meas_type: str) -> List:
        return self.files.get_files(meas_type)

    child_mapping = {
        'laser' : Laser,
        'daq' : Daq,
        'etalon' : Etalon,
        'filters' : Filters,
        'files' : Files,
        'channels' : Channels,
    }

    uninitialized_attrs = set([
        'parentconfig',
    ])


@dataclasses.dataclass
class TotalConfig(_ParentConfig):
    """ Holds all _ChildConfigs.
    
    Args:
        laser: Laser _ChildConfig
        daq: Daq _ChildConfig
        etalon: Etalon _ChildConfig
        filters: Filters _ChildConfig
        files: Files _ChildConfig
        channels: Channels _ChildConfig
        folder: location of files
    """
    laser : Laser
    daq : Daq
    etalon : Etalon
    filters : Filters
    files : Files
    channels : Channels
    folder : Optional[str] = None


@dataclasses.dataclass
class EtalonConfig(_ParentConfig):
    """ Holds _ChildConfigs necessary for etalon processing.
    
    Args:
        laser: Laser _ChildConfig
        etalon: Etalon _ChildConfig
        files: Files _ChildConfig
        channels: Channels _ChildConfig
        folder: location of files
    """
    laser : Laser
    etalon : Etalon
    files : Files
    channels : Channels
    folder : Optional[str] = None


@dataclasses.dataclass
class MeasurementConfig(_ParentConfig):
    """ Holds _ChildConfigs necessary for measurment importing.
    
    Args:
        laser: Laser _ChildConfig
        daq: Daq _ChildConfig
        files: Files _ChildConfig
        channels: Channels _ChildConfig
        folder: location of files
    """
    laser : Laser
    daq : Daq
    files : Files
    channels : Channels
    folder : Optional[str] = None


@dataclasses.dataclass
class WMSConfig(_ParentConfig):
    """ Holds _ChildConfigs necessary for wms processing.
    
    Args:
        laser: Laser _ChildConfig
        daq: Daq _ChildConfig
        filters: Filters _ChildConfig
        files: Files _ChildConfig
        channels: Channels _ChildConfig
        folder: location of files
    """
    laser : Laser
    daq : Daq
    filters : Filters
    files : Files
    channels : Channels
    folder : Optional[str] = None


@dataclasses.dataclass
class FileAndChannelHandler:
    """ Pairs files and channels from _ChildConfigs for importing
    
    Args:
        cfg: _ParentConfig object
        meas_type: attribute type to pair (e.g. 'et', 'ref', 'bkg', 'meas', or 'dark')
    """
    cfg : _ParentConfig
    meas_type : str

    def get_files(self) -> List:
        """ Get files of type self.meas_type returned as a list. 
        
        Returns:
            files list
        """ 
        return self.cfg.get_files(self.meas_type)

    def get_channels(self) -> List:
        """ Get channels of type self.meas_type returned as a list. 
        
        Returns:
            channels list
        """ 
        return self.cfg.get_channels(self.meas_type)

    def get_file_channel_pairs(self) -> List[Tuple]:
        """ Get files and channels of type self.meas_type paired as tuples. 
        
        Returns:
            list of paired tuples [(file1,channel1), (file2,channel2), ...]
        """
        files = self.get_files()
        channels = self.get_channels()
        files = self._combine_folder_and_files(files)
        files, channels = self._adjust_list_sizes(files, channels)
        return self._zip_files_and_channels_to_list(files, channels)

    def _adjust_list_sizes(self, files: List, channels: List) -> Tuple:
        """ Ensure file and channel list sizes match. """
        nf = len(files)
        nc = len(channels)
        if nf == 0:
            raise ValueError(f'No files given of type {self.meas_type}.')
        if nc == 0:
            raise ValueError(f'No channels given of type {self.meas_type}.')
        if nf == nc:
            return files, channels
        elif nf == 1:
            return files*nc, channels
        elif nc == 1:
            return files, channels*nf
        raise ValueError(f'Files and channels have different sizes which cannot be easily mapped together.  {len(files)=}, {len(channels)=}.')

    def _zip_files_and_channels_to_list(self, files: List, channels: List) -> List[Tuple]:
        output = [(f,c) for (f,c) in zip(files, channels) if (f is not None) and (c is not None)]
        if len(output) == 0:
            raise ValueError(f'Only NoneTypes given for at least one entry in each file/channel pair of type {self.meas_type}.')
        return output

    def _combine_folder_and_files(self, files: List) -> List[str]:
        if self.cfg.folder is None:
            return files
        p = pathlib.Path(self.cfg.folder)
        return [str(p.joinpath(f)) for f in files]


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='total_config', node=TotalConfig)
cs.store(name='etalon_config', node=EtalonConfig)
cs.store(name='measurement_config', node=MeasurementConfig)
cs.store(name='wms_config', node=WMSConfig)
