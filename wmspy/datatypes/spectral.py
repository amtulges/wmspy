from __future__ import annotations
import dataclasses
import hydra
import numpy as np
from typing import List, Tuple, Optional

from wmspy import utilities


@dataclasses.dataclass
class TipsTemperatureBounds:
    low: float
    mid: float
    high: float


@dataclasses.dataclass
class TipsCoefficients:
    a: List
    b: List
    c: List
    d: List


@dataclasses.dataclass
class TIPS:
    Tbounds: TipsTemperatureBounds
    coeffs:  TipsCoefficients
    _coeffs_attrs: List = dataclasses.field(init=False, default_factory=lambda:['a', 'b', 'c', 'd'], repr=False)

    def __call__(self, T: float) -> float:
        return self._calc_partition_sum(T)

    def _get_coeffs_index(self, T: float) -> int:
        if T > self.Tbounds.high:
            return 2
        if T > self.Tbounds.mid:
            return 1
        if T > self.Tbounds.low:
            return 0
        raise ValueError(f'Input temperature {T} K below {self.Tbounds.low=} K. ')

    def _get_coeffs(self, i: int) -> List:
        return [getattr(self.coeffs, attr)[i] for attr in self._coeffs_attrs]

    def _calc_partition_sum(self, T: float) -> float:
        c = self._get_coeffs(self._get_coeffs_index(T))
        return np.dot(c, float(T)**np.arange(len(c)))

    @classmethod
    def from_dictconfig(cls, dictconfig: hydra.core.utils.DictConfig) -> TIPS:
        classes = {}
        for key, val in dictconfig.items():
            if isinstance(val, (dict, hydra.core.utils.DictConfig)):
                val = {k:v for (k,v) in val.items()}
                classes[key] = cls.child_mapping[key](**val)
            else:
                classes[key] = val
        return cls(**classes)

    @classmethod
    def from_yaml(cls, filepath: str) -> TIPS:
        return cls.from_dictconfig(utilities.import_yaml(filepath))

    child_mapping = {
        'Tbounds' : TipsTemperatureBounds,
        'coeffs' : TipsCoefficients,
        }


@dataclasses.dataclass
class SpectralParameters:
    wavenumber: float
    intensity: float
    lowerstateenergy: float
    referencetemperature: float
    widths: List
    temperaturedependenceofwidths: Optional[List] = None
    pressureshifts: Optional[List] = None
    temperaturedependenceofpressureshifts: Optional[List] = None
    acoefficient: Optional[float] = None


@dataclasses.dataclass
class QuantumParameters:
    uppervibrationalquanta: Optional[str] = None
    lowervibrationalquanta: Optional[str] = None
    upperlocalquanta: Optional[str] = None
    lowerlocalquanta: Optional[str] = None
    errorcodes: Optional[str] = None
    referencecodes: Optional[str] = None
    flagforlinemixing: Optional[str] = None
    upperstatisticalweight: Optional[float] = None
    lowerstatisticalweight: Optional[float] = None


@dataclasses.dataclass
class SpeciesParameters:
    molarmass: float
    moleculenumber: Optional[int] = None
    isotopologuenumber: Optional[int] = None


@dataclasses.dataclass
class Line:
    spectral: SpectralParameters
    quantum: QuantumParameters
    species: SpeciesParameters

    @classmethod
    def from_dictconfig(cls, dictconfig: hydra.core.utils.DictConfig) -> Line:
        classes = {}
        for key, val in dictconfig.items():
            if isinstance(val, (dict, hydra.core.utils.DictConfig)):
                val = {k:v for (k,v) in val.items()}
                classes[key] = cls.child_mapping[key](**val)
            else:
                classes[key] = val
        return cls(**classes)

    @classmethod
    def from_yaml(cls, filepath: str) -> Line:
        return cls.from_dictconfig(utilities.import_yaml(filepath))

    child_mapping = {
        'spectral' : SpectralParameters,
        'quantum' : QuantumParameters,
        'species' : SpeciesParameters,
    }


@dataclasses.dataclass
class Spectra:
    lines: List[Line]

    @property
    def nlines(self):
        return len(self.lines)

    @property
    def wavenumber_range(self) -> Tuple:
        wavenumbers = self._get_lines_attr('wavenumber', 'spectral')
        return min(wavenumbers), max(wavenumbers)

    def _get_lines_attr(self, attr: str, child_type: str = 'spectral'):
        return [getattr(getattr(l, child_type), attr) for l in self.lines]

    @classmethod
    def from_dictconfig(cls, dictconfig: hydra.core.utils.DictConfig) -> Spectra:
        lines = []
        for i, _ in enumerate(dictconfig.spectral.wavenumber):
            linedict = {}
            for child in Line.child_mapping.keys():
                childdict = {}
                for k, v in getattr(dictconfig, child).items():
                    if v is None:
                        childdict[k] = None
                    else:
                        childdict[k] = v[i]
                linedict[child] = childdict
            lines.append(Line.from_dictconfig(linedict))
        return cls(lines)

    @classmethod
    def from_yaml(cls, filepath: str) -> Spectra:
        return cls.from_dictconfig(utilities.import_yaml(filepath))

    @classmethod
    def from_hitran(cls,
                    parfile: str,
                    molarmass: float,
                    Tref: float = 296.0,
                    temperaturedependenceofselfwidth: Optional[float] = 0.75,
                    temperaturedependenceofpressureshifts: float = 0.96,
                    minwavenumber = 0,
                    maxwavenumber = np.inf,
                    minintensity = 0,
                    ) -> Spectra:
        lines = []
        with open(parfile, 'r') as f:
            for textline in f.readlines():
                wavenumber = float(textline[3:15])
                intensity  = float(textline[15:25])
                if (wavenumber < minwavenumber) or (wavenumber > maxwavenumber) or (intensity < minintensity):
                    continue

                linedict = {child: {} for child in Line.child_mapping.keys()}

                linedict['species']['molarmass'] = molarmass
                linedict['species']['moleculenumber'] = int(textline[0:2])
                linedict['species']['isotopologuenumber'] = int(textline[2:3])

                linedict['spectral']['wavenumber'] = wavenumber
                linedict['spectral']['intensity'] = intensity
                linedict['spectral']['lowerstateenergy'] = float(textline[45:55])
                linedict['spectral']['referencetemperature'] = Tref
                linedict['spectral']['widths'] = [float(textline[40:45]), ]
                linedict['spectral']['widths'].append(float(textline[35:40]))
                nair = float(textline[55:59])
                nself = nair if temperaturedependenceofselfwidth is None else temperaturedependenceofselfwidth
                linedict['spectral']['temperaturedependenceofwidths'] = [nself, nair]
                linedict['spectral']['pressureshifts'] = [float(textline[59:67]), ]
                linedict['spectral']['temperaturedependenceofpressureshifts'] = [temperaturedependenceofpressureshifts, ]
                linedict['spectral']['acoefficient'] = float(textline[25:35])

                linedict['quantum']['uppervibrationalquanta'] = textline[67:82]
                linedict['quantum']['lowervibrationalquanta'] = textline[82:97]
                linedict['quantum']['upperlocalquanta'] = textline[97:112]
                linedict['quantum']['lowerlocalquanta'] = textline[112:127]
                linedict['quantum']['errorcodes'] = textline[127:133]
                linedict['quantum']['referencecodes'] = textline[133:145]
                linedict['quantum']['flagforlinemixing'] = textline[145:146]
                linedict['quantum']['upperstatisticalweight'] = float(textline[146:153])
                linedict['quantum']['lowerstatisticalweight'] = float(textline[153:160])

                lines.append(Line.from_dictconfig(linedict))
        return cls(lines)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='tips', node=TIPS)
cs.store(name='line', node=Line)
cs.store(name='spectra', node=Spectra)
