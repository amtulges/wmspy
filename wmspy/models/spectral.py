import lmfit
from scipy import constants
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

from wmspy.datatypes import spectral
from wmspy.models import etalon
from wmspy.models.utilities import sum_args
from wmspy import utilities


c2 = constants.physical_constants['second radiation constant'][0] * 100 # K/cm-1
NA = constants.N_A # unitless
P0 = constants.physical_constants['standard atmosphere'][0] # Pa/atm
R  = constants.R # J/(mol*K)


def normalized_mole_fractions(X: float, X_remainder: List[float]) -> List:
    if len(X_remainder) == 0:
        return [1,]
    s = sum(X_remainder)
    if s == 0:
        return [1,] + [0,]*len(X_remainder)
    return [X,] + [xr*(1-X)/s for xr in X_remainder]

def calc_pressure_shifted_wavenumber(T: float, P: float, X: float, linedata: spectral.Line, X_remainder: List[float]) -> float:
    n = len(linedata.spectral.pressureshifts)
    if (n != 1) and (len(X_remainder) != n-1):
        raise ValueError(f'When {n=} pressure shift coefficients are given, must have {n-1=} elements in X_remainder.')
    if n == 1:
        X_all = [1,]
    else:
        X_all = normalized_mole_fractions(X, X_remainder)
    output = linedata.spectral.wavenumber
    for Xi, dP, nP in zip(X_all, linedata.spectral.pressureshifts, linedata.spectral.temperaturedependenceofpressureshifts):
        output += Xi*P*dP*(linedata.spectral.referencetemperature/T)**nP
    return output

def calc_line_strength(T: float, P: float, X: float, L: float, linedata: spectral.Line, tips: spectral.TIPS) -> float:
    return P*X*L*NA*P0*linedata.spectral.intensity/(R*T*100**2) * \
           (tips(linedata.spectral.referencetemperature) / tips(T)) * \
           np.exp(-c2 * linedata.spectral.lowerstateenergy * (1/T - 1/linedata.spectral.referencetemperature)) * \
           (1 - np.exp(-c2 * linedata.spectral.wavenumber/T)) / (1 - np.exp(-c2 * linedata.spectral.wavenumber/linedata.spectral.referencetemperature))

def calc_doppler_width(T, linedata, pressure_shifted_wavenumber) -> float:
    return 7.1623E-7 * pressure_shifted_wavenumber * np.sqrt(T/linedata.species.molarmass)

def calc_lorentzian_width(T: float, P: float, X: float, linedata: spectral.Line, X_remainder: List[float]) -> float:
    output = 0
    X_all = normalized_mole_fractions(X, X_remainder)
    for Xi, gamma, n in zip(X_all, linedata.spectral.widths, linedata.spectral.temperaturedependenceofwidths):
        output += Xi*P*2*gamma*(linedata.spectral.referencetemperature/T)**n
    return output

def calc_wavenumber_array(x: npt.ArrayLike, fsr_c: float, vshift_c: float, wavenumber: etalon.WavenumberModel, extraparams: lmfit.Parameters) -> npt.ArrayLike:
    wavenumber_params = wavenumber.make_params()
    wavenumber_params.update(extraparams)
    wavenumber_params[etalon.fsr_prefix+'c'].value = fsr_c
    wavenumber_params[etalon.vshift_prefix+'c'].value = vshift_c
    wavenumber_params.update_constraints()
    return wavenumber.eval(params=wavenumber_params, x=x)

def calc_absorbance_array(v: npt.ArrayLike, T: float, P: float, X: float, L: float, spectra: spectral.Spectra, extraparams: lmfit.Parameters) -> npt.ArrayLike:
    spectra_params = spectra.make_params()
    spectra_params.update(extraparams)
    spectra_params['T'].value = T
    spectra_params['P'].value = P
    spectra_params['X'].value = X
    spectra_params['L'].value = L
    spectra_params.update_constraints()
    return spectra.eval(params=spectra_params, x=v)


class VoigtLineModel(lmfit.models.VoigtModel):

    def __init__(self,
                 linedata: spectral.Line,
                 tips: spectral.TIPS,
                 X_remainder: Optional[List] = None,
                 prefix: str = '',
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        self.linedata = linedata
        self.tips = tips
        self.X_remainder = [1,] if X_remainder is None else X_remainder
        self.pre = prefix
        super().__init__(prefix=prefix, nan_policy=nan_policy, **kwargs)
        self.set_param_hint('T', expr='T', vary=False, min=0)
        self.set_param_hint('P', expr='P', vary=False, min=0)
        self.set_param_hint('X', expr='X', vary=False, min=0, max=1)
        self.set_param_hint('L', expr='L', vary=False, min=0)
        self.set_param_hint('pressure_shifted_wavenumber', vary=False, min=0, expr=f'calc_pressure_shifted_wavenumber(T,P,X,{prefix}linedata,X_remainder)')
        self.set_param_hint('line_strength', vary=False, min=0, expr=f'calc_line_strength(T,P,X,L,{prefix}linedata,tips)')
        self.set_param_hint('doppler_width', vary=False, min=0, expr=f'calc_doppler_width(T,{prefix}linedata,{prefix}pressure_shifted_wavenumber)')
        self.set_param_hint('lorentzian_width', vary=False, min=0, expr=f'calc_lorentzian_width(T,P,X,{prefix}linedata,X_remainder)')
        self.set_param_hint('amplitude', vary=False, min=0, expr=f'{prefix}line_strength')
        self.set_param_hint('center', vary=False, min=0, expr=f'{prefix}pressure_shifted_wavenumber')
        self.set_param_hint('sigma', vary=False, min=0, expr=f'{prefix}doppler_width/2.35482')
        self.set_param_hint('gamma', vary=False, min=0, expr=f'{prefix}lorentzian_width/2')

    def make_params(self, verbose: bool = False, **kwargs) -> lmfit.Parameters:
        params = super().make_params(verbose=verbose, **kwargs)
        params._asteval.symtable[f'{self.pre}linedata'] = self.linedata
        params._asteval.symtable['tips'] = self.tips
        params._asteval.symtable['X_remainder'] = self.X_remainder
        params._asteval.symtable['calc_pressure_shifted_wavenumber'] = calc_pressure_shifted_wavenumber
        params._asteval.symtable['calc_line_strength'] = calc_line_strength
        params._asteval.symtable['calc_doppler_width'] = calc_doppler_width
        params._asteval.symtable['calc_lorentzian_width'] = calc_lorentzian_width
        params.add('T', value=300, vary=True, min=0)
        params.add('P', value=1.0, vary=False, min=0)
        params.add('X', value=0.1, vary=True, min=0, max=1)
        params.add('L', value=1.0, vary=False, min=0)
        params.update_constraints()
        return params

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> plt.Axes:
        y = self.eval(self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Wavenumber (cm-1)',
            ylabel = 'Absorbance',
            labels = [f'{self.linedata.spectral.wavenumber} cm-1',],
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)


class SpectraModel(lmfit.CompositeModel):

    def __init__(self,
                 spectradata: spectral.Spectra,
                 tips: spectral.TIPS,
                 model_types: Optional[List[lmfit.Model]] = None,
                 X_remainder: Optional[List] = None,
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        self.spectradata = spectradata
        self.tips = tips
        if model_types is None:
            self.model_types = [VoigtLineModel,]*spectradata.nlines
        elif isinstance(model_types, lmfit.Model):
            self.model_types = [model_types,]*spectradata.nlines
        else:
            if len(model_types) != spectradata.nlines:
                raise ValueError(f'model_types and spectradata.lines must have same length. {len(model_types)=}, {spectradata.nlines=} ')
            self.model_types = model_types
        self.X_remainder = [1,] if X_remainder is None else X_remainder
        kwargs.update({'nan_policy': nan_policy})
        left, right = self._make_left_right()
        super().__init__(left, right, lmfit.model.operator.add, **kwargs)
        self.set_param_hint('T', value=300.0, vary=False, min=0)
        self.set_param_hint('P', value=1.0, vary=False, min=0)
        self.set_param_hint('X', value=0.1, vary=False, min=0, max=1)
        self.set_param_hint('L', value=1.0, vary=False, min=0)

    def _make_models(self) -> lmfit.Model:
        models = []
        for i, (linedata, model_type) in enumerate(zip(self.spectradata.lines, self.model_types)):
            models.append(model_type(linedata, self.tips, X_remainder=self.X_remainder, prefix=self._line_prefix(i)))
        return models

    def _make_left_right(self) -> Tuple[lmfit.Model]:
        models = self._make_models()
        return models[0], sum_args(*models[1:])

    def _line_prefix(self, i: int) -> str:
        return f'line{i}_'

    def make_params(self, verbose: bool = False, **kwargs) -> lmfit.Parameters:
        params = super().make_params(verbose=verbose, **kwargs)
        for i, linedata in enumerate(self.spectradata.lines):
            params._asteval.symtable[self._line_prefix(i)+'linedata'] = linedata
        params._asteval.symtable['tips'] = self.tips
        params._asteval.symtable['X_remainder'] = self.X_remainder
        params._asteval.symtable['calc_pressure_shifted_wavenumber'] = calc_pressure_shifted_wavenumber
        params._asteval.symtable['calc_line_strength'] = calc_line_strength
        params._asteval.symtable['calc_doppler_width'] = calc_doppler_width
        params._asteval.symtable['calc_lorentzian_width'] = calc_lorentzian_width
        params.add('T', value=300, vary=True, min=0)
        params.add('P', value=1.0, vary=False, min=0)
        params.add('X', value=0.1, vary=True, min=0, max=1)
        params.add('L', value=1.0, vary=False, min=0)
        param_keys = params.keys()
        for k, v in kwargs.items():
            if k in param_keys:
                params[k].value = v
            else:
                raise KeyError(f'{k} not found in params of {self}')
        params.update_constraints()
        return params

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> plt.Axes:
        y = self.eval(self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Wavenumber (cm-1)',
            ylabel = 'Absorbance',
            labels = ['total absorbance',],
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)

    def plot_survey(self, settings: Optional[Dict] = None) -> plt.Axes:
        x = [[line.spectral.wavenumber,]*2 for line in self.spectradata.lines]
        y = [[0,line.spectral.intensity] for line in self.spectradata.lines]
        x0, x1 = x[0][0], x[-1][0]
        dx = x1-x0
        s = utilities.PlotSettings(
            xlabel = 'Wavenumber (cm-1)',
            ylabel = 'Intensity (cm/molecule)',
            linestyles = ['k-',],
            legendloc = '',
            ylog = True,
            xlim = (x0-0.05*dx, x1+0.05*dx),
            ).update(**(settings if settings else {}))
        return utilities.plot(x, y, s)


class AbsorbanceModel(lmfit.Model):

    def __init__(self,
                 wavenumber: etalon.WavenumberModel,
                 spectra: spectral.Spectra,
                 independent_vars: List[str] = ['x'],
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        self.wavenumber = wavenumber
        self.spectra = spectra
        kwargs.update({
            'nan_policy': nan_policy,
            'independent_vars': independent_vars,
            'param_names': ['T', 'P', 'X', 'L', etalon.fsr_prefix+'c', etalon.vshift_prefix+'c'],
            })
        super().__init__(self.absorbance_equation, **kwargs)
        self._set_paramhints_prefix()

    @staticmethod
    def absorbance_equation(x: npt.ArrayLike, T: float, P: float, X: float, L: float, fsr_c: float, vshift_c: float, wavenumber: etalon.WavenumberModel, spectra: spectral.Spectra, extraparams: lmfit.Parameter) -> npt.ArrayLike:
        v = calc_wavenumber_array(x, fsr_c, vshift_c, wavenumber, extraparams)
        return calc_absorbance_array(v, T, P, X, L, spectra, extraparams)

    def _set_paramhints_prefix(self) -> None:
        self.set_param_hint('T', **self.spectra.param_hints['T'])
        self.set_param_hint('P', **self.spectra.param_hints['P'])
        self.set_param_hint('X', **self.spectra.param_hints['X'])
        self.set_param_hint('L', **self.spectra.param_hints['L'])
        self.set_param_hint('fsr_c', **self.wavenumber.param_hints['fsr_c'])
        self.set_param_hint('vshift_c', **self.wavenumber.param_hints['vshift_c'])

    def make_params(self, verbose=False, **kwargs) -> lmfit.Parameters:
        params = self.wavenumber.make_params()
        params.update(self.spectra.make_params())
        params.update(super().make_params(verbose=verbose, **kwargs))
        param_keys = params.keys()
        for k, v in kwargs.items():
            if k in param_keys:
                params[k].value = v
            else:
                raise KeyError(f'{k} not found in params of {self}')
        params.update_constraints()
        return params

    def eval(self, params: lmfit.Parameters = None, x: npt.ArrayLike = None, **kwargs) -> npt.ArrayLike:
        return super().eval(params=params, x=x, wavenumber=self.wavenumber, spectra=self.spectra, extraparams=params, **kwargs)

    def plot(self, params: lmfit.Parameters, x: npt.ArrayLike, settings: Optional[Dict] = None, **kwargs) -> plt.Axes:
        p = self.make_params()
        p.update(params)
        y = self.eval(params=p, x=x, **kwargs)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            ylabel = 'Absorbance'
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)

    def plot_wavenumber(self, params: lmfit.Parameters, x: npt.ArrayLike, settings: Optional[Dict] = None, **kwargs) -> npt.ArrayLike:
        wavenumber_params = self.wavenumber.make_params()
        wavenumber_params.update(params)
        v = calc_wavenumber_array(x, wavenumber_params['fsr_c'].value, wavenumber_params['vshift_c'].value, self.wavenumber, params)
        p = self.make_params()
        p.update(params)
        y = self.eval(params=p, x=x, **kwargs)
        fig, ax = plt.subplots(2,2)
        s = utilities.PlotSettings(xlabel='Absorbance', ylabel='Wavenumber', ax=ax[0][0]).update(**(settings if settings else {}))
        utilities.plot([y,], [v,], s)
        s = utilities.PlotSettings(xlabel='Time', ylabel='Wavenumber', ax=ax[0][1]).update(**(settings if settings else {}))
        utilities.plot([x,], [v,], s)
        s = utilities.PlotSettings(xlabel='Time', ylabel='Absorbance', ax=ax[1][1]).update(**(settings if settings else {}))
        utilities.plot([x,], [y,], s)
        return ax
