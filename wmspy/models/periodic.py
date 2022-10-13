import re
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import lmfit
from typing import List, Tuple, Dict, Optional

from wmspy.models.utilities import sum_args
from wmspy import utilities


class SawtoothModel(lmfit.Model):
    """ A sawtooth model with three Parameters: `amplitude`, `frequency`, and `shift`. """

    def __init__(self,
                 independent_vars: List[str] = ['x'],
                 prefix: str = '',
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy, 'independent_vars': independent_vars})
        super().__init__(self.sawtooth_equation, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self) -> None:
        self.set_param_hint('amplitude', min=0)
        self.set_param_hint('frequency', min=0)
        self.set_param_hint('shift', min=-lmfit.models.tau-1.e-5, max=lmfit.models.tau+1.e-5)

    @staticmethod
    def sawtooth_equation(x: npt.ArrayLike, amplitude: float = 1.0, frequency: float = 1.0, shift: float = 0.0) -> npt.ArrayLike:
        return (-2*amplitude/np.pi) * np.arctan(1/np.tan((np.pi*frequency)*x + shift))

    def guess(self, data: npt.ArrayLike, x: npt.ArrayLike, **kwargs) -> lmfit.Parameters:
        pass
        ###TODO Implement Guess

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> plt.Axes:
        y = self.eval(self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            title = 'SawtoothModel',
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)


class TriangleModel(lmfit.Model):
    """ A triangle model with three Parameters: `amplitude`, `frequency`, and `shift`. """

    def __init__(self,
                 independent_vars: List[str] = ['x'],
                 prefix: str = '',
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy, 'independent_vars': independent_vars})
        super().__init__(self.triangle_equation, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self) -> None:
        self.set_param_hint('amplitude', min=0)
        self.set_param_hint('frequency', min=0)
        self.set_param_hint('shift', min=-lmfit.models.tau-1.e-5, max=lmfit.models.tau+1.e-5)

    @staticmethod
    def triangle_equation(x: npt.ArrayLike, amplitude: float = 1.0, frequency: float = 1.0, shift: float = 0.0) -> npt.ArrayLike:
        return (2*amplitude/np.pi) * np.arcsin(np.sin(2*np.pi*frequency*x + shift))

    def guess(self, data: npt.ArrayLike, x: npt.ArrayLike, **kwargs) -> lmfit.Parameters:
        pass
        ###TODO Implement Guess

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> plt.Axes:
        y = self.eval(self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            title = 'TriangleModel',
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)


class SinusoidModel(lmfit.models.SineModel):
    """ A linear frequency version of lmfit.models.SineModel """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.func = self.sine_equation

    @staticmethod
    def sine_equation(x, amplitude, frequency, shift):
        return amplitude * np.sin(2*np.pi*frequency*x + shift)


class SuperpositionOfHarmonicsModel(lmfit.CompositeModel):
    """ A composite model of a superposition of submodels which have 'amplitude', 'frequency', and 'shift' parameters. """

    freq_str = 'frequency'
    amp_str = 'amplitude'
    shift_str = 'shift'
    const_prefix = 'const_'
    ratio_prefix = 'ratio_'
    delta = 1E-16

    def __init__(self,
                 frequency: float,
                 nharmonics: int,
                 prefix: str = '',
                 submodel: lmfit.Model = SinusoidModel,
                 const: bool = False,
                 amplitude_bound: Optional[str] = None, # 'first', 'previous', or None
                 nan_policy: str ='raise',
                 **kwargs):
        self.frequency = frequency
        self.nharmonics = nharmonics
        self.submodel = submodel
        self._pre = prefix
        self.const = const
        self.amplitude_bound = amplitude_bound
        kwargs.update({'nan_policy': nan_policy})
        left, right = self._make_left_right()
        super().__init__(left, right, lmfit.model.operator.add, **kwargs)
        self._set_param_hints()

    @property
    def prefix(self) -> str:
        return self._pre

    @prefix.setter
    def prefix(self, value: str) -> None:
        oldprefix = self._pre
        for term in self._terms:
            term.prefix = term.prefix.replace(oldprefix, value)
        self._pre = value
        self._set_param_hints()
        self._param_names = []
        self._parse_params()

    @property
    def harmonics(self) -> npt.ArrayLike:
        return 1 + np.arange(self.nharmonics)

    @property
    def frequencies(self) -> npt.ArrayLike:
        return self.frequency * self.harmonics

    def _harmonic_prefix(self, harmonic: int) -> str:
        return f'{self.prefix}h{harmonic}_'

    def _constant_prefix(self) -> str:
        return f'{self.prefix}{self.const_prefix}'

    def _get_term_ref_ratio_strs(self, harmonic: int) -> Tuple[str]:
        term_str = f'{self._harmonic_prefix(harmonic)}{self.amp_str}'
        h_ref = self._get_reference_amplitude_harmonic(harmonic)
        ref_str = f'{self._harmonic_prefix(h_ref)}{self.amp_str}'
        ratio_str = f'{self.ratio_prefix}{term_str}_{ref_str}'
        return term_str, ref_str, ratio_str

    def _make_term(self, harmonic: int) -> lmfit.Model:
        term_prefix = self._harmonic_prefix(harmonic)
        term = self.submodel(prefix = term_prefix)
        term.set_param_hint(f'{self.freq_str}', vary=False, value=self.frequency*harmonic)
        #term = self._set_amplitude_hints(harmonic, term)
        return term

    def _make_constant(self) -> lmfit.Model:
        term_prefix = self._constant_prefix()
        return lmfit.models.ConstantModel(prefix = term_prefix)

    def _make_left_right(self) -> Tuple[lmfit.Model]:
        self._terms = [self._make_term(h) for h in self.harmonics]
        if self.const:
            self._terms.append(self._make_constant())
        if (len(self._terms) == 1) and (not self.const):
            self._terms.append(self._make_constant())
            self._terms[-1].set_param_hint(self._constant_prefix() + 'c', vary=False, value=0)
        return sum_args(*self._terms[:-1]), self._terms[-1]

    def _get_reference_amplitude_harmonic(self, harmonic: int) -> int:
        if self.amplitude_bound == 'first':
            return 1
        if self.amplitude_bound == 'previous':
            return harmonic - 1
        raise ValueError('Unrecognized amplitude_bound kwarg in SuperpositionOfHarmonicsModel.  Options are "first", "previous", or None. ')

    def _set_param_hints(self) -> None:
        self.param_hints = {}
        for term in self._terms:
            term_prefix = term.prefix
            for basename, hint in term.param_hints.items():
                self.param_hints[f'{term_prefix}{basename}'] = hint
        if self.amplitude_bound is None:
            return
        for harmonic in self.harmonics:
            term_str, ref_str, ratio_str = self._get_term_ref_ratio_strs(harmonic)
            if harmonic == 1:
                self.param_hints[term_str] = {'vary':True, 'min':self.delta}
            else:
                self.param_hints[term_str] = {
                    'vary' : False,
                    'min' : 0 if (self.amplitude_bound == 'first') else self.delta,
                    'expr' : f'{ratio_str}*{ref_str}',
                    }
                self.param_hints[ratio_str] = {
                    'vary' : True,
                    'min' : 0 if (self.amplitude_bound == 'first') else self.delta,
                    'max' : 1,
                    'value' : 0.01,
                    }

    def make_params(self, verbose: bool = False, **kwargs) -> lmfit.Parameters:
        params = super().make_params(verbose=verbose, **kwargs)
        for k, v in kwargs.items():
            if k in params.keys():
                params[k].value = v
            else:
                raise KeyError(f'{k} not found in params of {self}')
        params.update_constraints()
        return params

    def guess(self, data: npt.ArrayLike, x: npt.ArrayLike, **kwargs) -> lmfit.Parameters:
        params = self.make_params()
        freq_keys = [key for key in params.keys() if key.endswith(self.freq_str)]
        amplitude_estimates = {}
        max_resolvable_frequency = 1/(2*np.mean(np.diff(x)))
        min_resolvable_frequency = max_resolvable_frequency/(len(x)//2 - 1)
        for freq_key in freq_keys:
            frequency = params[freq_key].value
            prefix = freq_key.replace(self.freq_str, '')
            if (min_resolvable_frequency <= frequency) and (frequency <= max_resolvable_frequency):
                amplitude, shift = utilities.fft_component(data, x, frequency=frequency)
                shift = np.mod(shift + 2*np.pi*frequency*x[0], 2*np.pi)
            else:
                amplitude = 0 if self.amplitude_bound is None else self.delta
                shift = 0
            params[prefix + self.amp_str].value = amplitude
            params[prefix + self.shift_str].value = shift
            amplitude_estimates[prefix + self.amp_str] = amplitude
        ratio_keys = [key for key in params.keys() if key.startswith(self.ratio_prefix)]
        for ratio_key in ratio_keys:
            numerator_key, denominator_key = re.findall('|'.join(amplitude_estimates.keys()), ratio_key)
            params[ratio_key].value = amplitude_estimates[numerator_key] / amplitude_estimates[denominator_key]
        const_keys = [key for key in params.keys() if self.const_prefix in key]
        for const_key in const_keys:
            if params[const_key].vary:
                params[const_key].value = np.mean(data)
        params.update_constraints()
        return params

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> plt.Axes:
        y = self.eval(self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            title = 'SuperpositionOfHarmonicsModel',
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)


class MulitpleFrequenciesModel(lmfit.CompositeModel):
    """ A composite model of SuperpositionOfHarmnonicsModels with multiple frequencies. """

    const_prefix = 'const_'

    def __init__(self,
                 frequencies: npt.ArrayLike,
                 nharmonics: Optional[npt.ArrayLike] = None,
                 prefix: str = '',
                 subprefixes: Optional[List[str]] = None,
                 submodels: Optional[List[lmfit.Model]] = None,
                 const: bool = False,
                 amplitude_bound: Optional[str] = None,
                 nan_policy: str = 'raise',
                 **kwargs):
        nf = len(frequencies)
        self.frequencies = frequencies
        self.nharmonics = np.ones(nf) if nharmonics is None else nharmonics
        self._pre = prefix
        self.subprefixes = [f'f{i}_' for i in np.arange(nf)] if subprefixes is None else subprefixes
        self.submodels = [SinusoidModel]*nf if submodels is None else submodels
        self.const = const
        self.amplitude_bound = amplitude_bound
        kwargs.update({'nan_policy': nan_policy})
        left, right = self._make_left_right()
        super().__init__(left, right, lmfit.model.operator.add, **kwargs)
        self._set_param_hints()

    @property
    def prefix(self) -> str:
        return self._pre

    @prefix.setter
    def prefix(self, value: str) -> None:
        oldprefix = self._pre
        for term in self._terms:
            term.prefix = term.prefix.replace(oldprefix, value)
        self._pre = value
        self._set_param_hints()
        self._param_names = []
        self._parse_params()

    def _make_term(self, frequency: float, nharmonics: int, subprefix: str, submodel: lmfit.Model) -> lmfit.Model:
        return SuperpositionOfHarmonicsModel(
            frequency = frequency,
            nharmonics = nharmonics,
            prefix = self.prefix + subprefix,
            submodel = submodel,
            const = False,
            amplitude_bound = self.amplitude_bound,
            )

    def _make_constant(self) -> lmfit.Model:
        return lmfit.models.ConstantModel(prefix=self.const_prefix)

    def _make_left_right(self) -> Tuple[lmfit.Model]:
        self._terms = []
        for i, (freq, nh, subprefix, submodel) in enumerate(zip(self.frequencies, self.nharmonics, self.subprefixes, self.submodels)):
            self._terms.append(self._make_term(freq, nh, subprefix, submodel))
        if self.const:
            self._terms.append(self._make_constant())
        if (len(self._terms) == 1) and (not self.const):
            self._terms.append(self._make_constant())
            self._terms[-1].set_param_hint(self.const_prefix + 'c', vary=False, value=0)
        return sum_args(*self._terms[:-1]), self._terms[-1]

    def _set_param_hints(self) -> None:
        self.param_hints = {}
        for term in self._terms:
            for basename, hint in term.param_hints.items():
                self.param_hints[f'{basename}'] = hint

    def make_params(self, verbose: bool = False, **kwargs) -> lmfit.Parameters:
        params = self._terms[0].make_params(verbose=verbose, **kwargs)
        for term in self._terms[1:]:
            params.update(term.make_params(verbose=verbose, **kwargs))
        for k, v in kwargs.items():
            if k in params.keys():
                params[k].value = v
            else:
                raise KeyError(f'{k} not found in params of {self}')
        params.update_constraints()
        return params

    def guess(self, data: npt.ArrayLike, x: npt.ArrayLike, **kwargs) -> lmfit.Parameters:
        params = self._terms[0].guess(data=data, x=x, **kwargs)
        for term in self._terms[1:]:
            params.update(term.guess(data=data, x=x, **kwargs))
        params.update_constraints()
        return params

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> plt.Axes:
        y = self.eval(self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            title = 'MultipleFrequenciesModel',
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)
