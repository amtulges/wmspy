import lmfit
import numpy as np
import numpy.typing as npt
from scipy import signal, optimize
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

from wmspy.models.utilities import make_dependent_copy
from wmspy import utilities


fsr_prefix = 'fsr_'
vshift_prefix = 'vshift_'


class InterferenceModel(lmfit.CompositeModel):
    """
    A composite model for the interference following the equation:
    (1-k)/(2*k) * (-1 + (1+k)/(1 - k*np.sin(2*np.pi * wave_array)))
    """

    def __init__(self,
                 wave: lmfit.Model,
                 k_min: float = 1E-6,
                 k_max: float = 1.0 - 1E-6,
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        self.k_min = k_min
        self.k_max = k_max
        self.wave = wave
        kwargs.update({'nan_polic': nan_policy})
        left, right = self._make_left_right()
        super().__init__(left, right, self.interference_operator, **kwargs)

    def _make_constant(self) -> lmfit.Model:
        const = lmfit.models.ConstantModel(prefix='k_')
        const.set_param_hint('k_c', min=self.k_min, max=self.k_max, value=0.1)
        return const

    def _make_left_right(self) -> Tuple[lmfit.Model]:
        return self.wave, self._make_constant()

    @staticmethod
    def interference_operator(wave_arr: npt.ArrayLike, const_arr: npt.ArrayLike) -> npt.ArrayLike:
        return 2 * ((1-const_arr)/(2*const_arr)) * (-1 + (1+const_arr)/(1 - const_arr*np.sin(2*np.pi * wave_arr))) - 1

    def guess(self, data: npt.ArrayLike, x: npt.ArrayLike, **kwargs) -> lmfit.Parameters:
        pass
        ###TODO Implement Guess

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> plt.Axes:
        y = self.eval(self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            title = 'InterferenceModel',
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)


class VertexQuadraticModel(lmfit.models.QuadraticModel):
    """ Quadratic of the form y = a*(x-b)**2 + c """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = self.vertex_quadratic_equation

    @staticmethod
    def vertex_quadratic_equation(x: npt.ArrayLike, a: float, b: float, c: float) -> npt.ArrayLike:
        return a * (x-b)**2 + c


class LinearInterferenceModel(InterferenceModel):

    def __init__(self,
                 k_min: float = 1E-6,
                 k_max: float = 1.0 - 1E-6,
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        wave = VertexQuadraticModel()
        wave.set_param_hint('a', min=0)
        wave.set_param_hint('c', min=-2*np.pi, max=2*np.pi)
        super().__init__(wave, k_min, k_max, nan_policy, **kwargs)

    def guess(self, data: npt.ArrayLike, x: npt.ArrayLike, min_prominence: float = 0.5, min_width: int = 2, n_fev = 10, **kwargs) -> lmfit.Parameters:
        peaks, props = signal.find_peaks(data, prominence=min_prominence, width=min_width)
        print(f'{len(peaks)=}')
        params = self.make_params()
        params['c'].value = 0
        params['b'].value = np.mean(x) if (len(peaks)==0) else x[peaks[np.argmax(props['widths'])]]
        params['b'].min = np.min(x)
        params['b'].max = np.max(x)
        target = utilities.arc_length(data, x)
        fit_eq = lambda a : utilities.arc_length(self.wave.func(x, a, params['b'].value, params['c'].value), x) - target
        params['a'].value = 4*target/(x[-1]-x[0])**2
        params['a'].max = 2*params['a'].value
        results = []
        a_guesses = utilities.get_points_from_distribution(
            mean = params['a'].value,
            stdev = (params['a'].max - params['a'].value)/2,
            low = params['a'].min,
            high = params['a'].max,
            n = n_fev,
            )
        for a in a_guesses:
            params['a'].value = a
            results.append(super().fit(data, params=params, x=x, method='differential_evolution', **kwargs))
        return results[np.argmin([result.redchi for result in results])].params


class EtalonModel(lmfit.CompositeModel):
    """
    A composite model for etalon transmission with interference following the equation:
    interference * (upper - lower) + lower
    """

    def __init__(self,
                 lower: lmfit.Model,
                 upper: lmfit.Model,
                 wave: lmfit.Model,
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        self.lower = lower
        self.upper = upper
        self.wave = wave
        self.interference = InterferenceModel(self.wave)
        kwargs.update({'nan_policy': nan_policy})
        left, right = self._make_left_right()
        super().__init__(left, right, lmfit.model.operator.add, **kwargs)

    def _make_left_right(self) -> Tuple[lmfit.Model]:
        lower_dependent = make_dependent_copy(self.lower, self.lower.prefix + 'dep_')
        return self.interference * (self.upper - self.lower), lower_dependent

    def guess(self) -> lmfit.Parameters:
        pass
        ###TODO Implement guess

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> plt.Axes:
        y = self.eval(self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            title = 'EtalonModel',
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)


class FsrModel(lmfit.models.ConstantModel):

    def __init__(self,
                 value: float = 1.0,
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        super().__init__(prefix=fsr_prefix, nan_policy=nan_policy, **kwargs)
        self.set_param_hint('c', min=0, value=value, vary=False)


class VShiftModel(lmfit.models.ConstantModel):

    def __init__(self,
                 value: float = 0.0,
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        super().__init__(prefix=vshift_prefix, nan_policy=nan_policy, **kwargs)
        self.set_param_hint('c', value=value, vary=True)


class WavenumberModel(lmfit.CompositeModel):

    def __init__(self,
                 wave: lmfit.Model,
                 fsr: float = 1.0,
                 vshift: float = 0.0,
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        self.wave = wave
        self.fsr = fsr
        self.vshift = vshift
        kwargs.update({'nan_policy': nan_policy})
        super().__init__(self.wave*FsrModel(value=self.fsr), VShiftModel(value=self.vshift), lmfit.model.operator.add, **kwargs)

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> plt.Axes:
        y = self.eval(params=self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            ylabel = 'Wavenumber (cm-1)',
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)
