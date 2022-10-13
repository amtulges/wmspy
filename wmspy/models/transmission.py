import lmfit
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Optional

from wmspy.models import spectral
from wmspy import utilities


class IncidentData(lmfit.Model):

    def __init__(self,
                 data: npt.ArrayLike,
                 independent_vars: List[str] = ['x'],
                 prefix: str = '',
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        self.data = data
        kwargs.update({
            'prefix': prefix,
            'nan_policy': nan_policy,
            'independent_vars': independent_vars,
            'param_names': ['shift',],
            })
        super().__init__(self.incidentdata_equation, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self) -> None:
        self.set_param_hint('shift', value=0, vary=False)

    @staticmethod
    def incidentdata_equation(x: npt.ArrayLike, shift: float = 0.0, data: npt.ArrayLike = None) -> npt.ArrayLike:
        return data + shift

    def eval(self, params: lmfit.Parameters = None, x: npt.ArrayLike = None, **kwargs) -> npt.ArrayLike:
        return super().eval(params=params, x=x, data=self.data, **kwargs)


    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> utilities.plt.Axes:
        y = self.eval(self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            title = 'IncidentModel',
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)


class TransmissionModel(lmfit.CompositeModel):

    def __init__(self,
                 incident: lmfit.Model,
                 absorbance: spectral.AbsorbanceModel,
                 nan_policy: str = 'raise',
                 **kwargs,
                 ):
        self.incident = incident
        self.absorbance = absorbance
        kwargs.update({'nan_policy': nan_policy})
        super().__init__(self.incident, self.absorbance, self.transmission_operator, **kwargs)

    @staticmethod
    def transmission_operator(incident_intensity: npt.ArrayLike, absorbance: npt.ArrayLike) -> npt.ArrayLike:
        return incident_intensity * np.exp(-absorbance)

    def make_params(self, verbose=False, **kwargs) -> lmfit.Parameters:
        params = self.absorbance.make_params()
        params.update(self.incident.make_params())
        params.update(super().make_params(verbose=verbose, **kwargs))
        param_keys = params.keys()
        for k, v in kwargs.items():
            if k in param_keys:
                params[k].value = v
            else:
                raise KeyError(f'{k} not found in params of {self}')
        params.update_constraints()
        return params

    def plot(self, x: npt.ArrayLike, settings: Optional[Dict] = None, **params) -> utilities.plt.Axes:
        y = self.eval(params=self.make_params(**params), x=x)
        s = utilities.PlotSettings(
            xlabel = 'Time',
            ylabel = 'Transmission',
            ).update(**(settings if settings else {}))
        return utilities.plot([x,], [y,], s)
