from __future__ import annotations
import dataclasses
import hydra
import pathlib
import glob
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy import fftpack
from scipy.stats import truncnorm
import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Optional


def arc_length(data: npt.ArrayLike, x: npt.ArrayLike) -> float:
    return np.sum(np.sqrt(np.diff(data)**2 + np.diff(x)**2))

def get_points_from_distribution(mean: float, stdev: float, low: float, high: float, n: int) -> np.ndarray:
    generator = truncnorm( (low-mean)/stdev, (high-mean)/stdev, loc=mean, scale=stdev )
    return generator.rvs(n)

# Fourier Transforms

def fft(time: npt.ArrayLike, data: npt.ArrayLike) -> Tuple:
    freq = np.linspace(0.0, 1.0/(2.0*np.mean(np.diff(time))), data.size//2)
    gain = np.abs(fftpack.fft(data))[:data.size//2]*(2.0/data.size)
    return freq, gain

def fft_component(data: npt.ArrayLike, x: npt.ArrayLike, frequency: float) -> Tuple[float]:
    n = len(data)
    sr = (x[-1] - x[0]) / (len(x) - 1)
    k = frequency * sr * n
    component = np.sum(data*np.exp(-2j*np.pi*k*np.arange(n)/n))
    amplitude = 2*abs(component)/n
    phase = np.mod(np.angle(component) + np.pi/2, 2*np.pi)
    return amplitude, phase


# Yaml Importing

yaml_suffix_messages = {
    '.yml' : 'File type should be ".yaml" and the suffix should be excluded when importing',
    '.yaml' : 'The suffix (".yaml") should be excluded when importing',
    }

def import_yaml(filepath: str) -> hydra.core.utils.DictConfig:
    absolutepath = pathlib.Path(filepath).absolute()
    check_exists(absolutepath)
    check_suffix(absolutepath.suffix)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(absolutepath.parent)):
        cfg = hydra.compose(config_name=absolutepath.stem)
    return cfg

def check_suffix(suffix: str) -> None:
    if suffix == '':
        return
    if suffix in yaml_suffix_messages.keys():
        raise ValueError(yaml_suffix_messages[suffix])
    else:
        raise ValueError(yaml_suffix_messages['.yml'])

def check_exists(path: pathlib.Path) -> None:
    if not glob.glob(f'{str(path)}*'):
        raise FileNotFoundError(f'{str(path)} not found.')


# Plotting

_linestyles = ['-', '--', ':', '-.']
_linecolors = ['k', 'r', 'g', 'b']
_symbols    = ['^', 'v', 'o', 's', '+']

def linestyle_cycler(n, cycle_colors=False, cycle_styles=True, color=None, style=None):
    if color is None:
        color = _linecolors[0]
    if style is None:
        style = _linestyles[0]
    colorcycler = itertools.cycle(_linecolors if cycle_colors else color)
    stylecycler = itertools.cycle(_linestyles if cycle_styles else style)
    output = []
    for _ in range(n):
        output.append(next(colorcycler) + next(stylecycler))
    return output

def symbol_cycler(n, cycle_colors=False, cycle_symbols=True, color=None, symbol=None):
    if color is None:
        color = _linecolors[0]
    if symbol is None:
        symbol = _symbols[0]
    colorcycler = itertools.cycle(_linecolors if cycle_colors else color)
    symbolcycler = itertools.cycle(_symbols if cycle_symbols else symbol)
    output = []
    for _ in range(n):
        output.append(next(colorcycler) + next(symbolcycler))
    return output


@dataclasses.dataclass
class PlotSettings:
    linestyles: List[str] = dataclasses.field(default_factory=lambda:['k-',])
    labels: List[str] = dataclasses.field(default_factory=lambda:['',])
    legendloc: str = ''
    xlabel: str = ''
    ylabel: str = ''
    title: str = ''
    xlog: bool = False
    ylog: bool = False
    xsci: bool = False
    ysci: bool = False
    show: bool = True
    xlim: Optional[Tuple[float]] = None
    ylim: Optional[Tuple[float]] = None
    ax: Optional[plt.Axes] = None

    def __post_init__(self):
        labels_set = set(self.labels)
        labels_set.remove('') if ('' in labels_set) else None
        if (self.legendloc == '') and (len(labels_set) > 0):
            self.legendloc = 'best'

    def update(self, **kwargs) -> PlotSettings:
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def to_dict(self):
        output = {}
        for attr in dir(self):
            value = getattr(self, attr)
            if (not attr.startswith('_')) and (not callable(value)):
                output[attr] = value
        return output


def _pair_lists(x, y):
    nx, ny = len(x), len(y)
    if nx == ny:
        return x, y
    if nx == 1:
        return x*ny, y
    if ny == 1:
        return x, y*nx
    raise ValueError(f'x and y must have same length. {nx=}, {ny=}')

def plot(x: List[npt.ArrayLike],
         y: List[npt.ArrayLike],
         settings: Optional[PlotSettings] = None,
         ) -> plt.Axes:
    if isinstance(x[0], (float, int)) or isinstance(y[0], (float, int)):
        raise ValueError('x and y must be lists of array-likes. ')
    if settings is None:
        settings = PlotSettings()
    ax = plt.subplots(1)[1] if settings.ax is None else settings.ax
    ax.set_title(settings.title) if settings.title else None
    ax.set_xlabel(settings.xlabel) if settings.xlabel else None
    ax.set_ylabel(settings.ylabel) if settings.ylabel else None
    if isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter):
        ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0)) if settings.xsci else None
    if isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter):
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) if settings.ysci else None
    x, y = _pair_lists(x,y)
    _, linestyles = _pair_lists(x, settings.linestyles)
    _, labels = _pair_lists(x, settings.labels)
    for xi, yi, linestyle, label in zip(x, y, linestyles, labels):
        ax.plot(xi, yi, linestyle, label=label)
    if settings.legendloc and (len(settings.labels) > 0):
        ax.legend(loc=settings.legendloc)
    ax.set_yscale('log') if settings.ylog else None
    ax.set_xscale('log') if settings.xlog else None
    ax.set_xlim(settings.xlim) if settings.xlim else None
    ax.set_ylim(settings.ylim) if settings.ylim else None
    plt.show() if settings.show else None
    return ax

def save_fig(fig: plt.Figure, file: str) -> None:
    fig.savefig(fname=file, bbox_inches='tight')

def close_fig(fig: plt.Figure) -> None:
    plt.close(fig)
