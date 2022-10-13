import dataclasses
from typing import Tuple, List

from wmspy.datatypes import data
from wmspy.processing.etalon import peakfinding, intensityfitting
from wmspy import utilities


@dataclasses.dataclass
class HalfModulationSegment:
    dat: data._BaseData
    peaks: peakfinding.Peaks
    intensity: intensityfitting.IntensityFitResult
    center: float

    @property
    def turns(self) -> Tuple:
        return self.peaks[0], self.peaks[-1]

    @property
    def turn_times(self) -> List:
        return [peak.t for peak in self.turns]

    def plot_intensity(self, dat, **settings) -> utilities.plt.Axes:
        x, y = []
        for i, position in enumerate(intensitydata.intensity_positions):
            if getattr(self.intensity.fits, position) is not None:
                if (position == 'center') and (self.intensity.fits.upper is not None) and (self.intensity.fits.lower is not None):
                    continue
                x.append(dat.t)
                y.append(self.intensity.eval(dat.t, position))
                linestyles.append(utilities.linestyle_cycler(i+1, color='r')[-1])
                labels.append(position + ' intensity')

    def plot(self, intensity: bool = False, **settings) -> utilities.plt.Axes:
        x, y, linestyles, labels = [], [], [], []
        if intensity:

        for i, position in enumerate(peakdata.peak_positions):
            if getattr(self.peaks, position).all:
                indices = getattr(self.peaks, position).indices
                x.append(self.dat.t[indices])
                y.append(self.dat.I[indices])
                linestyles.append(utilities.symbol_cycler(i+1, cycle_colors=True)[-1])
                labels.append(position + ' peaks')
        s = utilities.PlotSettings(
            xlabel = 'Time',
            linestyles = linestyles,
            labels = labels,
            ).update(**settings)
        return utilities.plot(x, y, s)


class EtalonSegments(list):
