from nptdms import TdmsFile
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import datatypes, utilities

def read_tdms(path, channels='*', idx=None):
    '''
    Read and return just the specified channels from the tdms file.
    '''
    if channels is not None and not isinstance(channels, (list, tuple)):
        channels = [channels]

    sl = slice(None, None)
    if idx is not None:
        sl = slice(*idx)

    with TdmsFile.open(path) as tdms_file:
        group = tdms_file.groups()[0]
        fchannels = group.channels()

        # clean the channels
        fchannels = dict((f.name.strip(), f.name) for f in fchannels)

        if channels is None:
            return fchannels

        # loop over the channels
        channel_list = []
        for c in channels:
            channel_list.extend(fnmatch.filter(fchannels, c))
        channel_list = list(channel_list)

        data = []
        time = None
        labels = ['time']
        for c in channel_list:
            if time is None:
                try:
                    time = group[fchannels.get(c)].time_track()[sl]
                except KeyError:
                    pass
            data.append(group[fchannels.get(c)][sl])
            labels.append(c)
        return labels, time, data

class TDMSData(object):

    def __init__(self, data, background, LD1, LD2,
                 ch=0, ch_et=1, npoints=64000, npoints_et=640000,
                 SR=10E6, SR_et=100E6, fs=20E3, fm=[200E3, 250E3],
                 t_pretrigger=0):
        self.data = data
        self.bkg = background
        self.LD1 = LD1
        self.LD2 = LD2
        self.ch = ch
        self.ch_et = ch_et
        self.n = npoints
        self.n_et = npoints_et
        self.n_pretrigger = int(np.round(SR * t_pretrigger))
        self.n_pretrigger_et = int(np.round(SR_et * t_pretrigger))
        self.t = np.arange(npoints)/SR
        self.t_et = np.arange(npoints_et)/SR_et
        self.config = utilities.Configuration(SR=SR, SR_et=SR_et, fs=fs, fm=fm)

    def parsedata(self):
        downsample = lambda x : resample_poly(x, 1, self.config.SR_et/self.config.SR)
        nsections = self.data[0].size//self.n
        self.sections = []
        for i in range(nsections):
            i_points = np.arange(*(self.n * (i + np.arange(2))))
            i_points_et = np.arange(*(self.n_et * (i + np.arange(2))))
            i_pre = i_points[:self.n_pretrigger]
            i_points = i_points[self.n_pretrigger:]
            i_points_et = i_points_et[self.n_pretrigger_et:]
            self.sections.append(Section(self.data[self.ch][i_points] if self.data is not None else np.array([]),
                                         self.bkg[self.ch][i_points] if self.bkg is not None else np.array([]),
                                         downsample(self.LD1[self.ch][i_points_et]) if self.LD1 is not None else np.array([]),
                                         downsample(self.LD2[self.ch][i_points_et]) if self.LD2 is not None else np.array([]),
                                         self.LD1[self.ch_et][i_points_et] if self.LD1 is not None else np.array([]),
                                         self.LD2[self.ch_et][i_points_et] if self.LD1 is not None else np.array([]),
                                         self.t[self.n_pretrigger:],
                                         self.t_et[self.n_pretrigger_et:],
                                         self.config,
                                         meas_pretrigger = self.data[self.ch][i_pre] if self.data is not None else None))

class Section(object):

    def __init__(self, measurement, background, reference1, reference2, etalon1, etalon2, t, t_et, config, meas_pretrigger=None):
        self.meas = datatypes.Measurement(measurement, t=t,    config=config)
        self.bkg = datatypes.Background(background,    t=t,    config=config)
        self.io1 = datatypes.Reference(reference1,     t=t,    config=config, ind=0)
        self.io2 = datatypes.Reference(reference2,     t=t,    config=config, ind=1)
        self.et1 = datatypes.Etalon(etalon1,           t=t_et, config=config, ind=0)
        self.et2 = datatypes.Etalon(etalon2,           t=t_et, config=config, ind=1)
        self.pre = meas_pretrigger

    def plot(self, ind=None):
        i1, i2 = None, None if ind is None else ind
        plt.figure()
        plt.plot(self.meas.t[i1:i2], self.meas.I[i1:i2], 'k-')
        plt.plot(self.bkg.t[i1:i2],  self.bkg.I[i1:i2],  'r-')
        plt.plot(self.io1.t[i1:i2],  self.io1.I[i1:i2],  'g--')
        plt.plot(self.io2.t[i1:i2],  self.io2.I[i1:i2],  'b--')
        plt.plot(self.et1.t[i1:i2],  self.et1.I[i1:i2],  'g-')
        plt.plot(self.et2.t[i1:i2],  self.et2.I[i1:i2],  'b-')
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.legend(['Meas', 'Bkg', 'RefLD1', 'RefLD2', 'EtLD1', 'EtLD2'])
