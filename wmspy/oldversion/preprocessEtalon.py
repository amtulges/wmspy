import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, bisect, newton, differential_evolution
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy.integrate import simps

from datatypes import Etalon
from utilities import plotData

def assignMethodsToDatatype(dtype):
    """
    assignMethodsToDatatype(dtype)
    Assigns functions from preprocessEtalon to methods of dtype, including:
        findPeaksAndTurns
        findFirstTurn
        removeTurnPeaks
        findTurnPeaks
        findAllPeakIndices
        peaksBetweenTurns
        countPeaks
        fitI
        IEquation
        fitV
        vEquation
        fitK
        interferenceEquation
        fitInterferenceEquation
    Also creates attributes in dtype, including:
        iturn1,     int of 1st turn index
        turnlist,   list of sets of turn indices
        peaklist,   list of 2-lists of arrays of peak indices [[ipos, ineg()], [,], ...]
        tpeakcount, array of peak times
        peakcount,  array of cumulative peak count
    """
    dtype.turnlist = []
    dtype.peaklist = []
    dtype.iturn1 = int()
    dtype.tpeakcount = np.array([])
    dtype.peakcount  = np.array([])
    equations = [findPeaksAndTurns,
                 findFirstTurn,
                 removeTurnPeaks,
                 findTurnPeaks,
                 findAllPeakIndices,
                 filterPeakPairsByWidth,
                 peaksBetweenTurns,
                 countPeaks,
                 cumulativeCount,
                 calcVrelFromFitModelResults,
                 plotFitModelResults,
                 tryModelVCombinations,
                 fitModel,
                 fitModelSegment,
                 fitI,
                 IEquation,
                 fitV,
                 vEquation,
                 dvdt,
                 dv2dt2,
                 modelEquation,
                 fitK,
                 interferenceEquation,
                 fitInterferenceEquation,
                 fitScaledData,
                 find_dvdt_zero,
                 handle_I_GuessAndBounds,
                 handle_v_GuessAndBounds,
                 handle_ind,
                 handle_necessary,
                 handle_points,
                 checkBounds,
                 ]
    for equation in equations:
        setattr(dtype, equation.__name__, equation)

def countPeaks(self, ind=(0,None), prominence=0.5, min_width=None, removepairs=False, min_height=0.5, neighbors=3, firstturn=None, lastturn=None, manualturns=None, Iconst=True, Imods=(True,True), Iscans=(True,True), printupdates=True, plotflag=False, plotflag_peaks=False, plotflag_turns=False):
    """
    countPeaks(ind=(0,None), prominence=0.5, min_width=None, min_height=0.5, neighbors=3, firstturn=None, lastturn=None, manualturns=None, Imods=(True,True), Iscans=(True,True), printupdates=True, plotflag=False, plotflag_peaks=False, plotflag_turns=False)
    Find valid peaks in range ind and create a cumulative count
    ind = (i_min, i_max) global index bounds to search, default is all data
    prominence = float 0:2, min prominence in "normalized" data for peak finding.  Default 0.5
    min_width  = int, minimum width of acceptable peak. Smaller width peaks will be filtered out. Default None
    removepairs = bool, whether to remove peak following one below min_width. Defalut False
    min_height = float in range (0:2), minimum height of turn peak for obvious filtering
    neighbors = int, number of neighboring peaks to check around suspicious turn peak
    firstturn = int, first turn,
    lastturn = int, last turn,
    manualturns = array of global indices of manually found turn peaks to use instead of search algorithm
    Iconst = bool
    Imods  = (bool, bool) to use (1st, 2nd) modulation harmonics in fitting I
    Iscans = (bool, bool) to use (1st, 2nd) scan harmonics in fitting I
    plotflag = bool, plots counting results
    plotflag_peaks = bool, plots peak finding in segments
    plotflag_turns = bool, plots turn finding in segments
    Returns dictionary with keys:
        iturn1,
        turnlist,
        peaklist,
        tpeakcount,
        peakcount,
    """
    # Find relevant points
    i_min, i_max = ind
    if firstturn is None:
        iturnfirst = self.findFirstTurn(ind=ind, first=True,  prominence=prominence,
                                    min_width=min_width, min_height=min_height,
                                    manualturns=manualturns,
                                    Iconst=Iconst, Imods=Imods, Iscans=Iscans, neighbors=neighbors)
    else:
        iturnfirst = firstturn
    if lastturn is None:
        iturnlast  = self.findFirstTurn(ind=ind, first=False, prominence=prominence,
                                    min_width=min_width, min_height=min_height,
                                    manualturns=manualturns,
                                    Iconst=Iconst, Imods=Imods, Iscans=Iscans, neighbors=neighbors)
    else:
        iturnlast = lastturn
    if i_max is None:
        iend = self.npoints
    elif i_max < 0:
        iend = self.npoints + i_max
    else:
        iend = i_max
    # Set up index ranges to loop through
    ilist = np.concatenate((np.array([i_min]),
                            np.arange(iturnfirst+self.nm//4, iturnlast, self.nm/2),
                            np.array([iend]),
                            ))
    ilist = np.array([np.int(np.round(i_)) for i_ in ilist])
    # Loop through ranges to find peaks
    peaklist, turnlist, wrongrangelist = [], [], []
    for i, (i1, i2) in enumerate(zip(ilist[:-2], ilist[2:])):
        if printupdates:
            print('Fitting irange = {}'.format((i1,i2)))
        peaks, turns = self.findPeaksAndTurns(ind=(i1,i2), prominence=prominence,
                                              min_width=min_width,
                                              removepairs=removepairs,
                                              min_height=min_height,
                                              manualturns=manualturns,
                                              Iconst=Iconst, Imods=Imods, Iscans=Iscans,
                                              neighbors=neighbors,
                                              plotflag_peaks=plotflag_peaks,
                                              plotflag_turns=plotflag_turns)
        if len(turns['i_global_pos']) + len(turns['i_global_neg']) != 2:
            wrongrangelist.append( (i1, i2) )
            plotData([(self.t[i1:i2], self.I[i1:i2]),
                      (self.t[peaks['i_global_pos']], self.I[peaks['i_global_pos']]),
                      (self.t[peaks['i_global_neg']], self.I[peaks['i_global_neg']]),
                      (self.t[turns['i_global_pos']], self.I[turns['i_global_pos']]),
                      (self.t[turns['i_global_neg']], self.I[turns['i_global_neg']]),
                      ],
                     labels = ['data',
                               'upper peaks',
                               'lower peaks',
                               'upper turns',
                               'lower turns'],
                     linestyles = ['k-', 'g^', 'gv', 'bo', 'bo'],
                     xlabel = 'Time (s)',
                     ylabel = 'Signal (V)',
                     title = 'preprocessEtalon.countPeaks: RunTimeError: Wrong number of turn peaks found'
                     )
            warnings.warn('preprocessEtalon.countPeaks() found {} turn peaks in index range {} - {}. 2 expected.'.format(len(turns['i_global_pos']) + len(turns['i_global_neg']), i1, i2))
        else:
            firstpeaks, centerpeaks, endpeaks = self.peaksBetweenTurns(peaks, turns)
            turns_ = np.concatenate((turns['i_global_pos'], turns['i_global_neg']))
            turns_.sort()
            turn1, turn2 = turns_
            if i == 0:
                peaklist.append(firstpeaks)
                turnlist.append(set([turn1]))
            peaklist.append(centerpeaks)
            turnlist[-1].add(turn1)
            turnlist.append(set([turn2]))
    peaklist.append(endpeaks)
    tpeakcount, peakcount = self.cumulativeCount(peaklist=peaklist)
    if plotflag:
        plotData((tpeakcount,peakcount), linestyles='ko', labels='Data',
                 xlabel='Time (s)', ylabel='Cumulative Count', title='countPeaks')
    return {'iturn1'     : iturnfirst,
            'turnlist'   : turnlist,
            'peaklist'   : peaklist,
            'tpeakcount' : tpeakcount,
            'peakcount'  : peakcount,
            'wrongrange' : wrongrangelist,
            }

def cumulativeCount(self, peaklist):
    """
    cumulativeCount(peaklist=None)
    peaklist = list of 2-lists of arrays of peak indices [[ipos, ineg()], [,], ...]
    Returns tpeakcount, peakcount arrays
    """
    ipeakcount = []
    peakcount  = []
    sign = -1
    for peaks_ in peaklist:
        sign *= -1
        peaks = np.concatenate(peaks_)
        peaks.sort()
        ipeaks = [ipeak for ipeak in peaks]
        lastcount = peakcount[-1] if len(peakcount) != 0 else 0
        peakcount.extend([lastcount + sign*i for i in range(len(ipeaks))])
        ipeakcount.extend(ipeaks)
    tpeakcount = self.t[ipeakcount]
    peakcount = np.array(peakcount)
    return tpeakcount, peakcount

def findFirstTurn(self, ind=None, first=True, prominence=0.5, min_width=None, min_height=0.5, neighbors=3, manualturns=None, maxfev=1000, Iconst=True, Imods=(True,True), Iscans=(False,False), plotflag=False):
    """
    findFirstTurn()
    Finds index of first (or last) modulation turn
    ind = (i_min, i_max) global index bounds to search, default is first 2 modulation periods
    first = bool, returns 1st turn if True, otherwise returns last turn
    prominence = float 0:2, min prominence in "normalized" data for peak finding.  Default 0.5
    min_width  = int, minimum width of acceptable peak. Smaller width peaks will be filtered out. Default None
    min_height = float in range (0:2), minimum height of turn peak for obvious filtering
    neighbors = int, number of neighboring peaks to check around suspicious turn peak
    manualturns = array of global indices of manually found turn peaks to use instead of search algorithm
    Iconst = bool
    Imods  = (bool, bool) to use (1st, 2nd) modulation harmonics in fitting I
    Iscans = (bool, bool) to use (1st, 2nd) scan harmonics in fitting I
    Return global index of first (or last) modulation turn index
    """
    if ind is None:
        if first:
            ind = (0, 2*self.nm)
        else:
            ind = (-2*self.nm, None)
    ind = self.handle_ind(ind)
    _, turndata = self.findPeaksAndTurns(ind=ind, prominence=prominence,
                                         min_width=min_width, min_height=min_height,
                                         manualturns=manualturns,
                                         maxfev=maxfev,
                                         Iconst=Iconst, Imods=Imods, Iscans=Iscans,
                                         neighbors=neighbors,
                                         plotflag_turns=plotflag)
    iturns = np.concatenate((turndata['i_global_pos'], turndata['i_global_neg']))
    iturns.sort()
    if plotflag:
        i_min, i_max = ind
        i_first_or_last = 0 if first else -1
        plotData([(self.t[i_min:i_max],self.I[i_min:i_max]),
                  (self.t[iturns[i_first_or_last]], self.I[iturns[i_first_or_last]])],
                 linestyles = ['k-', 'bo'],
                 labels = ['data', 'first turn' if first else 'last turn'],
                 xlabel = 'Time (s)',
                 ylabel = 'Signal (V)',
                 title = 'findFirstTurn')
    return iturns[0] if first else iturns[-1]

def peaksBetweenTurns(self, peakdata, turndata):
    """
    peaksBetweenTurns(peakdata, turndata)
    Separate the peak data into separate lists, split at turn times
    peakdata = dict result from findAllPeakIndices
    turndata = dict result from findTurnPeaks
    Returns list of 2-lists of arrays containing global peak indices [[i_pos, i_neg], [i_pos,i_neg], ...]
    """
    iturns = np.concatenate((turndata['i_global_pos'], turndata['i_global_neg'],
                             np.array([0]), np.array([np.inf])))
    iturns.sort()
    allpeaks = []
    for j, (iturn1, iturn2) in enumerate(zip(iturns[:-1], iturns[1:])):
        allpeaks.append([])
        for i, suffix in enumerate(['_pos', '_neg']):
            allpeaks[j].append(peakdata['i_global' + suffix][(peakdata['i_global' + suffix] > iturn1) &
                                                             (peakdata['i_global' + suffix] < iturn2)])
    return allpeaks

def findPeaksAndTurns(self, ind=(0,None), prominence=0.5, min_width=None, removepairs=False, min_height=0.5, neighbors=3, manualturns=None, maxfev=1000, Iconst=True, Imods=(True,True), Iscans=(True,True), plotflag_peaks=False, plotflag_turns=False):
    """
    findPeaksAndTurns(ind=(0,None), prominence=0.5, min_width=None, min_height=0.5, neighbors=3, plotflag=False)
    Find indices of valid peaks and turn points in range ind
    ind = (i_min, i_max) global index bounds to search, default is all data
    prominence = float 0:2, min prominence in "normalized" data for peak finding.  Default 0.5
    min_width  = int, minimum width of acceptable peak. Smaller width peaks will be filtered out. Default None
    removepairs = bool, whether to remove peak following one below min_width. Defalut False
    min_height = float in range (0:2), minimum height of turn peak for obvious filtering
    neighbors = int, number of neighboring peaks to check around suspicious turn peak
    manualturns = array of global indices of manually found turn peaks to use instead of search algorithm
    Iconst = bool
    Imods  = (bool, bool) to use (1st, 2nd) modulation harmonics in fitting I
    Iscans = (bool, bool) to use (1st, 2nd) scan harmonics in fitting I
    plotflag_peaks = bool, plot peak finding 2nd round
    plotflag_turns = bool, plot turn finding 2nd round
    Returns dictionaries:
        peaks with keys:
            ind, index bounds of search
            i_local_pos & i_local_neg, local indices of peaks within range of ind
            i_global_pos & i_global_neg, global indices of peaks
            heights_pos & heights_neg, heights of normalized peaks relative to -1 (or 1)
            widths_pos & widths_neg, peak widths from scipy.signal.find_peaks
        turns with keys:
            i_local_pos & i_local_neg, local indices within range peakdata['ind'] of invalid peaks
            i_global_pos  & i_global_neg,  global indices of invalid peaks
    """
    i_guess = None
    for i in range(2):
        peaks = self.findAllPeakIndices(ind=ind, i_guess=i_guess,
                                        prominence=prominence, min_width=min_width, removepairs=removepairs,
                                        maxfev=maxfev,
                                        Iconst=Iconst, Imods=Imods, Iscans=Iscans,
                                        plotflag=plotflag_peaks if i==1 else False)
        turns = self.findTurnPeaks(peaks, min_height=min_height,
                                   neighbors=neighbors, manualturns=manualturns,
                                   plotflag=plotflag_turns if i==1 else False)
        peaks = self.removeTurnPeaks(peaks, turns)
        i_guess = (peaks['i_global_pos'], peaks['i_global_neg'])
    return peaks, turns

def removeTurnPeaks(self, peakdata, turndata):
        """
        removeTurnPeaks(peakdata, turndata)
        Removes turn peaks from peak data
        peakdata = dict result from findAllPeakIndices
        turndata = dict result from findTurnPeaks
        Returns peakdata dict with turn points removed
        """
        for suffix in ['_pos', '_neg']:
            peakkeys = [key for key in peakdata.keys() if key.endswith(suffix)]
            for i_global in turndata['i_global' + suffix]:
                i_peak = np.where(peakdata['i_global' + suffix] == i_global)[0]
                for key in peakkeys:
                    peakdata[key] = np.delete(peakdata[key],i_peak)
        return peakdata

def findTurnPeaks(self, peakdata, min_height=0.5, neighbors=3, manualturns=None, plotflag=False):
    """
    findTurnPeaks(peakdata, minheight=0.5, neighbors=3, plotflag=False)
    peakdata = dictionary output from findAllPeakIndices()
    min_height = float in range (0:2), minimum height of turn peak
    neighbors = int, number of neighboring peaks to check around suspicious peak
    manualturns = array of global indices of manually found turn peaks to use instead of search algorithm
    plotflag = bool
    Find peaks corresponding to wavelength reversals (invalid peaks) in peakdata
    Returns dictionary with keys:
        i_local_pos & i_local_neg, local indices within range peakdata['ind'] of invalid peaks
        i_global_pos  & i_global_neg,  global indices of invalid peaks
    """
    def plotTurns():
        ind_data = np.arange(peaks[in_range]['i_global'][0],peaks[in_range]['i_global'][-1])
        ind_points = peaks[in_range]['i_global']
        xy = [(self.t[ind_data],self.I[ind_data]),
              (self.t[ind_points],self.I[ind_points])]
        labels = ['data', 'peaks']
        styles = ['k-', 'g+']
        if not obvious:
            ind_check = peaks[check][order3_dt][order3_t]['i_global']
            xy.extend([(self.t[ind_check],self.I[ind_check])])
            labels.append('checked peaks')
            styles.append('ro')
        if not successive:
            ind_turn = peaks['i_global'][turns[-1]]
        else:
            _, ind_turn, _ = successive_peaks[-1]
        xy.extend([(self.t[ind_turn],self.I[ind_turn])])
        labels.append('turn')
        styles.append('bd')
        xlabel = 'Time (s)'
        ylabel = 'Signal (V)'
        title  = 'findTurnPeaks'
        plotData(xy, labels=labels, linestyles=styles, xlabel=xlabel, ylabel=ylabel, title=title)
    # Get peak times
    t_pos = self.t[peakdata['i_global_pos']]
    t_neg = self.t[peakdata['i_global_neg']]
    # Format peak data in structured array
    peakdtype = [('i_peak',np.int), ('i_global',np.int), ('i_local',np.int), ('pos','?'), ('t',np.float64),
                 ('dt_for',np.float64), ('dt_rev',np.float64), ('dt_cen',np.float64),
                 ('height',np.float64), ('width',np.float64)]
    peaks = np.zeros(len(t_pos) + len(t_neg) - 4, dtype=peakdtype)
    for i in range(1, len(t_pos)-1):
        peaks[i-1] = np.array([(0,
                               peakdata['i_global_pos'][i],
                               peakdata['i_local_pos'][i],
                               True,
                               t_pos[i],
                               t_pos[i+1] - t_pos[i],
                               t_pos[i] - t_pos[i-1],
                               t_pos[i+1] - t_pos[i-1],
                               peakdata['heights_pos'][i],
                               peakdata['widths_pos'][i],
                               )], dtype=peakdtype)
    for i in range(1, len(t_neg)-1):
        peaks[len(t_pos)-3+i] = np.array([(0,
                               peakdata['i_global_neg'][i],
                               peakdata['i_local_neg'][i],
                               False,
                               t_neg[i],
                               t_neg[i+1] - t_neg[i],
                               t_neg[i] - t_neg[i-1],
                               t_neg[i+1] - t_neg[i-1],
                               peakdata['heights_neg'][i],
                               peakdata['widths_neg'][i],
                               )], dtype=peakdtype)
    # Sort peaks by increasing index / time
    peaks = peaks[np.argsort(peaks, order=['i_global'])]
    peaks['i_peak'] = np.arange(len(peaks))
    npeaks = len(peaks)
    # If a manualturn exists in the range, use that for initialization
    if (manualturns is not None) and np.any((manualturns>=peaks['i_global'][0]) & (manualturns<=peaks['i_global'][-1])):
        i_init = manualturns[(manualturns>=peaks['i_global'][0]) & (manualturns<=peaks['i_global'][-1])][0]
    # Otherwise, find index of one turn peak in the whole time series with the smallest height, or largest central time gap
    else:
        if np.min(peaks['height']) < 2-min_height:
            i_init = peaks['i_global'][np.argmin(peaks['height'])]
        else:
            i_init = peaks['i_global'][np.argmax(peaks['dt_cen'])]
    # Define intervals based on this peak to search for invalid peaks
    i_list = np.arange(peaks['i_global'][0] + np.mod(i_init-peaks['i_global'][0], (self.SR/self.fm)/2) + (1/4)*(self.SR/self.fm),
                       peaks['i_global'][-1] - np.mod(peaks['i_global'][-1]-i_init, (self.SR/self.fm)/2),
                       (self.SR/self.fm)/2)
    i_list = np.array([np.int(np.round(i_)) for i_ in i_list])
    i_start_list = np.concatenate((np.array([peaks['i_global'][0]]), i_list))
    i_end_list   = np.concatenate((i_list, np.array([peaks['i_global'][-1]])))
    nseg = len(i_start_list)
    # Check peaks in each interval to find invalid peaks
    turns = []
    successive_peaks = []
    raiseError = False
    for i, (i_start, i_end) in enumerate(zip(i_start_list,i_end_list)):
        in_range = np.where((peaks['i_global']>=i_start) & (peaks['i_global']<=i_end))[0]
        max_central = np.argmax(peaks[in_range]['dt_cen']) if len(in_range)>0 else None
        obvious = False
        successive = False
        # If any manualturn in range use nearest turn to that manualturn index
        if (manualturns is not None) and np.any((manualturns>=i_start) & (manualturns<=i_end)):
            obvious = True
            mturn = manualturns[(manualturns>=i_start) & (manualturns<=i_end)]
            if len(mturn) > 1:
                raise ValueError('Too many manualturns entered in same half modulation period: {}'.format(mturn))
            turns.append(peaks[in_range]['i_peak'][np.argmin(np.abs(peaks[in_range]['i_global'] - mturn))])
        # If min height peak is obviously low, use that peak
        elif (len(in_range)>1) and (np.min(peaks[in_range]['height']) < 2-min_height):
            obvious = True
            turns.append(peaks[in_range]['i_peak'][np.argmin(peaks[in_range]['height'])])
        # Otherwise check neighbors around max central dt
        elif (max_central is not None) and ((i!=0 and i!=nseg-1) or (i==0 and max_central>1) or (i==nseg-1 and max_central<npeaks-2)):
            check = np.arange(peaks[in_range]['i_peak'][max_central] - neighbors,
                              peaks[in_range]['i_peak'][max_central] + neighbors + 1)
            check = check[(check>=0) & (check<npeaks)]
            order3_dt = np.argsort(peaks[check], order=['dt_for'])[-3:]
            order3_t  = np.argsort(peaks[check][order3_dt], order=['t'])
            ii_check  = peaks[check][order3_dt][order3_t]['i_peak']
            # Check for any successive peaks of same type
            ii_check_all = np.arange(ii_check[0], ii_check[-1]+1)
            for ii_ in ii_check_all[:-1]:
                if peaks['pos'][ii_] == peaks['pos'][ii_+1]:
                    successive = True
                    s_local  = np.int(np.mean([peaks['i_local'][ii_],
                                               peaks['i_local'][ii_+1]]))
                    s_global = np.int(np.mean([peaks['i_global'][ii_],
                                               peaks['i_global'][ii_+1]]))
                    s_pos    = False if peaks['pos'][ii_] else True
                    successive_peaks.append((s_local, s_global, s_pos))
                    break
            # Otherwise use peak order
            if not successive:
                dii_check = np.abs(np.diff(ii_check))
                if dii_check[0]!=1 and dii_check[1]==1:
                    turns.append(ii_check[-1])
                elif dii_check[0]==1 and dii_check[1]!=1:
                    turns.append(ii_check[-2])
                elif dii_check[0]!=1 and dii_check[1]!=1:
                    # If no consecutive peaks found, plot and error
                    plotflag = True
                    raiseError = True
                    break
                else:
                    turns.append(ii_check[-1])
        if plotflag:
            plotTurns()
    # Filter peaks into positive and negative lists
    turns_pos = peaks[turns][peaks[turns]['pos']==True]
    turns_neg = peaks[turns][peaks[turns]['pos']==False]
    i_local_pos  = turns_pos['i_local']
    i_local_neg  = turns_neg['i_local']
    i_global_pos = turns_pos['i_global']
    i_global_neg = turns_neg['i_global']
    # Add successive peaks to lists
    for (s_local, s_global, s_pos) in successive_peaks:
        if s_pos:
            i_local_pos =  np.append(i_local_pos,  s_local)
            i_global_pos = np.append(i_global_pos, s_global)
        else:
            i_local_neg =  np.append(i_local_neg,  s_local)
            i_global_neg = np.append(i_global_neg, s_global)
    i_local_pos.sort()
    i_local_neg.sort()
    i_global_pos.sort()
    i_global_neg.sort()
    if raiseError:
        raise RuntimeError('Non-consecutive suspicious peaks found in findTurnPeaks near ind=({},{})'.format(i_start,i_end))
    return {'i_local_pos' : i_local_pos,
            'i_local_neg' : i_local_neg,
            'i_global_pos'  : i_global_pos,
            'i_global_neg'  : i_global_neg,
            }

def findAllPeakIndices(self, ind=(0,None), prominence=0.5, min_width=None, removepairs=False, i_guess=None, maxfev=1000, Iconst=True, Imods=(True,True), Iscans=(True,True), plotflag=False, plotflag_Ifits=False):
    """
    findAllPeakIndices(ind=(0,None), prominence=0.5, min_width=None, i_guess=None, Imods=(True,True), Iscans=(True,True), plotflag=False)
    ind = (i_min, i_max) global index bounds to search, default is all data
    prominence = float 0:2, min prominence in "normalized" data.  Default 0.5
    min_width  = int, minimum width of acceptable peak. Smaller width peaks will be filtered out. Default None
    removepairs = bool, whether to remove peak following one below min_width. Defalut False
    i_guess = (i_upper, i_lower) int arrays, upper and lower global peak indices to fit IEquation for normalization.
    Iconst = bool
    Imods  = (bool, bool) to use (1st, 2nd) modulation harmonics in fitting I
    Iscans = (bool, bool) to use (1st, 2nd) scan harmonics in fitting I
    plotflag = bool, plot peak finding results
    plotflag_Ifits = bool, plot Intensity envelope fits
    Find indices of all peaks in data
    Returns dictionary with keys:
        ind, index bounds of search
        i_local_pos & i_local_neg, local indices of peaks within range of ind
        i_global_pos & i_global_neg, global indices of peaks
        heights_pos & heights_neg, heights of normalized peaks relative to -1 (or 1)
        widths_pos & widths_neg, peak widths from scipy.signal.find_peaks
    """
    ind = self.handle_ind(ind)
    i_min, i_max = ind
    mod1f, mod2f = Imods
    scan1f, scan2f = Iscans
    min_width = 0 if min_width is None else min_width
    # If starting from scratch, estimate normalization
    if i_guess is None:
        popt = self.fitI(ind=ind, maxfev=maxfev, plotflag=plotflag_Ifits,
                         const=Iconst, mod1f=mod1f, mod2f=mod2f,
                         scan1f=scan1f, scan2f=scan2f)
        I_norm = -1 + np.divide(self.I[i_min:i_max],
                                self.IEquation(self.t[i_min:i_max],*popt))
        I_norm = np.divide(I_norm,
                           2*np.median(np.abs(I_norm)))
    # Else use fit at given points to normalize data
    else:
        popt_pos, popt_neg = [self.fitI(points=i_, maxfev=maxfev, plotflag=plotflag_Ifits,
                                        const=Iconst, mod1f=mod1f, mod2f=mod2f,
                                        scan1f=scan1f, scan2f=scan2f) for i_ in i_guess]
        I_fit_pos, I_fit_neg = [self.IEquation(self.t[i_min:i_max], *popt) for popt in [popt_pos, popt_neg]]
        I_norm = -1 + 2*np.divide(self.I[i_min:i_max]-I_fit_neg, I_fit_pos-I_fit_neg)
    # Find peaks, heights, and parameters from normalized data
    (i_local_pos, params_pos), (i_local_neg, params_neg) = [find_peaks(k*I_norm,
                                                                       prominence=prominence,
                                                                       width=0 if removepairs else min_width)
                                                            for k in [1,-1]]
    if removepairs:
        (i_local_pos, params_pos), (i_local_neg, params_neg) = self.filterPeakPairsByWidth((i_local_pos, params_pos),
                                                                                           (i_local_neg, params_neg),
                                                                                           min_width)
    (heights_pos, heights_neg) = [k*I_norm[ii_] - -1 for k,ii_ in [(1,i_local_pos), (-1,i_local_neg)]]
    i_global_pos, i_global_neg = i_local_pos + i_min, i_local_neg + i_min
    # Plot
    if plotflag:
        xy = [(self.t[i_min:i_max],self.I[i_min:i_max]),
              (self.t[i_global_pos],self.I[i_global_pos]),
              (self.t[i_global_neg],self.I[i_global_neg])]
        styles = ['k-','r^','rv']
        labels = ['data','upper peaks','lower peaks']
        if i_guess is not None:
            xy.extend([(self.t[i_min:i_max],I_fit_pos),
                       (self.t[i_min:i_max],I_fit_neg)])
            styles.extend(['g-','g--'])
            labels.extend(['upper peak fit','lower peak fit'])
        else:
            xy.extend([(self.t[i_min:i_max],self.IEquation(self.t[i_min:i_max],*popt))])
            styles.extend(['g-'])
            labels.extend(['single IEquation fit'])
        plotData(xy,
                 linestyles=styles,
                 labels=labels,
                 title='findAllPeakIndices',
                 xlabel='Time (s)',
                 ylabel='Signal (V)')
    return {'ind' : ind,
            'i_local_pos' : i_local_pos,
            'i_local_neg' : i_local_neg,
            'i_global_pos' : i_global_pos,
            'i_global_neg' : i_global_neg,
            'heights_pos' : heights_pos,
            'heights_neg' : heights_neg,
            'widths_pos' : params_pos['widths'],
            'widths_neg' : params_neg['widths']}

def filterPeakPairsByWidth(self, peakdata_pos, peakdata_neg, min_width):
    """
    preprocessEtalon.filterPeakPairsByWidth(peakdata_pos, peakdata_neg, min_width)
    peakdata_* = tuple of:
        i_local_* index array
        params_* dict of peak finding parameters
    min_width = minimum peak width, in samples, to filter by
    Returns filtered peakdata_pos, peakdata_neg
    """
    (i_local_pos, params_pos), (i_local_neg, params_neg) = peakdata_pos, peakdata_neg
    # Check if any peaks below min_width, otherwise do nothing
    if np.any(params_pos['widths'] < min_width) or np.any(params_neg['widths'] < min_width):
        # Format peak data in structured array
        peakdtype = [('i_peak',np.int), ('i_local',np.int), ('pos','?'), ('width',np.float64)]
        peaks = np.zeros(len(i_local_pos) + len(i_local_neg), dtype=peakdtype)
        for i in range(len(i_local_pos)):
            peaks[i] = np.array([(i,
                                  i_local_pos[i],
                                  True,
                                  params_pos['widths'][i],
                                  )],
                                dtype=peakdtype)
        for i in range(len(i_local_neg)):
            peaks[len(i_local_pos)+i] = np.array([(i,
                                  i_local_neg[i],
                                  False,
                                  params_neg['widths'][i],
                                  )],
                                dtype=peakdtype)
        # Sort peaks by increasing index / time
        peaks = peaks[np.argsort(peaks, order=['i_local'])]
        # Find peaks with widths below min_width
        i_less = np.where(peaks['width'] < min_width)[0]
        # Determine peaks to remove. Remove following point if not part of a pair.
        remove = [i_less[0], i_less[0]+1]
        for i_ in i_less[1:]:
            remove.append(i_)
            if i_-1 not in remove:
                remove.append(i_+1)
        remove = np.array(remove)
        remove = remove[remove < len(peaks)]
        # Remove peaks
        peaks = np.delete(peaks, remove)
        # Get original indices of valid peaks
        i_pos = peaks[peaks['pos']==True]['i_peak']
        i_neg = peaks[peaks['pos']==False]['i_peak']
        # Filter peaks back into lists / dicts
        i_local_pos = i_local_pos[i_pos]
        i_local_neg = i_local_neg[i_neg]
        for key, value in params_pos.items():
            params_pos[key] = value[i_pos]
        for key, value in params_neg.items():
            params_neg[key] = value[i_neg]
    return (i_local_pos, params_pos), (i_local_neg, params_neg)

def handle_ind(self, ind):
    """

    """
    i_min, i_max = ind
    if i_min is None:
        i_min = 0
    elif i_min < 0:
        i_min += self.npoints
    if i_max is None:
        i_max = self.npoints-1
    elif i_max < 0:
        i_max += self.npoints
    return i_min, i_max

def handle_points(self, points=None, ind=None):
    """

    """
    if points is None:
        i_min, i_max = self.handle_ind(ind)
        points = np.arange(i_min, i_max)
    return points

def handle_necessary(self, guess=[None]*9, using=[True]*9):
    """

    """
    necessary = False
    for guess_, using_ in zip(guess, using):
        if (guess_ is None) and (using_):
            necessary = True
            break
    return necessary

def checkBounds(self, guess, bounds):
    """

    """
    bounds_lower, bounds_upper = bounds
    for i_, (guess_, bound_lower, bound_upper) in enumerate(zip(guess, bounds_lower, bounds_upper)):
        if (guess_ < bound_lower) or (guess_ > bound_upper):
            raise ValueError('guess[{}]={} is outside bounds ({}:{}) \n guess={} \n bounds={}'.format(i_, guess_, bound_lower, bound_upper, guess, bounds))

def handle_I_GuessAndBounds(self, points=None, ind=None, guess=None, bounds=None, using=None):
    """
    avg, mod1f, phase1f, mod2f, phase2f, scan1f, phase1f, scan2f, phase2f
    """
    if guess is None:
        guess = [None]*9
    if bounds is None:
        bounds = [None]*9, [None]*9
    if using is None:
        using = [True]*9
    replace   = np.zeros(9)
    bounds_lower, bounds_upper = bounds
    # Setup indices
    i_const, i_1fs, i_2fs, i_mods, i_scans = 0, [1, 5], [3, 7], [1, 3], [5, 7]
    points = self.handle_points(points=points, ind=ind)
    necessary = self.handle_necessary(guess=guess, using=using)
    if necessary:
        # Calculate stats
        I_avg = np.mean(self.I[points])
        I_amp = (np.max(self.I[points]) - np.min(self.I[points])) / 2
        # Setup replacement guesses for any Nones in guess
        replace[i_const] = I_avg
        replace[i_1fs]   = I_amp
        replace[i_2fs]   = I_amp / 10
    # Loop and assign guesses and bounds
    for i, guess_ in enumerate(guess):
        if guess_ is None:
            guess[i] = replace[i]
        if bounds_lower[i] is None:
            if i == i_const:
                bounds_lower[i] = -np.inf #I_avg-I_amp if necessary else -np.inf
            elif (i in i_mods) or (i in i_scans):
                bounds_lower[i] = 0
            else:
                bounds_lower[i] = guess[i] - 2*np.pi
        if bounds_upper[i] is None:
            if i == i_const:
                bounds_upper[i] = np.inf #I_avg+I_amp if necessary else  np.inf
            elif i in i_mods:
                bounds_upper[i] = 2*I_amp if necessary else np.inf
            elif i in i_scans:
                bounds_upper[i] = np.inf
            else:
                bounds_upper[i] = guess[i] + 2*np.pi
    return np.array(guess), (np.array(bounds_lower), np.array(bounds_upper))

def handle_v_GuessAndBounds(self, v=None, ind=None, guess=None, bounds=None, using=None):
    """
    either given v or ind
    avg, mod1f, phase1f, mod2f, phase2f, scan1f, phase1f, scan2f, phase2f
    """
    if guess is None:
        guess = [None]*9
    if bounds is None:
        bounds = [None]*9, [None]*9
    if using is None:
        using = [True]*9
    replace   = np.zeros(9)
    bounds_lower, bounds_upper = bounds
    # Setup indices
    i_const, i_1fs, i_2fs, i_mods, i_scans = 0, [1, 5], [3, 7], [1, 3], [5, 7]
    # Setup replacement guesses for any missing necessary guesses
    necessary = self.handle_necessary(guess=guess, using=using)
    if necessary:
        if v is None:
            count = self.countPeaks(ind=ind)
            v = (count['peakcount'] - np.mean(count['peakcount'])) * (1/2)
        # Calculate stats
        v_avg = np.mean(v)
        v_amp = (np.max(v) - np.min(v)) / 2
        # Setup replacement values for Nones in guess
        replace[i_const] = v_avg
        replace[i_1fs]   = v_amp
        replace[i_2fs]   = v_amp / 10
    # Loop and assign guesses and bounds
    for i, guess_ in enumerate(guess):
        if guess_ is None:
            guess[i] = replace[i]
        if bounds_lower[i] is None:
            if i == i_const:
                bounds_lower[i] = -np.inf #v_avg-v_amp if necessary else -np.inf
            elif (i in i_mods) or (i in i_scans):
                bounds_lower[i] = 0
            else:
                bounds_lower[i] = guess[i] - 2*np.pi
        if bounds_upper[i] is None:
            if i == i_const:
                bounds_upper[i] = np.inf #v_avg+v_amp if necessary else  np.inf
            elif i in i_mods:
                bounds_upper[i] = 2*v_amp if necessary else np.inf
            elif i in i_scans:
                bounds_upper[i] = np.inf
            else:
                bounds_upper[i] = guess[i] + 2*np.pi
    return np.array(guess), (np.array(bounds_lower), np.array(bounds_upper))

def fitScaledData(self, t, data, Eq, p0, bounds, maxfev=1000):
    """

    """
    data_min = np.min(data)
    scale = 1 / np.mean(data - data_min)
    Eq_ = lambda t, *params : (Eq(t,*params) - data_min) * scale
    return curve_fit(Eq_, t, (data-data_min)*scale, p0=p0, bounds=bounds, maxfev=maxfev)

def find_dvdt_zero(self, i_start, vArgs, bisection=False):
    """

    """
    f = lambda i : newton(self.dvdt,
                          self.t[i],
                          fprime = self.dv2dt2,
                          args = vArgs)
    g = lambda i1, i2 : bisect(self.dvdt,
                               self.t[i1],
                               self.t[i2],
                               args = vArgs)
    if (not bisection) and isinstance(i_start, (list, tuple, np.ndarray)):
        izero = self.nearestIndex([f(i_) for i_ in i_start])
    elif not bisection:
        izero = self.nearestIndex(f(i_start))
    else:
        izero = self.nearestIndex(g(*i_start))
    return izero

def fitModel(self, ind=(0,None), firstturn=None, lastturn=None, forward=False, reestimate_Ifits=False, reestimate_Isettings=(True,True,True,False,False), maxfev=1000, plotflag=False, plotflag_initial=False, printupdates=False, **kwargs):
    """

    """
    # Process input indices
    i_min, i_max = self.handle_ind(ind)
    if firstturn is None:
        firstturn = self.findFirstTurn(ind=(i_min,i_max), first=True, maxfev=maxfev)
    elif firstturn < 0:
        firstturn += self.npoints
    if lastturn is None:
        lastturn = self.findFirstTurn(ind=(i_min,i_max), first=False, maxfev=maxfev)
    elif lastturn < 0:
        lastturn += self.npoints
    # Create list of index ranges to fit
    i_start = firstturn - self.nm//4
    count = 0
    while i_start < i_min:
        i_start += self.nm//2
        count += 1
        if count >= 3:
            raise RuntimeError('Could not find turn index > ind[0] in preprocessEtalon.fitModel')
    i_end = lastturn + self.nm//4
    count = 0
    while i_end > i_max:
        i_end -= self.nm//2
        count += 1
        if count >= 3:
            raise RuntimeError('Could not find turn index < ind[1] in preprocessEtalon.fitModel')
    i_list = np.round(np.arange(i_start, i_end, (self.SR/self.fm)/2 )).astype(np.int)
    # Initialize data storage structured array
    vtype = {'names'   : ['v0', 'vm1', 'phim1', 'vm2', 'phim2', 'vs1', 'phis1', 'vs2', 'phis2'],
             'formats' : [np.float64]*9}
    Itype = {'names'   : ['I0', 'Im1', 'phim1', 'Im2', 'phim2', 'Is1', 'phis1', 'Is2', 'phis2'],
             'formats' : [np.float64]*9}
    dtype = [('i', np.int, 2),
             ('vrel', np.float64, 2),
             ('vshift', np.float64),
             ('vArgs', vtype),
             ('ILowerArgs', Itype),
             ('IUpperArgs', Itype),
             ('k', np.float64)]
    results = np.zeros(len(i_list)-2, dtype=dtype)
    # Try fitting first (or last) segment
    irange = (i_list[0 if forward else -3],
              i_list[2 if forward else -1])
    mfit, kwargs = self.tryModelVCombinations(irange,
                                              plotflag=plotflag_initial,
                                              **kwargs)
    kwargs['plotflag'] = plotflag
    # Find inner index where dvdt=0
    iturn = self.find_dvdt_zero(irange[-1]-self.nm//4 if forward else irange[0]+self.nm//4,
                                mfit['vArgs'])
    i1, i2 = (int(np.mean(irange)) if forward else irange[0],
              irange[1] if forward else int(np.mean(irange)))
    if (iturn < i1) or (iturn > i2):
        iturn = self.find_dvdt_zero((i1, i2),
                                    tuple(mfit['vArgs']),
                                    bisection=True)
    # Determine the endpoint indices of the range the first fit applies to
    i_apply = (i_min, iturn) if forward else (iturn, i_max)
    # Calculate the wavelength at these endpoints
    vrel = self.vEquation(self.t[np.array(i_apply)], *mfit['vArgs'])
    vshift = -vrel[0 if forward else -1]
    #Store the initial fit
    results[0 if forward else -1] = (i_apply,
                                     tuple(vrel),
                                     vshift,
                                     tuple(mfit['vArgs']),
                                     tuple(mfit['ILowerArgs']),
                                     tuple(mfit['IUpperArgs']),
                                     mfit['k'])
    # Loop through all central index ranges
    iranges = zip(i_list[1:-3], i_list[3:-1])
    for i, irange in enumerate(iranges) if forward else enumerate(reversed(list(iranges))):
        if printupdates:
            print('Fitting irange = {}'.format(irange))
        # Fit range with initial guesses as previous ranges' result
        if reestimate_Ifits:
            Iconst_refit, Imod1f_refit, Imod2f_refit, Iscan1f_refit, Iscan2f_refit = reestimate_Isettings
            peaks_refit, turns_refit = self.findPeaksAndTurns(ind=irange, maxfev=maxfev, Iconst=Iconst_refit, Imods=(Imod1f_refit,Imod2f_refit), Iscans=(Iscan1f_refit,Iscan2f_refit)) # manualturns=iturn + (np.arange(2)*self.nm//2*(1 if forward else -1)),
            Iguess_upper_refit = self.fitI(points=peaks_refit['i_global_pos'], guess=mfit['IUpperArgs'], maxfev=maxfev, const=Iconst_refit, mod1f=Imod1f_refit, mod2f=Imod2f_refit, scan1f=Iscan1f_refit, scan2f=Iscan2f_refit)
            Iguess_lower_refit = self.fitI(points=peaks_refit['i_global_neg'], guess=mfit['ILowerArgs'], maxfev=maxfev, const=Iconst_refit, mod1f=Imod1f_refit, mod2f=Imod2f_refit, scan1f=Iscan1f_refit, scan2f=Iscan2f_refit)
        kwargs['Iguess_upper'] = mfit['IUpperArgs'] if not reestimate_Ifits else Iguess_upper_refit
        kwargs['Iguess_lower'] = mfit['ILowerArgs'] if not reestimate_Ifits else Iguess_lower_refit
        kwargs['vguess']       = mfit['vArgs']
        kwargs['kguess']       = mfit['k']
        try:
            mfit = self.fitModelSegment(irange, **kwargs)
        except RuntimeError:
            print('Optimal parameters not found at indices {}\nTrying V setting combinations'.format(irange))
            del(kwargs['Iguess_upper'],
                kwargs['Iguess_lower'])
            mfit, kwargs = self.tryModelVCombinations(irange, **kwargs)
        # Find next index where dvdt=0
        iturn = self.find_dvdt_zero(iturn+self.nm//2 if forward else iturn-self.nm//2,
                                    mfit['vArgs'])
        i1, i2 = (int(np.mean(irange)) if forward else irange[0],
                  irange[1] if forward else int(np.mean(irange)))
        if (iturn < i1) or (iturn > i2):
            iturn = self.find_dvdt_zero((i1, i2),
                                        tuple(mfit['vArgs']), #TODO: CHECK THIS TUPLE ADDITION
                                        bisection=True)
        # Determine endpoints of the range the fit applies to
        i_apply = (i_apply[-1], iturn) if forward else (iturn, i_apply[0])
        # Calculate the previous relative wavelength at the endpoint after shifting
        vshift += vrel[-1] if forward else vrel[0]
        # Calculate the wavelength at these endpoints
        vrel = self.vEquation(self.t[np.array(i_apply)], *mfit['vArgs'])
        # Calculate the wavelength shift for stitching the fits together
        vshift -= vrel[0] if forward else vrel[-1]
        # Store the fit
        results[1+i if forward else -2-i] = (i_apply,
                                             tuple(vrel),
                                             vshift,
                                             tuple(mfit['vArgs']),
                                             tuple(mfit['ILowerArgs']),
                                             tuple(mfit['IUpperArgs']),
                                             mfit['k'])
    # Handle last fit period
    irange = (i_list[-3 if forward else 0],
              i_list[-1 if forward else 2])
    kwargs['Iguess_upper'] = mfit['IUpperArgs']
    kwargs['Iguess_lower'] = mfit['ILowerArgs']
    kwargs['vguess']       = mfit['vArgs']
    kwargs['kguess']       = mfit['k']
    try:
        mfit = self.fitModelSegment(irange, **kwargs)
    except RuntimeError:
        print('Optimal parameters not found at indices {}\nTrying V setting combinations'.format(irange))
        del(kwargs['Iguess_upper'],
            kwargs['Iguess_lower'])
        mfit, kwargs = self.tryModelVCombinations(irange,  **kwargs)
    iturn = self.find_dvdt_zero(iturn+self.nm//2 if forward else iturn-self.nm//2,
                                mfit['vArgs'])
    i1, i2 = (int(np.mean(irange)) if forward else irange[0],
              irange[1] if forward else int(np.mean(irange)))
    if (iturn < i1) or (iturn > i2):
        iturn = self.find_dvdt_zero((i1, i2),
                                    tuple(mfit['vArgs']),
                                    bisection=True)
    i_apply = (i_apply[-1], i_max) if forward else (i_min, i_apply[0])
    vshift += vrel[-1] if forward else vrel[0]
    vrel = self.vEquation(self.t[np.array(i_apply)], *mfit['vArgs'])
    vshift -= vrel[0] if forward else vrel[-1]
    results[-1 if forward else 0] = (i_apply,
                                     tuple(vrel),
                                     vshift,
                                     tuple(mfit['vArgs']),
                                     tuple(mfit['ILowerArgs']),
                                     tuple(mfit['IUpperArgs']),
                                     mfit['k'])
    return results

def calcVrelFromFitModelResults(self, results, plotflag=False):
    """

    """
    vrel_arrays = []
    for result in results:
        vrel_arrays.append(result['vshift'] +
                           self.vEquation(self.t[result['i'][0]:
                                                 result['i'][1]],
                                          *result['vArgs'])
                           )
    vrel_arrays.append(np.array([results[-1]['vshift'] +
                                 self.vEquation(self.t[results[-1]['i'][1]],
                                                *result['vArgs'])]))
    vrel = np.concatenate(tuple(vrel_arrays))
    if plotflag:
        plotData((self.t[results[0]['i'][0]:
                         results[-1]['i'][-1]+1],
                  vrel),
                 linestyles = 'ko',
                 labels = 'Vrel',
                 xlabel = 'Time (s)',
                 ylabel = 'Relative Wavenumber',
                 title = 'preprocessEtalon.calcVrelFromFitModelResults')
    return vrel

def plotFitModelResults(self, results, vrel=True, vshift=True, vArgs=True, ILowerArgs=True, IUpperArgs=True, k=True):
    """

    """
    def plot_vrel_vshift_k(keystr):
        plotData((np.concatenate(results['i']),
                  np.concatenate(results[keystr]) if keystr=='vrel' else np.repeat(results[keystr], 2)),
                 linestyles = '',
                 labels = keystr,
                 xlabel = 'Data Index',
                 ylabel = keystr,
                 title = 'preprocessEtalon.plotFitModelResults {}'.format(keystr))
    def plot_Args(keystr):
        xy, linestyles, labels = [], [], []
        for subkey in results[keystr].dtype.names:
            xy.append((np.concatenate(results['i']),
                       np.repeat(results[keystr][subkey], 2)))
            linestyles.append('')
            labels.append(subkey)
        plotData(xy,
                 linestyles = linestyles,
                 labels = labels,
                 xlabel = 'Data Index',
                 ylabel = 'Value',
                 title = 'preprocessEtalon.plotFitModelResults {}'.format(keystr))
    if vrel:
        plot_vrel_vshift_k('vrel')
    if vshift:
        plot_vrel_vshift_k('vshift')
    if k:
        plot_vrel_vshift_k('k')
    if ILowerArgs:
        plot_Args('ILowerArgs')
    if IUpperArgs:
        plot_Args('IUpperArgs')
    if vArgs:
        plot_Args('vArgs')

def tryModelVCombinations(self, ind, **kwargs):
    """

    """
    if 'vconst' not in kwargs.keys():
        kwargs['vconst'] = False
    if 'vscans' not in kwargs.keys():
        kwargs['vscans'] = (True, False)
    for i in range(4):
        try:
            modelfit = self.fitModelSegment(ind, **kwargs)
        except RuntimeError as err:
            if i == 0:
                # Original didn't work. Switch vscan2f
                kwargs['vscans'] = (kwargs['vscans'][0], not kwargs['vscans'][1])
            elif i == 1:
                # Switching vscan2f didn't work. Switch vconst, and vscan2f again
                kwargs['vconst'] = not kwargs['vconst']
                kwargs['vscans'] = (kwargs['vscans'][0], not kwargs['vscans'][1])
            elif i == 2:
                # Switched vscan2f and vconst didn't work. Switch vscan2f again
                kwargs['vscans'] = (kwargs['vscans'][0], not kwargs['vscans'][1])
            else:
                # Nothing worked. Raise error.
                raise err
        else:
            # Found a fit. Break and return
            print('Found fit at ind={} with vconst={} and vscans={}'.format(ind, kwargs['vconst'], kwargs['vscans']))
            break
    return modelfit, kwargs

def fitModelSegment(self, ind, differentialevolution=False, Iguess_upper=None, Iguess_lower=None, vguess=None, Ibounds_upper=None, Ibounds_lower=None, vbounds=None, kguess=None, deltak=0.25, klowerbound=1E-6, fixunusedguesses=True, maxfev=1000, Iconst=True, Imods=(True,True), Iscans=(False,False), vconst=False, vmods=(True,True), vscans=(True,False), prominence=0.5, min_width=None, min_height=0.5, neighbors=3, plotflag=False):
    """

    """
    if Iguess_upper is None:
        Iguess_upper = [None]*9
    if Iguess_lower is None:
        Iguess_lower = [None]*9
    if vguess is None:
        vguess = [None]*9
    Iguess_upper = np.array(Iguess_upper)
    Iguess_lower = np.array(Iguess_lower)
    vguess = np.array(vguess)
    if Ibounds_upper is None:
        Ibounds_upper = ([None]*9, [None]*9)
    if Ibounds_lower is None:
        Ibounds_lower = ([None]*9, [None]*9)
    if vbounds is None:
        vbounds = ([-2*np.pi]+[None]*8, [2*np.pi]+[None]*8)
    # Check inputs
    if len(Iguess_upper) != 9:
        raise RuntimeError('Iguess_upper input to fitModelSegment must be length 9.  Input was {}'.format(Iguess_upper))
    if len(Iguess_lower) != 9:
        raise RuntimeError('Iguess_lower input to fitModelSegment must be length 9.  Input was {}'.format(Iguess_lower))
    if len(vguess) != 9:
        raise RuntimeError('vguess input to fitModelSegment must be length 9.  Input was {}'.format(vguess))
    # Unpack input settings
    i_min, i_max = ind
    Imod1f, Imod2f   = Imods
    Iscan1f, Iscan2f = Iscans
    vmod1f, vmod2f   = vmods
    vscan1f, vscan2f = vscans
    Iusing = [Iconst] + [Imod1f]*2 + [Imod2f]*2 + [Iscan1f]*2 + [Iscan2f]*2
    vusing = [vconst] + [vmod1f]*2 + [vmod2f]*2 + [vscan1f]*2 + [vscan2f]*2
    nI = Iusing.count(True)
    nv = vusing.count(True)
    initialguesses = np.concatenate((Iguess_lower, Iguess_upper, vguess, np.array([kguess]))) if fixunusedguesses else np.zeros(28)
    # Create equations for fitting
    def ILowerEq(t,*Iparams):
        allparams = np.nan_to_num(Iguess_lower) if fixunusedguesses else np.zeros(9)
        allparams[Iusing] = Iparams
        return self.IEquation(t, *allparams)
    def IUpperEq(t,*Iparams):
        allparams = np.nan_to_num(Iguess_upper) if fixunusedguesses else np.zeros(9)
        allparams[Iusing] = Iparams
        return self.IEquation(t, *allparams)
    def vEq(t,*vparams):
        allparams = np.nan_to_num(vguess) if fixunusedguesses else np.zeros(9)
        allparams[vusing] = vparams
        return self.vEquation(t, *allparams)
    def mEq(t,*params):
        ILowerArgs_ = params[:nI]
        IUpperArgs_ = params[nI:2*nI]
        vArgs_ = params[2*nI:2*nI+nv]
        k_ = params[2*nI+nv]
        return self.modelEquation(t, ILowerArgs_, IUpperArgs_,
                                  vArgs_, k_,
                                  ILowerEq=ILowerEq, IUpperEq=IUpperEq, vEq=vEq)
    ##TODO: Remove if DE doesn't work
    def objectivefunction(params):
        return np.sum((mEq(self.t[i_min:i_max], *params) - self.I[i_min:i_max])**2)
    ##
    # Calculate bounds and rough initial guesses if necessary
    necessary_Iu = self.handle_necessary(guess=Iguess_upper, using=Iusing)
    necessary_Il = self.handle_necessary(guess=Iguess_lower, using=Iusing)
    necessary_v  = self.handle_necessary(guess=vguess, using=vusing)
    if necessary_Iu or necessary_Il or necessary_v:
        peaks, turns = self.findPeaksAndTurns(ind=ind, prominence=prominence,
                                              min_width=min_width, min_height=min_height,
                                              maxfev=maxfev,
                                              neighbors=neighbors, Iconst=Iconst, Imods=Imods, Iscans=Iscans)
        if necessary_v:
            tpeakcount, peakcount = self.cumulativeCount(peaklist=self.peaksBetweenTurns(peaks, turns))
    Iu_kwargs = {'guess'  : Iguess_upper,
                 'bounds' : Ibounds_upper,
                 'using'  : Iusing}
    Il_kwargs = {'guess'  : Iguess_lower,
                 'bounds' : Ibounds_lower,
                 'using'  : Iusing}
    v_kwargs  = {'guess'  : vguess,
                 'bounds' : vbounds,
                 'using'  : vusing}
    if necessary_Iu:
        Iu_kwargs['points'] = peaks['i_global_pos']
    else:
        Iu_kwargs['ind'] = ind
    if necessary_Il:
        Il_kwargs['points'] = peaks['i_global_neg']
    else:
        Il_kwargs['ind'] = ind
    if necessary_v:
        v_kwargs['v'] = (peakcount - np.mean(peakcount))/2
    else:
        v_kwargs['ind'] = ind
    Iguess_upper, Ibounds_upper = self.handle_I_GuessAndBounds(**Iu_kwargs)
    Iguess_lower, Ibounds_lower = self.handle_I_GuessAndBounds(**Il_kwargs)
    vguess, vbounds = self.handle_v_GuessAndBounds(**v_kwargs)
    if necessary_v:
        vguess[0] = 0
    # Refine guesses and bounds by fitting if necessary
    if necessary_Iu:
        Iguess_upper = self.fitI(points=peaks['i_global_pos'], guess=Iguess_upper, bounds=Ibounds_upper,
                                 const=Iconst, mod1f=Imod1f, mod2f=Imod2f, scan1f=Iscan1f, scan2f=Iscan2f,
                                 maxfev=maxfev)
        Iguess_upper, Ibounds_upper = self.handle_I_GuessAndBounds(ind=ind, guess=Iguess_upper, bounds=Ibounds_upper, using=Iusing)
    if necessary_Il:
        Iguess_lower = self.fitI(points=peaks['i_global_neg'], guess=Iguess_lower, bounds=Ibounds_lower,
                                 const=Iconst, mod1f=Imod1f, mod2f=Imod2f, scan1f=Iscan1f, scan2f=Iscan2f,
                                 maxfev=maxfev)
        Iguess_lower, Ibounds_lower = self.handle_I_GuessAndBounds(ind=ind, guess=Iguess_lower, bounds=Ibounds_lower, using=Iusing)
    if necessary_v:
        tpeakcount, peakcount = self.cumulativeCount(peaklist=self.peaksBetweenTurns(peaks, turns))
        vguess = self.fitV(tpeakcount, peakcount/2, guess=vguess, bounds=vbounds,
                           const=vconst, mod1f=vmod1f, mod2f=vmod2f, scan1f=vscan1f, scan2f=vscan2f,
                           maxfev=maxfev)
        vguess[0] = 0
        vguess, vbounds = self.handle_v_GuessAndBounds(ind=ind, guess=vguess, bounds=vbounds, using=vusing)
    if kguess is None:
        Iupper = IUpperEq(self.t[i_min:i_max], *Iguess_upper[Iusing])
        Ilower = ILowerEq(self.t[i_min:i_max], *Iguess_lower[Iusing])
        Inorm  = np.divide(self.I[i_min:i_max] - Ilower, Iupper - Ilower)
        kguess = self.fitK(ind, Inorm, vguess[vusing], vEq=vEq, deltak=deltak, klowerbound=klowerbound, maxfev=maxfev)
    kbounds_lower = np.array([kguess-deltak if kguess-deltak > klowerbound else klowerbound])
    kbounds_upper = np.array([kguess+deltak if kguess+deltak < 1 else 1])
    p0 = np.concatenate((Iguess_lower[Iusing],
                         Iguess_upper[Iusing],
                         vguess[vusing],
                         np.array([kguess])))
    bounds_lower = np.concatenate((Ibounds_lower[0][Iusing],
                                   Ibounds_upper[0][Iusing],
                                   vbounds[0][vusing],
                                   kbounds_lower))
    bounds_upper = np.concatenate((Ibounds_lower[-1][Iusing],
                                   Ibounds_upper[-1][Iusing],
                                   vbounds[-1][vusing],
                                   kbounds_upper))
    bounds = bounds_lower, bounds_upper
    self.checkBounds(p0, bounds)
    # Fit data
    if differentialevolution:
        debounds = [_ for _ in zip(*bounds)]
        popt_res = differential_evolution(objectivefunction, debounds)
        popt = popt_res.x
    else:
        popt, pcov = self.fitScaledData(self.t[i_min:i_max], self.I[i_min:i_max], mEq, p0, bounds, maxfev=maxfev)
    # Adjust resulting phases to range (0:2*pi)
    istart = 2 if Iconst else 1
    popt[istart:nI:2] = np.mod(popt[istart:nI:2], 2*np.pi)
    popt[nI+istart:2*nI:2] = np.mod(popt[nI+istart:2*nI:2], 2*np.pi)
    istart = 2 if vconst else 1
    popt[2*nI+istart:2*nI+nv:2] = np.mod(popt[2*nI+istart:2*nI+nv:2], 2*np.pi)
    # Pack results in with fixed parameter zeros
    results = np.nan_to_num(initialguesses)
    results[2*Iusing + vusing + [True]] = popt
    # Plot
    if plotflag:
        time = self.t[i_min:i_max]
        data = self.I[i_min:i_max]
        print(p0)
        initialestimate = mEq(time, *p0)
        fitresult = self.modelEquation(
                                time,
                                results[:9],
                                results[9:18],
                                results[18:27],
                                results[27])
        plotData([(time, data),
                  (time, initialestimate),
                  (time, fitresult),
                  ],
                 linestyles = ['k-', 'g--', 'r-'],
                 labels = ['data', 'initial estimate', 'fit result'],
                 title = 'fitModelSegment',
                 xlabel = 'Time (s)',
                 ylabel = 'Signal (V)')
        plotData([(time, initialestimate - data),
                  (time, fitresult - data),
                  ],
                 linestyles = ['go', 'ro'],
                 labels = ['initial estimate', 'fit result'],
                 title = 'Residuals of fitModelSegment',
                 xlabel = 'Time (s)',
                 ylabel = 'Residual')
    return {'ILowerArgs' : results[:9],
            'IUpperArgs' : results[9:18],
            'vArgs'      : results[18:27],
            'k'          : results[27],
            }

def fitI(self, points=None, ind=(0,None), guess=None, bounds=None, maxfev=1000, const=True, mod1f=True, mod2f=True, scan1f=True, scan2f=True, plotflag=False):
    """
    fitI(points=None, ind=(0,None), guess=None, bounds=None, mod1f=True, mod2f=True, scan1f=False, scan2f=True, plotflag=False)
    points = global indices of points to fit
    ind = (i_min, i_max) global indices of bounds to fit, default is all data
    guess = list of parameter guesses for IEquation, None entries are estimated from self.I data
    bounds = 2-tuple of list of bounds (lowerbounds, upperbounds), None entries estimated from data
    const, mod1f, mod2f, scan1f, scan2f = bools, terms to include or exclude from fit
    plotflag = bool
    Fits IEquation to the data determined by points or ind
    If list of points specified, fits only those points
    Otherwise fits data in range of ind
    Returns optimal fit parameters of IEquation
    """
    if guess is None:
        guess = [None]*9
    if bounds is None:
        bounds = ([None]*9, [None]*9)
    if len(guess) != 9:
        raise RuntimeError('guess input to fitI must be length 9.  Input was {}'.format(guess))
    using = [const] + 2*[mod1f] + 2*[mod2f] + 2*[scan1f] + 2*[scan2f]
    points = self.handle_points(points=points, ind=ind)
    # Create equation for fitting
    def IEq(t,*Iparams):
        allparams = np.zeros(9)
        allparams[using] = Iparams
        return self.IEquation(t, *allparams)
    # Calculate initial guesses and bounds
    p0, (bounds_lower, bounds_upper) = self.handle_I_GuessAndBounds(points=points, guess=guess, bounds=bounds, using=using)
    p0 = p0[using]
    bounds = bounds_lower[using], bounds_upper[using]
    self.checkBounds(p0, bounds)
    # Fit data
    popt, pcov = self.fitScaledData(self.t[points], self.I[points], IEq, p0, bounds, maxfev=maxfev)
    # Adjust resulting phases to range (0:2*pi)
    popt[2::2] = np.mod(popt[2::2], 2*np.pi)
    # Pack results in with fixed parameter zeros
    results = np.zeros(9)
    results[using] = popt
    # Plot
    if plotflag:
        time = self.t[points]
        data = self.I[points]
        fit  = self.IEquation(time,*results)
        plotData([(time, data),
                  (time, fit)],
                 linestyles = ['k-' if points is None else 'ko', 'r-'],
                 labels = ['data','fit'],
                 title = 'fitI',
                 xlabel = 'Time (s)',
                 ylabel = 'Signal (V)')
        plotData((time, fit-data),
                 linestyles = 'ro',
                 title = 'Residuals of fitI',
                 xlabel = 'Time (s)',
                 ylabel = 'Residual')
    return np.array(results)

def fitInterferenceEquation(self, ind, Inorm, vguess=None, vbounds=None, maxfev=1000, const=True, mod1f=True, mod2f=True, scan1f=True, scan2f=True, kguess=None, deltak=0.25, klowerbound=1E-6, plotflag=False):
    """
    preprocessEtalon.fitInterferenceEquation(ind, Inorm, vguess=None, vbounds=None, const=True, mod1f=True, mod2f=True, scan1f=True, scan2f=True, kguess=None, deltak=0.25, klowerbound=1E-6, plotflag=False)
    ind = (i_min, i_max) indices of period to fit
    I_norm = normalized data, to range (0:1)
    vguess = list of parameter guesses for vEquation, None entries will be estimated from peak counting
    vbounds = 2-tuple of list of bounds for vEquation (lowerbounds, upperbounds), None entries estimated from input v data
    const, mod1f, mod2f, scan1f, scan2f = bools, parameters to include / exclude from fit of vEquation
    kguess = float(0,1), Default estimate by fitting integrated areas to determine initial estimate
    deltak = float(0,1), +/- limit of search range around kguess, default 0.25
    klowerbound = minimum allowable k value. Default 1E-6. Nonzero lower bound avoids numerical instability
    plotflag = bool
    Fits interferenceEquation to (self.t[ind[0]:ind[1]], Inorm)
    Returns dictionary of parameters used in fit, with keys:
        k     = k parameter in interferenceEquation
        vargs = arguments to vEquation
    """
    if vguess is None:
        vguess = [None]*9
    if vbounds is None:
        vbounds = ([None]*9, [None]*9)
    if len(vguess) != 9:
        raise RuntimeError('vguess input to fitInterferenceEquation must be length 9.  Input was {}'.format(vguess))
    i_min, i_max = ind
    using = [const] + 2*[mod1f] + 2*[mod2f] + 2*[scan1f] + 2*[scan2f]
    # Create v equation for fitting
    def vEq(t,*vparams):
        allparams = np.zeros(9)
        allparams[using] = vparams
        return self.vEquation(t, *allparams)
    # Calculate initial guesses and bounds
    p0v, (vbounds_lower, vbounds_upper) = self.handle_v_GuessAndBounds(ind=ind, guess=vguess, bounds=vbounds, using=using)
    if kguess is None:
        kguess = self.fitK(ind, Inorm, p0v[using], vEq=vEq, deltak=deltak, klowerbound=klowerbound)
    kbounds_lower = np.array([kguess-deltak if kguess-deltak > klowerbound else klowerbound])
    kbounds_upper = np.array([kguess+deltak if kguess+deltak < 1 else 1])
    p0 = np.concatenate((np.array([kguess]), p0v[using]))
    bounds_lower = np.concatenate((kbounds_lower,
                                   vbounds_lower[using]))
    bounds_upper = np.concatenate((kbounds_upper,
                                   vbounds_upper[using]))
    bounds = bounds_lower, bounds_upper
    self.checkBounds(p0, bounds)
    # Fit data
    intEq = lambda t, k, *vparams : self.interferenceEquation(t, vparams, k, vEq=vEq)
    popt, pcov = self.fitScaledData(self.t[i_min:i_max], Inorm, intEq, p0, bounds, maxfev=maxfev)
    # Adjust resulting phases to range (0:2*pi)
    i_start = 3 if const else 2
    popt[i_start::2] = np.mod(popt[i_start::2], 2*np.pi)
    # Pack results in with fixed parameter zeros
    results = np.zeros(10)
    results[[True]+using] = popt
    # Plot
    if plotflag:
        time = self.t[i_min:i_max]
        initialestimate = intEq(self.t[i_min:i_max], p0[0], *p0[1:])
        fitresult = self.interferenceEquation(
                                self.t[i_min:i_max],
                                results[1:],
                                results[0])
        plotData([(time, Inorm),
                  (time, initialestimate),
                  (time, fitresult),
                  ],
                 linestyles = ['ko', 'g--', 'r-'],
                 labels = ['data', 'initial estimate', 'fit result'],
                 title = 'fitInterferenceEquation',
                 xlabel = 'Time (s)',
                 ylabel = 'Signal (V)')
        plotData([(time, initialestimate - Inorm),
                  (time, fitresult - Inorm),
                  ],
                 linestyles = ['go', 'ro'],
                 labels = ['initial estimate', 'fit result'],
                 title = 'Residuals of fitInterferenceEquation',
                 xlabel = 'Time (s)',
                 ylabel = 'Residual')
    return {'k'     : results[0],
            'vargs' : results[1:],
            }

def fitV(self, t, v, guess=None, bounds=None, maxfev=1000, const=True, mod1f=True, mod2f=True, scan1f=True, scan2f=True, plotflag=False):
    """
    preprocessEtalon.fitV(t, v, guess=None, bounds=None, const=True, mod1f=True, mod2f=True, scan1f=True, scan2f=True, plotflag=False)
    t = time array
    v = wavelength/number array
    guess = list of parameter guesses for vEquation, None entries are estimated from input v data
    bounds = 2-tuple of list of bounds (lowerbounds, upperbounds), None entries estimated from input v data
    const, mod1f, mod2f, scan1f, scan2f = bools, parameters to include / exclude from fit
    plotflag = bool
    Fits vEquation to (t,v)
    Returns optimal fit parameters of vEquation
    """
    if guess is None:
        guess = [None]*9
    if bounds is None:
        bounds = ([None]*9, [None]*9)
    if len(guess) != 9:
        raise RuntimeError('guess input to fitV must be length 9.  Input was {}'.format(guess))
    using = [const] + 2*[mod1f] + 2*[mod2f] + 2*[scan1f] + 2*[scan2f]
    # Create equation for fitting
    def vEq(t,*vparams):
        allparams = np.zeros(9)
        allparams[using] = vparams
        return self.vEquation(t, *allparams)
    # Calculate initial guesses and bounds
    p0, (bounds_lower, bounds_upper) = self.handle_v_GuessAndBounds(v=v, guess=guess, bounds=bounds, using=using)
    p0 = p0[using]
    bounds = bounds_lower[using], bounds_upper[using]
    self.checkBounds(p0, bounds)
    # Fit data
    popt, pcov = self.fitScaledData(t, v, vEq, p0, bounds, maxfev=maxfev)
    # Adjust resulting phases to range (0:2*pi)
    i_start = 2 if const else 1
    popt[i_start::2] = np.mod(popt[i_start::2], 2*np.pi)
    # Pack results in with fixed parameter zeros
    results = np.zeros(9)
    results[using] = popt
    # Plot
    if plotflag:
        plotData([(t,v), (t,self.vEquation(t,*results))],
                 linestyles=['ko', 'r-'],
                 labels=['data','fit'],
                 title='fitV',
                 xlabel='Time (s)',
                 ylabel='V_rel')
    return np.array(results)

def fitK(self, ind, I_norm, vArgs, vEq=None, kguess=None, deltak=0.25, klowerbound=1E-6, maxfev=1000, plotflag=False):
    """
    preprocessEtalon.fitK(ind, I_norm, vArgs, vEq=None, kguess=None, deltak=0.25, klowerbound=1E-6, plotflag=False)
    Fit interferenceEquation, for the k parameter, to normalized data using the given vEq and associated parameters in vArgs
    ind = (i_min, i_max) indices of period to fit
    I_norm = normalized data, to range (0:1)
    vArgs = parameters to pass to vEq, excluding time
    vEq = optional custom wavenumber function to use in interferenceEquation. Default self.vEquation
    kguess = float(0,1), default fits integrated areas to determine initial estimate
    deltak = float(0,1), +/- limit of search range around kguess, default 0.25
    klowerbound = minimum allowable k value. Default 1E-6. Nonzero lower bound avoids numerical instability
    plotflag = bool
    Returns optimal k
    """
    i_min, i_max = ind
    if vEq is None:
        vEq = self.vEquation
    if kguess is None:
        I_integrated = simps(I_norm, x=self.t[i_min:i_max])
        fitEq = lambda k : simps(self.interferenceEquation(self.t[i_min:i_max],
                                                           vArgs,
                                                           k,
                                                           vEq=vEq),
                                 x=self.t[i_min:i_max]) - I_integrated
        kopt = klowerbound if fitEq(klowerbound)<=0 else bisect(fitEq,klowerbound,1)
    else:
        if (kguess < 0) or (kguess > 1):
            raise ValueError('kguess in fitK must be in range (0:1). Input kguess was {}'.format(kguess))
        kguess = kguess if kguess > klowerbound else klowerbound
        lowerbound = kguess-deltak if kguess-deltak > klowerbound else klowerbound
        upperbound = kguess+deltak if kguess+deltak < 1 else 1
        scale = 1/np.mean(I_norm)
        fitEq = lambda t,k : self.interferenceEquation(t, vArgs, k, vEq=vEq) * scale
        kopt, kcov = curve_fit(fitEq,
                               self.t[i_min:i_max],
                               I_norm * scale,
                               p0 = kguess,
                               bounds = ([lowerbound], [upperbound]),
                               maxfev = maxfev)
        kopt = kopt[0]
        if plotflag:
            plotData([(self.t[i_min:i_max],I_norm), (self.t[i_min:i_max],fitEq(self.t[i_min:i_max],kopt))],
                     linestyles = ['k-','r--'],
                     labels = ['data','fit'],
                     xlabel = 'Time (s)',
                     title = 'fitK')
    return kopt

def IEquation(self, t, I0, Im1, phim1, Im2, phim2, Is1, phis1, Is2, phis2):
    """
    preprocessEtalon.IEquation(t, I0, Im1, phim1, Im2, phim2, Is1, phis1, Is2, phis2)
    t = time array
    I0 = mean intensity
    I*1, I*2 = 1st and 2nd harmonic * intensity amplitudes
    phi*1, phi*2 = 1st and 2nd harmonic * phases
    m = modulation
    s = scan
    Returns intensity array
    """
    return I0 + Im1*np.sin(2*np.pi*self.fm*t + phim1) \
              + Im2*np.sin(4*np.pi*self.fm*t + phim2) \
              + Is1*np.sin(2*np.pi*self.fs*t + phis1) \
              + Is2*np.sin(4*np.pi*self.fs*t + phis2)

def vEquation(self, t, v0, vm1, phim1, vm2, phim2, vs1, phis1, vs2, phis2):
    """
    preprocessEtalon.vEquation(t, v0, vm1, phim1, vm2, phim2, vs1, phis1, vs2, phis2)
    t = time array
    v0 = mean wavenumber/length
    v*1, v*2 = 1st and 2nd harmonic * wavenumber/length amplitudes
    phi*1, phi*2 = 1st and 2nd harmonic * phases
    m = modulation
    s = scan
    Returns wavenumber/length array
    """
    return v0 + vm1*np.sin(2*np.pi*self.fm*t + phim1) \
              + vm2*np.sin(4*np.pi*self.fm*t + phim2) \
              + vs1*np.sin(2*np.pi*self.fs*t + phis1) \
              + vs2*np.sin(4*np.pi*self.fs*t + phis2)

def dvdt(self, t, v0, vm1, phim1, vm2, phim2, vs1, phis1, vs2, phis2):
    """

    """
    return   2*np.pi*self.fm*vm1*np.cos(2*np.pi*self.fm*t + phim1) \
           + 4*np.pi*self.fm*vm2*np.cos(4*np.pi*self.fm*t + phim2) \
           + 2*np.pi*self.fs*vs1*np.cos(2*np.pi*self.fs*t + phis1) \
           + 4*np.pi*self.fs*vs2*np.cos(4*np.pi*self.fs*t + phis2)

def dv2dt2(self, t, v0, vm1, phim1, vm2, phim2, vs1, phis1, vs2, phis2):
    """

    """
    return - 2*np.pi*self.fm*2*np.pi*self.fm*vm1*np.sin(2*np.pi*self.fm*t + phim1) \
           - 4*np.pi*self.fm*4*np.pi*self.fm*vm2*np.sin(4*np.pi*self.fm*t + phim2) \
           - 2*np.pi*self.fs*2*np.pi*self.fs*vs1*np.sin(2*np.pi*self.fs*t + phis1) \
           - 4*np.pi*self.fs*4*np.pi*self.fs*vs2*np.sin(4*np.pi*self.fs*t + phis2)

def interferenceEquation(self, t, vArgs, k, vEq=None):
     """
     preprocessEtalon.interferenceEquation(t, vArgs, k, vEq=None)
     t = time array
     vArgs = list of arguments to pass to vEq
     k = interference parameter, float in range (0:1)
     vEq = optional custom wavenumber function to use in fit. Default self.vEquation
     Returns the Airy-like normalized interference array
     """
     if vEq is None:
         vEq = self.vEquation
     return (1-k)/(2*k) * (-1 + (1+k)/(1 - k*np.sin(2*np.pi * vEq(t,*vArgs))))

def modelEquation(self, t, ILowerArgs, IUpperArgs, vArgs, k, IEq=None, ILowerEq=None, IUpperEq=None, vEq=None):
    """
    preprocessEtalon.modelEquation(t, ILowerArgs, IUpperArgs, vArgs, k, IEq=None, vEq=None)
    t = time array
    ILowerArgs = arguments of intensity equation for lower intensity envelope
    IUpperArgs = arguments of intensity equation for upper intensity envelope
    vArgs = arguments of vEquation
    k = k parameter of interferenceEquation
    IEq = optional custom intensity envelope function. Default self.IEquation
    vEq = optional custom wavenumber function. Default self.vEquation
    Returns array of simulated signal
    """
    if IEq is None:
        IEq = self.IEquation
    if ILowerEq is None:
        ILowerEq = IEq
    if IUpperEq is None:
        IUpperEq = IEq
    if vEq is None:
        vEq = self.vEquation
    Ilower = ILowerEq(t, *ILowerArgs)
    Iupper = IUpperEq(t, *IUpperArgs)
    interference = self.interferenceEquation(t, vArgs, k, vEq=vEq)
    return interference * (Iupper - Ilower) + Ilower

assignMethodsToDatatype(Etalon)






#%%

def runTask(task):
    """ Wrapper for ScanSegment.analyze having single input to enable parallelization.
    Input a task (single element of output from EtalonSegment.setupTasks).
    Each task in list is tuple of (taskindex, i_etstart, i_etend, ScanSegment object, vs_fit, plotflag).
    i_etstart and i_etend are indices within the parent EtalonSegment object defining the ScanSegment object.
    Returns taskindex, (i_etstart, i_etend), (i_firstmodturn, i_lastmodturn), v, ScanSegment. v is wavelength, i_*modturn are locations of extreme modulation turn points. """
    itask, i_etstart, i_etend, scanseg, vs_fit, testrun, plotflag = task
    print('Running task {}'.format(itask))
    v, i_modturns = scanseg.analyze(vs_fit=vs_fit,testrun=testrun,plotflag=plotflag)
    print('Finished task {}'.format(itask))
    return itask, (i_etstart, i_etend), i_modturns, v, scanseg

def runTasksSerial(tasks):
    """ Run the list of tasks in order serially.
    Input the list of tasks output from setupTasks().
    Return [taskresult, ...]  List of taskresults, each having tuple of (itask, i_et, i_modturns, v, scanseg)"""
    return [runTask(task) for task in tasks]

class EtalonSegment(Etalon):
    """ Segment or entirety of data object from Etalon data. """

    et_deviation_cutoff = 0.5
    et_neighboring_peaks = 3
    kfit_error_estimate = 0.15
    alpha_default = 0.99

    def __init__(self,parentobj,ind=None):
        """ Initialize with variables from parent object.
        ind defaults to whole range of parent object.
        Input parentobj, ind=(i1,i2).
        Return nothing. """
        members, values = parentobj.exportConstants()
        for member, value in zip(members,values):
            setattr(self,member,value)
        if ind is None:
            self.t = parentobj.t
            self.I = parentobj.I
        else:
            i1, i2 = ind
            if np.abs(i2-i1) < self.ns:
                raise ValueError('Invalid indices for preprocess.EtalonSegment(ind=[{},{}]). Minimum index range is self.ns={}'.format(i1,i2,self.ns))
            self.t = parentobj.t[i1:i2]
            self.I = parentobj.I[i1:i2]
            t_range = self.t[-1]-self.t[0]
            self.Ns = t_range*self.fs
            self.Nm = t_range*self.fm if isinstance(self.fm,(int,float)) else [t_range*x for x in self.fm]
        self.v = np.zeros(len(self.t))

    def setupTasks(self,vs_fit=None,testrun=False,plotflag=None):
        """ Create tasks for use with preprocess.runTask(), to process EtalonSegment object split into ScanSegment objects, which can be parallelized.
        Input vs_fit=(popt,pcov), testrun=bool, plotflag=bool.  vs_fit is output from EtalonSegment.estimateVs.
        Default is to run EtalonSegment.estimateVs() if None given, and None plotflag which uses defaults in estivateVs and False elsewhere.
        testrun=True fits only the initial modulation segment in each scan segment.
        Returns tasks.  List of tuples [(taskindex, iet_start, iet_end, scansegmentobject, vs_fit, plotflag)]. """
        n = len(self.t)
        l = int(np.round(self.ns/2)) + 2*self.nm
        di = int(np.round(self.ns/2))
        if vs_fit is None:
            if plotflag is not None:
                vs_fit = self.estimateVs(plotflag=plotflag)
            else:
                vs_fit = self.estimateVs()
        iturn1 = self.findTurnIndex(vs_fit)
        i_end1 = iturn1+self.nm if iturn1+self.nm>self.ns/2 else iturn1+di+self.nm
        i_end_list = np.concatenate((np.arange(i_end1, n-di-self.nm, di), np.array([n])))
        i_start_list = np.concatenate((np.array([0]), i_end_list[1:-1]-l, np.array([i_end_list[-2]-2*self.nm])))
        tasks = [(itask, ind[0], ind[1], ScanSegment(self,ind=[ind[0],ind[1]]), vs_fit, testrun, plotflag) for itask, ind in enumerate(zip(i_start_list,i_end_list))]
        print('Etalon Tasks Setup Complete')
        return tasks

    def estimateVs(self,plotflag=True):
        """ Get first-order wavelength scan parameters vs and phivs of ScanSegment.estimateVs in first scan period.
        Input plotflag=bool.  Default is plot.
        Returns (popt,pcov) of fit parameters [vs,phivs]. """
        return ScanSegment(self,ind=[0,self.ns]).estimateVs(plotflag=plotflag)

    def findTurnIndex(self,vs_fit):
        """ Estimate the scan wavelength turn index from vs_fit.
        Input vs_fit=array of wavelength parameters from estimateVs.
        Return iturn. """
        phivs = vs_fit[0][1]
        t0 = np.mod(self.t[0], self.Ts/2)
        tturn = np.mod((np.pi/2 - phivs) / (2*np.pi*self.fs), self.Ts/2)
        iturn = int(np.round(np.mod(tturn - t0, self.Ts/2) * self.SR))
        return iturn

    def stitchV(self,i_et,i_scan,v_scan,append=True):
        """ Stitch wavelength of scan segment into etalon segment.
        Input i_et, i_scan, v_scan, append=bool.
        append defines the direction to stitch into the etalon segment, True=append / False=prepend (i.e. forward/backward in time).
        i_et defines where to begin stitching data into the etalon segment, first point if appending or last point if prepending.
        i_scan defines the segment of v_scan to use, either [i_scan:] if appending or [:i_scan+1] if prepending.
        v_scan is an array of relative wavelengths over the entire scan segment.
        Returns nothing. """
        data = v_scan[i_scan:] if append else v_scan[:i_scan+1]
        n = len(data)
        if append:
            self.v[i_et:i_et+n] = data - data[0] + self.v[i_et]
        else:
            self.v[i_et-n+1:i_et+1] = data - data[-1] + self.v[i_et]

class ScanSegment(Etalon):
    """ Scan segment of a data object. """

    def __init__(self,parentobj,ind=None):
        """ Initialize with variables from parent object.
        ind defaults to whole range of parent object.
        Input parentobj, ind=(i1,i2).
        Return None. """
        members, values = parentobj.exportConstants()
        for member, value in zip(members,values):
            setattr(self,member,value)
        if ind is None:
            self.t = parentobj.t
            self.I = parentobj.I
        else:
            i1, i2 = ind
            if np.abs(i2-i1) < self.ns/2:
                raise ValueError('Invalid indices for preprocess.ScanSegment(ind=[{},{}]). Minimum index range is self.ns/2={}'.format(i1,i2,self.ns/2))
            self.t = parentobj.t[i1:i2]
            self.I = parentobj.I[i1:i2]
            t_range = self.t[-1]-self.t[0]
            self.Ns = t_range*self.fs
            self.Nm = t_range*self.fm if isinstance(self.fm,(int,float)) else [t_range*x for x in self.fm]
        self.v = np.zeros(len(self.t))

    def analyze(self,vs_fit=None,testrun=False,plotflag=False):
        """ Solve for the relative wavelength vs time over the scan segment.
        Input vs_fit=(popt,pcov), testrun=bool, plotflag=bool.  testrun=True stops after initial modulation segment fit.
        Solves sub-segments of length self.nm with a step of self.nm/2.  End segments may be longer length.
        Then stitches together wavelength starting from one extreme modulation wavelength turn point in sub-segments.
        Returns self.v, (i_firstmodturn, i_lastmodturn) """
        n = len(self.t)
        di = int(np.round(self.nm/2))
        iturn1, tturn1 = self.findExtremeModulationTurn()
        i_end1 = int(np.round(iturn1 + self.nm/4))
        while i_end1 < self.nm:
            i_end1 += di
        while i_end1 > 1.5*self.nm:
            i_end1 -= di
        self.i_end_list = np.concatenate((np.arange(i_end1, n-di, di), np.array([n])))
        self.i_start_list = np.concatenate((np.array([0]), self.i_end_list[1:-1]-self.nm, np.array([self.i_end_list[-3]])))
        nseg = len(self.i_end_list)
        self.popt_list, self.ci_list, self.r2_list = [], [], []
        if vs_fit is None:
            vs_fit = self.estimateVs(plotflag=plotflag)
        p0 = None
        # Analyze sub-segments in reverse order
        for subseg, ind in reversed(list(enumerate(zip(self.i_start_list,self.i_end_list)))):
            modseg = ModulationSegment(self,ind=[ind[0],ind[1]])
            if (ind[0] == 0) and (self.config['ignorefirstmod']):  # Do not try to fit initial segment while lasers are not in equilibrium
                ci = np.NaN
                r2 = np.NaN
            else:
                if plotflag is not None:
                    p0, ci, r2 = modseg.analyze(p0=p0, vs_fit=vs_fit, plotIfFromScratch=plotflag, plotflag=plotflag)
                else:
                    p0, ci, r2 = modseg.analyze(p0=p0, vs_fit=vs_fit)
                    plotflag = False
            self.popt_list.append(p0)
            self.ci_list.append(ci)
            self.r2_list.append(r2)
            popt_v = modseg.getVparams(p0)
            v = modseg.calcV(popt_v,plotflag=plotflag)
            if subseg == nseg-1:
                self.v[ind[0]:ind[1]] = v
            else:
                i_mod = modseg.findExtremeTurnIndex(popt_v,first=False)
                self.stitchV(ind[0]+i_mod, i_mod, v, append=False)
            i_firstmodturn = ind[0] + modseg.findExtremeTurnIndex(popt_v,first=True) if subseg==0 else None
            i_lastmodturn  = ind[0] + modseg.findExtremeTurnIndex(popt_v,first=False) if subseg==nseg-1 else None
            if testrun:
                break
        self.popt_list = self.popt_list[::-1]
        self.ci_list = self.ci_list[::-1]
        self.r2_list = self.r2_list[::-1]
        if plotflag:
            plotData([(self.t,self.v)], linestyles=['k-'], title='ScanSegment.analyze', xlabel='Time (s)', ylabel='Relative Wavelength * 2$\pi$/FSR')
        return self.v, (i_firstmodturn, i_lastmodturn)

    def estimateVs(self,plotflag=False):
        """ Estimate first-order wavelength scan parameters vs and phivs.
        Input plotflag=bool.  Default is no plot.
        Returns (popt,pcov) of fit parameters [vs,phivs]. """
        if plotflag is None:
            plotflag = True
        iturn1, _ = self.findExtremeModulationTurn()
        i_start = int(np.round(iturn1 + self.nm/4))
        tcumsum = np.array([])
        cumsum = np.array([])
        ttoppeaks = np.array([])
        Itoppeaks = np.array([])
        sgn = 1
        for imod in range(int(self.modsperscan)):
            i1 = i_start+int(np.round(self.nm/2))*imod
            i2 = i_start+int(np.round(self.nm/2))*(imod+2)
            modseg = ModulationSegment(self,ind=[i1,i2])
            peakdata = modseg.findValidPeakIndices()
            ipeaks, tpeaks = modseg.peaksBetweenTurns(peakdata=peakdata)
            if len(ipeaks) != 3:
                modseg.findValidPeakIndices(plotflag=True)
                raise Warning('Non-3 sections of modulation segment found for ind=[{},{}]'.format(i1,i2))
            prevsum = cumsum[-1] if cumsum.size != 0 else 0
            cumsum = np.append(cumsum,prevsum+sgn*np.pi*np.arange(len(ipeaks[1])))
            tcumsum = np.append(tcumsum,tpeaks[1])
            sgn = -sgn
            itoppeaks = peakdata[0][0][0]
            ttoppeaks = np.append(ttoppeaks,modseg.t[itoppeaks])
            Itoppeaks = np.append(Itoppeaks,modseg.I[itoppeaks])
        guess = [np.max(cumsum)-np.min(cumsum), 0, np.mean(cumsum)]
        bound = ([0,-2*np.pi,-np.inf], [np.inf,2*np.pi,np.inf])
        fitEq = lambda t,vs,phivs,v0 : v0 + vs*np.sin(2*np.pi*self.fs*t+phivs)
        popt, pcov = curve_fit(fitEq,tcumsum,cumsum,p0=guess,bounds=bound)
        popt[1] = np.mod(popt[1],2*np.pi)
        # Fit intensity to check phase relation (wavelength lags intensity by pi<lag<2*pi)
        Iguess = [(np.max(Itoppeaks)-np.min(Itoppeaks))/2, 0, np.mean(Itoppeaks)]
        Ibound = ([0, -2*np.pi, -np.inf], [np.inf, 2*np.pi, np.inf])
        IfitEq = lambda t,Is,phiIs,I0 : I0 + Is*np.sin(2*np.pi*self.fs*t+phiIs)
        Iopt, _ = curve_fit(IfitEq,ttoppeaks,Itoppeaks,p0=Iguess,bounds=Ibound)
        Iopt[1] = np.mod(Iopt[1],2*np.pi)
        relphase = np.mod(popt[1]-Iopt[1],2*np.pi)  # phivs - phiIs
        print('Relative wavelength-intensity scan phase = {} pi.'.format(relphase/np.pi))
        if relphase > np.pi:
            print('Shifting phivs by pi and flipping cumulative sum because relative wavelength-intensity scan phase > pi.')
            popt[1] = np.mod(popt[1]+np.pi,2*np.pi)
            popt[2] = -popt[2]
            cumsum = -cumsum
        if plotflag:
            plotData([(tcumsum,cumsum),(tcumsum,fitEq(tcumsum,*popt))], linestyles=['+','r-'], labels=['data','fit'], title='estimateVs', xlabel='Time (s)', ylabel='Cumulative Peak Count * $\pi$')
            plotData([(self.t[:self.ns],self.I[:self.ns]),(ttoppeaks,Itoppeaks)], linestyles=['-','r+'], labels=['data','upper peaks'], title='estimateVs', xlabel='Time (s)', ylabel='Signal (V)')
        return popt[:-1], pcov[:-1,:-1]

    def findExtremeModulationTurn(self,first=True,plotflag=False):
        """ Find extreme (first or last) modulation wavelength turn in the scan segment.
        Input first=bool, plotflag=bool.  Default is to find the first turn, no plot.
        Returns (i_turn,t_turn). """
        i_range = [0,self.nm] if first else [-self.nm,len(self.t)]
        seg = ModulationSegment(self,i_range)
        _, iturns_all = seg.findTurnPeaks(seg.findAllPeakIndices(i_guess=seg.findAllPeakIndices()[0])[:2])
        iturn = np.min(np.concatenate(iturns_all)) if first else np.max(np.concatenate(iturns_all))
        tturn = seg.t[iturn]
        return iturn, tturn

    def stitchV(self,i_scan,i_mod,v_mod,append=True):
        """ Stitch wavelength of modulation segment into scan segment.
        Input i_scan, i_mod, v_mod, append=bool.
        append defines the direction to stitch into the scan, True=append / False=prepend (i.e. forward/backward in time).
        i_scan defines where to begin stitching data into the scan, first point if appending or last point if prepending.
        i_mod defines the segment of v_mod to use, either [i_mod:] if appending or [:i_mod+1] if prepending.
        v_mod is an array of relative wavelengths over the entire modulation segment.
        Returns nothing. """
        data = v_mod[i_mod:] if append else v_mod[:i_mod+1]
        n = len(data)
        if append:
            self.v[i_scan:i_scan+n] = data - data[0] + self.v[i_scan]
        else:
            self.v[i_scan-n+1:i_scan+1] = data - data[-1] + self.v[i_scan]

class ModulationSegment(Etalon):
    """ Modulation segment of a data object. """

    et_deviation_cutoff = EtalonSegment.et_deviation_cutoff
    et_neighboring_peaks = EtalonSegment.et_neighboring_peaks
    kfit_error_estimate = EtalonSegment.kfit_error_estimate
    alpha_default = EtalonSegment.alpha_default

    def __init__(self,parentobj,ind=None):
        """ Initialize with variables from parent object.  parentobj is readData.Data, EtalonSegment, or ScanSegment
        ind=(i1,i2) ints. Defaults to [0,-1], entire range of parent object. Min ind[1]-ind[0] is parentobj.nm, number of datapoints per modulation.
        Return None. """
        members, values = parentobj.exportConstants()
        for member, value in zip(members,values):
            setattr(self,member,value)
        if ind is None:
            self.t = parentobj.t
            self.I = parentobj.I
        else:
            i1, i2 = ind
            if np.abs(i2-i1) < self.nm:
                raise ValueError('Invalid indices for preprocess.ModulationSegment(ind=({},{})). Minimum range is parentobj.nm={}'.format(i1,i2,self.nm))
            self.t = parentobj.t[i1:i2]
            self.I = parentobj.I[i1:i2]
            t_range = self.t[-1]-self.t[0]
            self.Ns = t_range*self.fs
            self.Nm = t_range*self.fm if isinstance(self.fm,(int,float)) else [t_range*x for x in self.fm]

    def analyze(self,p0=None,bounds=None,vs_fit=None,plotIfFromScratch=True,plotflag=False):
        """ Input p0=array, bounds=(lower,upper), vs_fit=(popt,pcov), plotIfFromScratch=bool, plotflag=bool.
        p0 are initial parameter guesses.  Bounds is tuple of lower bounds, upper bounds arrays.
        vs_fit (popt,pcov) are fit results for parameters [vs,phivs] from ScanSegment.estimateVs.
        plotIfFromScratch plots all calculation steps if no p0 given
        Default p0=None calculates initial guesses from scratch.
        Default bounds=None calculates custom bounds based on p0 values.
        For standard bounds from scratch use bounds=[anything but "None"], e.g. bounds=0
        Default is no plotting unless calculating from scratch.
        Return (popt, ci, r2).  Fit results from fitAll. """
        if p0 is None:
            if plotIfFromScratch:
                plotflag = True
            peaks, turns = self.findValidPeakIndices(plotflag=plotflag)
            peakCountFit = self.fitPeakCount(peaks,turns,vs_fit=vs_fit,plotflag=plotflag)
            peakIfits = self.fitPeaksI(peaks[0],plotflag=plotflag)
            vFit = self.fitV(peakIfits,peakCountFit,plotflag=plotflag)
            p0, ci0 = self.collectFits(peakIfits,vFit)
            if bounds is not None:
                bounds = ci0
        p, ci, r2 = self.fitAll(p0,bounds=bounds,plotflag=plotflag)
        return p, ci, r2

    def findValidPeakIndices(self,plotflag=False):
        """ Find indices of valid peaks in self.t,self.I data.  Excludes wavelength turns from found peaks.
        Return (((ipeaks_pos, ipeaks_neg), ipeaks_sorted), ((iturns_pos, iturns_ng), iturns_sorted)).
        i*_pos and i*_neg are indices of upper and lower peaks or turns.  i*_sorted are concatenated & sorted i*_pos & i*_neg."""
        ipeaks, dev, params = self.findAllPeakIndices(i_guess=self.findAllPeakIndices()[0])
        iiturns, iturns = self.findTurnPeaks((ipeaks,dev))
        ipeaks, dev, params = self.findAllPeakIndices(i_guess=self.removeTurnPeaks(ipeaks,iiturns)[0])

        """
        ipeaks = list(ipeaks)
        dev = list(dev)
        params = list(params)
        for i_type in range(2):
            to_delete = []
            widths = params[i_type]['widths']
            for i_width in range(1,len(widths)-1):
                if (widths[i_width] < widths[i_width-1]) and (widths[i_width] < widths[i_width+1]):
                    if widths[i_width] < (1/2) * np.min([widths[i_width-1], widths[i_width+1]]):
                        to_delete.append(i_width)
            for i_width in reversed(to_delete):
                ipeaks[i_type] = np.delete(ipeaks[i_type],i_width)
                dev[i_type] = np.delete(dev[i_type],i_width)
                for key in params[i_type].keys():
                    params[i_type][key] = np.delete(params[i_type][key],i_width)
        """

        iiturns, iturns = self.findTurnPeaks((ipeaks,dev),plotflag=plotflag)
        ipeaks, ipeaks_sorted = self.removeTurnPeaks(ipeaks,iiturns)
        iturns_sorted = np.sort(np.concatenate(iturns))
        return (ipeaks, ipeaks_sorted), (iturns, iturns_sorted)

    def findAllPeakIndices(self,prominence=None,i_guess=None,plotflag=False):
        """ Find indices of all peaks in self.t,self.I data.
        prominence=float in range (0:2), min prominence for peak finding from "normalized" data.  Defaults to self.et_deviation_cutoff.
        i_guess=(i_upper,i_lower) int arrays, upper and lower peak indices to fit IEquation for normalization.
        Returns ((i_pos, i_neg), (height_pos, height_neg), (params_pos, params_neg)).  i_* are peak indices.  height_* are 'heights' of normalized peaks relative to -1. """
        if prominence is None:
            prominence = self.et_deviation_cutoff
        if i_guess is None:
            popt, _ = self.fitI()
            I_norm = np.divide(self.I, self.IEquation(self.t,*popt)) - 1
            I_norm = np.divide(I_norm,2*np.median(np.abs(I_norm)))
            plot_I_fits = False
        else:
            i_pos, i_neg = i_guess
            popt_pos, _ = self.fitI(tI=(self.t[i_pos],self.I[i_pos]))
            popt_neg, _ = self.fitI(tI=(self.t[i_neg],self.I[i_neg]))
            I_fit_pos = self.IEquation(self.t,*popt_pos)
            I_fit_neg = self.IEquation(self.t,*popt_neg)
            I_norm = 2*np.divide((self.I-I_fit_neg),(I_fit_pos-I_fit_neg)) - 1
            plot_I_fits = True
        i_pos, param_pos = find_peaks( I_norm,prominence=prominence,width=0)
        i_neg, param_neg = find_peaks(-I_norm,prominence=prominence,width=0)
        heights_pos = I_norm[i_pos] - -1
        heights_neg = 1 - I_norm[i_neg]
        if plotflag:
            xy = [(self.t,self.I),(self.t[i_pos],self.I[i_pos]),(self.t[i_neg],self.I[i_neg])]
            styles = ['k-','r^','rv']
            labels = ['data','upper peaks','lower peaks']
            if plot_I_fits:
                xy.extend([(self.t,I_fit_pos),(self.t,I_fit_neg)])
                styles.extend(['g-','g--'])
                labels.extend(['upper peak fit','lower peak fit'])
            else:
                xy.extend([(self.t,self.IEquation(self.t,*popt))])
                styles.extend(['g-'])
                labels.extend(['single IEquation fit'])
            plotData(xy,linestyles=styles,labels=labels,title='findAllPeakIndices',xlabel='Time (s)',ylabel='Signal (V)')
        return (i_pos, i_neg), (heights_pos, heights_neg), (param_pos, param_neg)

    def findTurnPeaks(self,peakdata,plotflag=False):
        """ Find peaks corresponding to wavelength reversals in peakdata.
        peakdata is output from findAllPeakIndices.
        Returns ((iiturns_pos, iiturns_neg), (iturns_pos, iturns_neg)).  Where iturns_* = peakdata[0][*][iiturns_*]. """
        ipeaks, heights = peakdata
        ipeaks_pos, ipeaks_neg = ipeaks
        heights_pos, heights_neg = heights
        t_pos = self.t[ipeaks_pos]
        t_neg = self.t[ipeaks_neg]
        data = {'dt_central'  :   np.array([t3-t1 for t1,t3 in zip(t_pos[:-2],t_pos[2:])] + [t3-t1 for t1,t3 in zip(t_neg[:-2],t_neg[2:])]),
                't'           :   np.concatenate((t_pos[1:-1],t_neg[1:-1])),
                'height'      :   np.concatenate((heights_pos[1:-1],heights_neg[1:-1])),
                'dt_forward'  :   np.array([t2-t1 for t1,t2 in zip(t_pos[1:-1],t_pos[2:])] + [t2-t1 for t1,t2 in zip(t_neg[1:-1],t_neg[2:])])}
        df = pd.DataFrame.from_dict(data)
        df.sort_values('t',inplace=True)
        df.reset_index(drop=True,inplace=True)
        npeaks = len(df)
        data2 = {'ipeaks'     : np.concatenate(ipeaks),
                't'           : self.t[np.concatenate(ipeaks)],
                'height'      : np.concatenate(heights)}
        df2 = pd.DataFrame.from_dict(data2)
        df2.sort_values('t',inplace=True)
        df2.reset_index(drop=True,inplace=True)
        # Find time of one turn peak in the whole time series with the smallest height or largest central time gap if peak were removed
        if pd.Series.min(df['height']) < 2-self.et_deviation_cutoff:
            t_init = df.loc[pd.Series.idxmin(df['height']),'t']
        else:
            t_init = df.loc[pd.Series.idxmax(df['dt_central']),'t']
        # Define time intervals based on this peak to search for all peaks which lie within a complete half-modulation-period centered on times of t_init +/- n*(Tm/2)
        t_end1 = t_init + self.Tm/4
        while t_end1-self.Tm/2 > self.t[0]+self.Tm/4:
            t_end1 -= self.Tm/2
        t_end_list = np.concatenate((np.arange(t_end1,self.t[-1]-self.Tm/4,self.Tm/2), np.array([self.t[-1]])))
        t_start_list = np.concatenate((np.array([self.t[0]]), t_end_list[:-1]))
        nseg = len(t_end_list)
        # Find indices in dataframes of turn peaks
        iiturns = []
        for i, t_endpoints in enumerate(zip(t_start_list,t_end_list)):
            t_start, t_end = t_endpoints
            i_inrange = np.where((df['t']>=t_start) & (df['t']<t_end))[0]
            i_inrange2 = np.where((df2['t']>=t_start) & (df2['t']<t_end))[0]
            ii_maxcentral = pd.Series.idxmax(df.loc[i_inrange,'dt_central']) if len(i_inrange)>0 else None
            if (len(i_inrange2)>0) and (pd.Series.min(df2.loc[i_inrange2,'height']) < 2-self.et_deviation_cutoff):
                iiturns.append(pd.Series.idxmin(df2.loc[i_inrange2,'height']))
            elif (ii_maxcentral is not None) and ((i!=0 and i!=nseg-1) or (i==0 and ii_maxcentral>1) or (i==nseg-1 and ii_maxcentral<npeaks-2)):
                ii_check = np.arange(ii_maxcentral-self.et_neighboring_peaks, ii_maxcentral+self.et_neighboring_peaks+1)
                ii_check = ii_check[(ii_check>=0) & (ii_check<npeaks)]
                df_3sorted = df.loc[ii_check].sort_values('dt_forward',ascending=False).iloc[0:3].sort_values('t')
                iiturns_3sorted = [df_3sorted.iloc[i_3].name for i_3 in range(3)]
                diiturns_3sorted = np.abs(np.diff(iiturns_3sorted))
                iiturn = iiturns_3sorted[-1]
                if diiturns_3sorted[0]!=1 or diiturns_3sorted[1]!=1:
                    xy = [(self.t,self.I),(self.t[ipeaks_pos],self.I[ipeaks_pos]),(self.t[ipeaks_neg],self.I[ipeaks_neg])]
                    linestyles = ['k-','g^','gv']
                    labels = ['data','upper peaks','lower peaks']
                    plotData(xy, linestyles=linestyles, labels=labels, title='errorPeaks', xlabel='Time (s)', ylabel='Signal (V)')
                    print('Non-consecutive top 3 dt_central found in findTurnPeaks at indices {}, time = {}'.format(iiturn+2,df_3sorted['t'].values[1]))
                iiturns.append(iiturn+2)
        # Filter the peaks into positive and negative peak lists
        iiturns_pos = [np.where(ipeaks_pos==df2.loc[iiturn,'ipeaks'])[0][0] for iiturn in iiturns if df2.loc[iiturn,'ipeaks'] in ipeaks_pos]
        iiturns_neg = [np.where(ipeaks_neg==df2.loc[iiturn,'ipeaks'])[0][0] for iiturn in iiturns if df2.loc[iiturn,'ipeaks'] in ipeaks_neg]
        iturns_pos  = ipeaks_pos[iiturns_pos]
        iturns_neg  = ipeaks_neg[iiturns_neg]
        if plotflag:
            popt_pos,pcov_pos = self.fitI(tI=(self.t[ipeaks_pos],self.I[ipeaks_pos]))
            popt_neg,pcov_neg = self.fitI(tI=(self.t[ipeaks_neg],self.I[ipeaks_neg]))
            tupper = np.linspace(self.t[ipeaks_pos[0]],self.t[ipeaks_pos[-1]])
            tlower = np.linspace(self.t[ipeaks_neg[0]],self.t[ipeaks_neg[-1]])
            Iupper = self.IEquation(tupper,*popt_pos)
            Ilower = self.IEquation(tlower,*popt_neg)
            xy = [(self.t,self.I),(tupper,Iupper),(tlower,Ilower),(self.t[ipeaks_pos],self.I[ipeaks_pos]),(self.t[ipeaks_neg],self.I[ipeaks_neg]),(self.t[iturns_pos],self.I[iturns_pos]),(self.t[iturns_neg],self.I[iturns_neg])]
            linestyles = ['k-','r-','r--','g^','gv','b^','bv']
            labels = ['data','upper envelope','lower envelope','upper peaks','lower peaks','upper turns','lower turns']
            plotData(xy, linestyles=linestyles, labels=labels, title='findTurnPeaks', xlabel='Time (s)', ylabel='Signal (V)')
        return (iiturns_pos, iiturns_neg), (iturns_pos, iturns_neg)

    def removeTurnPeaks(self,ipeaks,iiturns):
        """ Remove the turn peaks from lists of peak indices.
        ipeaks=(ipeaks_pos,ipeaks_neg) includes all peaks
        iiturns=(iiturns_pos,iiturns_neg), so removed will be ipeaks_*[iiturns_*].
        Returns ((peaks_pos, peaks_neg), peaks_all_sorted). """
        peaks_pos = np.delete(ipeaks[0],iiturns[0])
        peaks_neg = np.delete(ipeaks[1],iiturns[1])
        peaks_all = np.sort(np.concatenate((peaks_pos,peaks_neg)))
        return (peaks_pos, peaks_neg), peaks_all

    def fitAll(self,p0,bounds=None,alpha=alpha_default,plotflag=False):
        """ Fit full etalon model to the segment data: IEquation to upper and lower peaks, and interferenceEquation with vEquation*.
        Input p0, bounds=(lower,upper), alpha=float(0:1), plotflag=bool. p0 are initial parameter guesses.
        Defaults to bounds=custom bounds based on p0, alpha=alpha_default class variable, and no plot.
        Returns (popt,confint,r2).  popt are resulting fit parameters, confint is confidence interval from fit calculated with alpha, r2 is goodness of fit. """
        p0_len = [15, 17]
        if bounds is None:
            Imax = max(self.I)
            lowerbounds, upperbounds = np.zeros(len(p0)), np.zeros(len(p0))
            lowerbounds[3:5]   = p0[3:5]   - np.pi/4  # upper peaks intensity fit phases
            lowerbounds[8:10]  = p0[8:10]  - np.pi/4  # lower peaks intensity fit phases
            lowerbounds[13:15] = p0[13:15] - np.pi/4  # vEquation fit phases
            upperbounds[0] = Imax                     # upper peaks I0
            upperbounds[1] = Imax/2                   # upper peaks Im
            upperbounds[2] = Imax/2                   # upper peaks Im2
            upperbounds[3:5] = p0[3:5] + np.pi/4      # upper peaks intensity fit phases
            upperbounds[5] = Imax                     # lower peaks I0
            upperbounds[6] = Imax/2                   # lower peaks Im
            upperbounds[7] = Imax/2                   # lower peaks Im2
            upperbounds[8:10] = p0[8:10] + np.pi/4    # lower peaks intensity fit phases
            upperbounds[10] = 1                       # k
            upperbounds[11:13] = 1.5*p0[11:13]        # vm & vs
            upperbounds[13:15] = p0[13:15] + np.pi/4  # vEquation fit phases
            if p0.size == p0_len[1]: # vm2 & phivm2
                lowerbounds[16] = p0[16] - np.pi
                upperbounds[15] = p0[11]
                upperbounds[16] = p0[16] + np.pi
            bounds = (lowerbounds,upperbounds)
        if p0.size == p0_len[0]:
            fitEq = lambda t,I0p,Imp,Im2p,phiImp,phiIm2p,I0n,Imn,Im2n,phiImn,phiIm2n,k,vm,vs,phivm,phivs            : self.IEquation(t,I0n,Imn,Im2n,phiImn,phiIm2n) + np.multiply( self.IEquation(t,I0p,Imp,Im2p,phiImp,phiIm2p) - self.IEquation(t,I0n,Imn,Im2n,phiImn,phiIm2n) , self.interferenceEquation(t,k,self.vEquation,[vm,vs,phivm,phivs]) )
        elif p0.size == p0_len[1]:
            fitEq = lambda t,I0p,Imp,Im2p,phiImp,phiIm2p,I0n,Imn,Im2n,phiImn,phiIm2n,k,vm,vs,phivm,phivs,vm2,phivm2 : self.IEquation(t,I0n,Imn,Im2n,phiImn,phiIm2n) + np.multiply( self.IEquation(t,I0p,Imp,Im2p,phiImp,phiIm2p) - self.IEquation(t,I0n,Imn,Im2n,phiImn,phiIm2n) , self.interferenceEquation(t,k,self.vEquation_SecondOrder,[vm,vs,phivm,phivs,vm2,phivm2]) )
        else:
            raise ValueError('Unknown length of p0 entered to preprocess.ModulationSegment.fitAll(). len(p0)={} Needs to be either {} or {}'.format(p0.size,p0_len[0],p0_len[1]))
        try:
            warnings.simplefilter("ignore",category=RuntimeWarning)
            popt, pcov  = curve_fit(fitEq,self.t,self.I,p0=p0,bounds=bounds)
        except RuntimeError:
            print('Optimal parameters not found for t = {} - {} from fitAll.  Using p0.'.format(self.t[0],self.t[-1]))
            popt = p0
            pcov = np.zeros((p0.size,p0.size))
        popt[3:5]   = np.mod(popt[3:5],2*np.pi)
        popt[8:10]  = np.mod(popt[8:10],2*np.pi)
        popt[13:15] = np.mod(popt[13:15],2*np.pi)
        if p0.size == p0_len[1]:
            popt[-1] = np.mod(popt[-1],2*np.pi)
        confint_fit = self.calcCI((popt,pcov),alpha=alpha)
        r2 = self.calcR2(self.I,fitEq(self.t,*popt))
        if plotflag:
            plotData([(self.t,self.I),(self.t,fitEq(self.t,*popt))], linestyles=['k-','r-'], labels=['data','fit'], title='fitAll', xlabel='Time (s)', ylabel='Signal (V)')
        return popt, confint_fit, r2

    def fitI(self,tI=None,plotflag=False):
        """ Fit IEquation to the segment data.
        Input optional tI=(t,I), plotflag=True. Defaults to whole segment data and no plot.
        Returns (popt,pcov) of fit.  Fit parameters are those of IEquation. """
        t = self.t if tI is None else tI[0]
        I = self.I if tI is None else tI[1]
        linestyle = 'k-' if tI is None else 'kd'
        mx, avg, rng = np.max(I), np.mean(I), np.max(I)-np.min(I)
        guess = [avg, rng/2, rng/20, 0, 0]
        bound = ([0,0,0,-2*np.pi,-2*np.pi], [mx,rng,rng,2*np.pi,2*np.pi])
        scale = 1/np.mean(I)   # Used to normalize I data before fitting to get best fit result
        IEq = lambda t,I0,Im,Im2,phiIm,phiIm2 : self.IEquation(t,I0,Im,Im2,phiIm,phiIm2)*scale
        popt, pcov = curve_fit(IEq,t,I*scale,p0=guess,bounds=bound)
        popt[-2:] = np.mod(popt[-2:],2*np.pi)
        if plotflag:
            plotData([(t,I),(t,self.IEquation(t,*popt))], linestyles=[linestyle,'r-'], labels=['data','fit'], title='fitI', xlabel='Time (s)', ylabel='Signal (V)')
        return popt, pcov

    def fitV(self,peakIFits,peakDtFit,alpha_v=None,v_secondorder=True,plotflag=False):
        """ Fit interferenceEquation and a vEquation* to the normalized segment data.
        Input peakIFits, peakDtFit, alpha_v=None or float(0:1), v_secondorder=bool, plotflag=bool.
        peakIFits from fitI, peakDtFit from fitPeakDt or fitPeakCount, v_secondorder=True uses vEquation_SecondOrder() by default.
        alpha_v=None defaults to custom bounds & quarter pi phase bounds.  Default is no plot.
        Returns (popt,pcov) of fit.  Fit parameters are those of vEquation* (vEquation if v_secondorder False otherwise vEquation_SecondOrder). """
        I_pos, I_neg = peakIFits
        I_norm = self.normalizeData(I_pos[0],I_neg[0])
        kguess = self.fitK(I_norm,peakDtFit)
        vguess = peakDtFit[0]
        v_lowerbounds = np.zeros(len(vguess))
        v_upperbounds = np.zeros(len(vguess))
        if alpha_v is None:
            v_lowerbounds[0] = vguess[0] - np.pi/4
            v_upperbounds[0] = vguess[0] + np.pi/4
            v_lowerbounds[1] = vguess[1]*0.5  # - 2*np.pi
            v_upperbounds[1] = vguess[1]*1.5  # + 2*np.pi
            v_lowerbounds[-2:] = vguess[-2:] - np.pi/4
            v_upperbounds[-2:] = vguess[-2:] + np.pi/4
        else:
            v_lowerbounds, v_upperbounds = self.calcCI(peakDtFit,alpha=alpha_v)
        v_lowerbounds[np.isnan(v_lowerbounds) | np.isinf(v_lowerbounds)] = 0
        v_upperbounds[np.isnan(v_upperbounds)] = np.inf
        v_lowerbounds = np.maximum(v_lowerbounds, np.array([0,0,vguess[2]-np.pi,vguess[3]-np.pi]))
        v_upperbounds = np.minimum(v_upperbounds, np.array([np.inf,np.inf,vguess[2]+np.pi,vguess[3]+np.pi]))
        k_lower = 1E-12  # if kguess-self.kfit_error_estimate < 1E-12 else kguess-self.kfit_error_estimate
        k_upper = 1     if kguess+self.kfit_error_estimate > 1     else kguess+self.kfit_error_estimate
        guess = np.concatenate((np.array([kguess]), vguess))
        lowerbound = np.concatenate((np.array([k_lower]), v_lowerbounds))
        upperbound = np.concatenate((np.array([k_upper]), v_upperbounds))
        if v_secondorder:
            guess = np.concatenate((guess, np.array([0, 0])))
            lowerbound = np.concatenate((lowerbound, np.array([0, -2*np.pi])))
            upperbound = np.concatenate((upperbound, np.array([upperbound[1], 2*np.pi])))
        bound = (lowerbound, upperbound)
        if v_secondorder:
            fitEq = lambda t,k,vm,vs,phivm,phivs,vm2,phivm2 : self.interferenceEquation(t,k,self.vEquation_SecondOrder,[vm,vs,phivm,phivs,vm2,phivm2])
        else:
            fitEq = lambda t,k,vm,vs,phivm,phivs : self.interferenceEquation(t,k,self.vEquation,[vm,vs,phivm,phivs])
        SSE_list, popt_list, pcov_list = [], [], []
        deltav = 2*np.pi
        dv = np.pi/8
        vm_lower_list = np.arange(bound[0][1]-deltav, bound[1][1]+deltav+dv/2, dv)
        vm_upper_list = vm_lower_list + dv
        vm_guess_list = vm_lower_list + dv/2
        warnings.simplefilter("ignore",category=RuntimeWarning)
        for vm_lower, vm_upper, vm_guess in zip(vm_lower_list,vm_upper_list,vm_guess_list):
            bound[0][1] = vm_lower
            bound[1][1] = vm_upper
            guess[1]    = vm_guess
            try:
                popt, pcov = curve_fit(fitEq,self.t,I_norm,p0=guess,bounds=bound)
            except RuntimeError:
                popt, pcov = np.zeros(guess.size), np.zeros((guess.size,guess.size))
            popt_list.append(popt)
            pcov_list.append(pcov)
            if np.max(popt)==0:
                SSE_list.append(np.inf)
            else:
                SSE_list.append(np.sum((I_norm-fitEq(self.t,*popt))**2))
        i_best = np.argmin(SSE_list)
        if np.isinf(SSE_list[i_best]):
            raise RuntimeError('Optimal parameters not found in fitV for t = {} - {}'.format(self.t[0],self.t[-1]))
        popt = popt_list[i_best]
        pcov = pcov_list[i_best]
        popt[3:5] = np.mod(popt[3:5],2*np.pi)
        if v_secondorder:
            popt[-1] = np.mod(popt[-1],2*np.pi)
        if plotflag:
            plotData([(self.t,I_norm),(self.t,fitEq(self.t,*popt))], linestyles=['k-','r-'], labels=['data','fit'], title='fitV', xlabel='Time (s)', ylabel='Normalized Signal')
        return popt, pcov

    def fitK(self,I_norm,peakDtFit,kguess=None,plotflag=False):
        """ Fit the interferenceEquation, for the k parameter, to normalized data assuming the given vEquation parameters in peakDtFit.
        Input I_norm, peakDtFit, kguess=float(0,1).  Where I_norm is normalized intensity to [0:1], and peakDtFit are fit results output from fitPeakDt.
        Optional kwarg kguess fits normalized data.  Default None instead fits integrated areas.
        Returns kopt """
        vm, vs, phivm, phivs = peakDtFit[0]
        if kguess is None:
            I_integrated = simps(I_norm,x=self.t)
            fitEq = lambda k : simps(self.interferenceEquation(self.t,k,self.vEquation,[vm,vs,phivm,phivs]),x=self.t) - I_integrated
            kopt = 1E-12 if fitEq(1E-12)<=0 else bisect(fitEq,1E-12,1)
        else:
            if (kguess<0) or (kguess>1):
                raise ValueError('kwarg kguess in fitK outside acceptable range of 0-1. Input kguess={}'.format(kguess))
            lowerbound = kguess-self.kfit_error_estimate if kguess-self.kfit_error_estimate > 0 else 0
            upperbound = kguess+self.kfit_error_estimate if kguess+self.kfit_error_estimate < 1 else 1
            bound = ([lowerbound],[upperbound])
            fitEq = lambda t,k : self.interferenceEquation(t,k,self.vEquation,[vm,vs,phivm,phivs])
            kopt, kcov = curve_fit(fitEq,self.t,I_norm,p0=kguess,bounds=bound)
            kopt = kopt[0]
            if plotflag:
                plotData([(self.t,I_norm),(self.t,fitEq(self.t,kopt))],linestyles=['k-','r--'],labels=['data','fit'],xlabel='Time (s)',title='fitK')
        return kopt

    def fitPeaksI(self,ipeaks,plotflag=False):
        """ Fit the second order intensity function to the data at the peaks.
        Input ipeaks, plotflag=bool.  ipeaks=(ipos,ineg)
        Returns (fit_pos,fit_neg). Where fit_* is output from fitI. """
        ipos, ineg = ipeaks
        fit_pos = self.fitI(tI=(self.t[ipos],self.I[ipos]),plotflag=plotflag)
        fit_neg = self.fitI(tI=(self.t[ineg],self.I[ineg]),plotflag=plotflag)
        return fit_pos, fit_neg

    def fitPeakCount(self,peaks,turns,vs_fit=None,alpha_vs=None,plotflag=False):
        """ Fit the cumulative peak count * pi, between turns to the first order wavelength function.
        peaks=((ipeaks_pos,ipeaks_neg), ipeaks_all)
        iturns=((iturns_pos,iturns_neg), iturns_all). Values in iturns should not appear in ipeaks.
        vs_fit (popt,pcov) of fit parameters [vs,phivs] from ScanSegment.estimateVs.
        alpha_vs=None or float(0:1). alpha_vs used to calculate vs bounds from fit. Default alpha_vs=None uses custom bounds.
        Returns (popt,pcov). Fit results of fitting vEquation. """
        i_between, t_between = self.peaksBetweenTurns(peakdata=(peaks,turns))
        tcum = t_between[1]
        vcum = np.arange(len(i_between[1]))*np.pi
        popt_list, pcov_list, SSE_list = [], [], []
        dt = np.abs(tcum[-1]-tcum[0])
        vm_est = (vcum[-1]/2) / np.sin(2*np.pi*self.fm*dt/2)
        vm_max = np.max([vcum[-1], vcum[-1]*self.Tm/dt])
        if vs_fit is not None:
            vs,phivs = vs_fit[0]
            guess = [vm_est, np.pi, 0]
            bound = ([0,0,-np.inf],[vm_max,2*np.pi,np.inf])
            fitEq = lambda t,vm,phivm,v0 : v0 + self.vEquation(t,vm,vs,phivm,phivs)
        else:
            guess = [vm_est, 0, np.pi, 0, 0]
            bound = ([0,0,0,-2*np.pi,-np.inf],[vm_max,np.inf,2*np.pi,2*np.pi,np.inf])
            fitEq = lambda t,vm,vs_,phivm,phivs_,v0 : v0 + self.vEquation(t,vm,vs_,phivm,phivs_)
        for _ in range(2):
            popt,pcov = curve_fit(fitEq,tcum,vcum,p0=guess,bounds=bound)
            popt_list.append(popt)
            pcov_list.append(pcov)
            SSE_list.append(np.sum((vcum-fitEq(tcum,*popt))**2))
            vcum = -vcum
        i_best = np.argmin(SSE_list)
        if vs_fit is not None:
            vm, phivm, v0 = popt_list[i_best]
            popt = np.array([vm, vs, phivm, phivs])
            pcov = np.zeros((4,4))
            pcov[::2,::2] = pcov_list[i_best][:2,:2]
        else:
            popt = popt_list[i_best]
            pcov = pcov_list[i_best]
        if plotflag:
            plotData([(tcum,vcum if i_best==0 else -vcum),(tcum,fitEq(tcum,*popt_list[i_best]))],linestyles=['o','-'],labels=['data','fit'],title='fitPeakCount cumulative peak fitting')
        return popt, pcov

    def collectFits(self,peakIFits,vFit,alpha=alpha_default):
        """ Combine fitFilteredPeaksI and fitV results for input to fitAll.
        Input peakIFits, vFit, alpha=float(0,1).
        Returns paramfits, confint.   """
        popt_pos, pcov_pos = peakIFits[0]
        popt_neg, pcov_neg = peakIFits[1]
        popt_v,   pcov_v   = vFit
        paramfits = np.concatenate((popt_pos, popt_neg, popt_v))
        ci_pos = self.calcCI((popt_pos, pcov_pos),alpha=alpha)
        ci_neg = self.calcCI((popt_neg, pcov_neg),alpha=alpha)
        ci_v   = self.calcCI((popt_v,   pcov_v),  alpha=alpha)
        confint = (np.concatenate((ci_pos[0], ci_neg[0], ci_v[0])), np.concatenate((ci_pos[1], ci_neg[1], ci_v[1])))
        return paramfits, confint

    def getVparams(self,popt):
        """ Extract the vEquation* wavelength parameters from the results of fitAll or collectFits.
        Input popt.
        Returns popt_v. """
        popt_len = [17, 15]
        if popt.size == popt_len[0]:
            popt_v = popt[-6:]
        elif popt.size == popt_len[1]:
            popt_v = popt[-4:]
        else:
            raise ValueError('Unknown length of parameters entered to preprocess.ModulationSegment.getVparams(). len(popt)={} Needs to be either {} or {}'.format(popt.size,popt_len[0],popt_len[1]))
        return popt_v

    def calcV(self,popt,plotflag=False):
        """ Calculate v(t) over the modulation segment using vEquation* (first or second order) based on length of popt.
        Input popt.  popt is array of parameters for vEquation*.
        Returns v same length as self.t. """
        popt_len = [6,4]
        if popt.size == popt_len[0]:
            v = self.vEquation_SecondOrder(self.t,*popt)
        elif popt.size == popt_len[1]:
            v = self.vEquation(self.t,*popt)
        else:
            raise ValueError('Unknown length of parameters entered to preprocess.ModulationSegment.calcV(). len(popt)={} Needs to be either {} or {}'.format(popt.size,popt_len[0],popt_len[1]))
        return v

    def findExtremeTurnIndex(self,popt_v,first=True):
        """ Find first or last turn index based on wavelength fit parameters.
        Input popt_v=array, first=True.  wavelength fit parameters for v_all_derivates.
        Return iturn. """
        popt_v = popt_v[:4]
        _, dvdt, _ = self.v_all_derivatives(self.t,*popt_v)
        if first:
            if dvdt[0] == 0:
                iturn = 0
            else:
                iturn = np.argmin(dvdt/dvdt[0] > 0)
        else:
            if dvdt[-1] == 0:
                iturn = len(self.t)-1
            else:
                samesigns = dvdt/dvdt[-1] > 0
                iturn = len(self.t) - 1 - np.argmin(samesigns[::-1])
        return iturn

    def calcCI(self,fit_result,alpha=alpha_default):
        """ Calculate confidence interval of fit_result parameters assuming gaussian error.
        Input fit_result, alpha=float(0:1).  fit_result=(popt,pcov).
        Returns (ci_lower, ci_upper). """
        popt, pcov = fit_result
        ci_lower, ci_upper = norm.interval(alpha, loc=popt, scale=np.sqrt(np.diagonal(pcov)))
        ci_lower[np.isnan(ci_lower)] = -np.inf
        ci_upper[np.isnan(ci_upper)] =  np.inf
        return ci_lower, ci_upper

    def calcR2(self,data,fit):
        """ Calculate the R-squared goodness of fit parameter.
        Input data, fit.  Both as arrays
        Returns r2. """
        SSE = np.sum((data-fit)**2)
        SST = np.sum((data-np.mean(data))**2)
        return 1-(SSE/SST)

    def peaksBetweenTurns(self,peakdata=None,plotflag=False):
        """ Separate the indices and times of peaks into lists of arrays separated at wavelength turns.
        Input peakdata,plotflag=bool.  peakdata is output from findValidPeakIndices(). Default is to run findValidPeakIndices() and no plot.
        Returns (list of indices arrays, list of times arrays). """
        if peakdata is None:
            ipeaks, iturns = self.findValidPeakIndices(plotflag=plotflag)
        else:
            ipeaks, iturns = peakdata
        _, ipeaks_sorted = ipeaks
        _, iturns_sorted = iturns
        iturns_sorted = np.concatenate((np.array([-1]),iturns_sorted,np.array([np.max(ipeaks_sorted)+1])))
        i_out, t_out = [], []
        for iturn1, iturn2 in zip(iturns_sorted[:-1],iturns_sorted[1:]):
            i_inrange = ipeaks_sorted[(ipeaks_sorted>iturn1) & (ipeaks_sorted<iturn2)]
            i_out.append(i_inrange)
            t_out.append(self.t[i_inrange])
        return i_out, t_out

    def normalizeData(self,popt_pos,popt_neg):
        """ Normalize the segment data to the range [0:1].
        Input popt_pos,popt_neg.  Where popt_* are fitI parameters to the upper and lower peaks. """
        I_pos = self.IEquation(self.t,*popt_pos)
        I_neg = self.IEquation(self.t,*popt_neg)
        return (self.I - I_neg) / (I_pos - I_neg)

    def IEquation(self,t,I0,Im,Im2,phiIm,phiIm2):
        """ Second order sinusoidal intensity function. """
        return I0 + Im*np.sin(2*np.pi*self.fm*t + phiIm) + Im2*np.sin(4*np.pi*self.fm*t + phiIm2)

    def interferenceEquation(self,t,k,vEq,vArgsList):
        """ The Airy-like normalized interference function for etalon signal fitting.
        Input a time array, a single float constant (k), the wavelength function vEq,
        and a list of arguments to pass to vEq.
        Returns interferenceEquation(k,vEq(t,args)). """
        return (1-k)/(2*k) * (-1 + (1+k)/(1 - k*np.sin(vEq(t,*vArgsList))))

    def vEquation(self,t,vm,vs,phivm,phivs):
        """ First order wavelength function given times and wavelength parameters.  Returns v(t). """
        return self.v_all_derivatives(t,vm,vs,phivm,phivs)[0]

    def vEquation_SecondOrder(self,t,vm,vs,phivm,phivs,vm2,phivm2):
        """ Second order wavelength function given times and wavelength parameters.  Returns v(t). """
        return self.vEquation(t,vm,vs,phivm,phivs) + vm2*np.sin(4*np.pi*self.fm*t+phivm2)

    def v_all_derivatives(self,t,vm,vs,phivm,phivs):
        """ First order wavelength function given times and wavelength parameters.
        Returns v, v', v'' """
        v      =                         vm*np.sin(2*np.pi*self.fm*t+phivm) +                         vs*np.sin(2*np.pi*self.fs*t+phivs)
        dvdt   =   2*np.pi*self.fm     * vm*np.cos(2*np.pi*self.fm*t+phivm) +   2*np.pi*self.fs     * vs*np.cos(2*np.pi*self.fs*t+phivs)
        d2vdt2 = -(2*np.pi*self.fm)**2 * vm*np.sin(2*np.pi*self.fm*t+phivm) + -(2*np.pi*self.fs)**2 * vs*np.sin(2*np.pi*self.fs*t+phivs)
        return v, dvdt, d2vdt2

    def fitPeakDt(self,peaks,turns,vs_fit=None,alpha_vs=None,plotflag=False):
        """ Fit the forward time difference between peaks to the first order wavelength function.
        peaks=((ipeaks_pos,ipeaks_neg), ipeaks_all)
        iturns=((iturns_pos,iturns_neg), iturns_all). Values in iturns should not appear in ipeaks.
        vs_fit (popt,pcov) of fit parameters [vs,phivs] from ScanSegment.estimateVs.
        alpha_vs=None or float(0:1). alpha_vs used to calculate vs bounds from fit. Default alpha_vs=None uses custom bounds.
        Returns (popt,pcov). Fit results of fitting vEquation. """
        ipeaks_, ipeaks_all = peaks
        iturns_, iturns_all = turns
        ipeaks_pos, ipeaks_neg = ipeaks_
        iturns_pos, iturns_neg = iturns_
        iall_pos = np.sort(np.concatenate((ipeaks_pos,iturns_pos)))
        iall_neg = np.sort(np.concatenate((ipeaks_neg,iturns_neg)))
        iiturns_pos = np.where(np.in1d(iall_pos,iturns_pos))[0]
        iiturns_neg = np.where(np.in1d(iall_neg,iturns_neg))[0]
        t_pos = self.t[iall_pos]
        t_neg = self.t[iall_neg]
        data = {'dt': np.array([t2-t1 for t1,t2 in zip(t_pos[:-1],t_pos[1:])] + [t2-t1 for t1,t2 in zip(t_neg[:-1],t_neg[1:])]),
                't' : np.concatenate((t_pos[:-1],t_neg[:-1])),
                'type': np.array(['pos']*(len(t_pos)-1) + ['neg']*(len(t_neg)-1)),
                'i_peak': np.array(list(range(len(t_pos)-1)) + list(range(len(t_neg)-1))),}
        df = pd.DataFrame.from_dict(data)
        df.sort_values('t',inplace=True)
        df.reset_index(drop=True,inplace=True)
        df.drop(df[np.in1d(df['i_peak'],iiturns_pos) & (df['type']=='pos')].index, inplace=True)
        df.drop(df[np.in1d(df['i_peak'],iiturns_neg) & (df['type']=='neg')].index, inplace=True)
        df.drop(df[np.in1d(df['i_peak'],np.subtract(iiturns_pos,1)) & (df['type']=='pos')].index, inplace=True)
        df.drop(df[np.in1d(df['i_peak'],np.subtract(iiturns_neg,1)) & (df['type']=='neg')].index, inplace=True)
        if len(iturns_pos)+len(iturns_neg) < 2:
            df.drop([0,1,2,3]+df.tail(3).index.tolist(), inplace=True)  # Ignore first and last 4 peaks
        n_valid_peaks = len(ipeaks_pos)+len(ipeaks_neg)
        n_max_peaks = len(ipeaks_pos)+len(ipeaks_neg)+2*len(iturns_pos)+2*len(iturns_neg) # If reversal peaks were valid + a valid peak were missed at each reversal
        t_all_peaks = np.concatenate((t_pos,t_neg))
        t_range_peaks = np.max(t_all_peaks) - np.min(t_all_peaks)
        vm_est = np.pi*n_valid_peaks/(4*self.fm*t_range_peaks)
        vm_max = np.pi*n_max_peaks/(4*self.fm*t_range_peaks)
        t_turn = np.mean(np.mod(self.t[np.concatenate(iturns_)],self.Tm/2))
        phivm_est = np.mod(2*np.pi * (t_turn - self.Tm/4) / self.Tm, 2*np.pi)
        guess = [vm_est, 0, phivm_est, 0]
        bound = ([0,0,phivm_est-np.pi/2,-np.pi], [vm_max,np.Inf,phivm_est+np.pi/2,np.pi])
        scale = 1/np.mean(df['dt'].values)   # Used to normalize dt data before fitting to get best fit result
        # Fit once, then check for better fit at phivm +/- pi
        popt_list = []
        pcov_list = []
        SSE_list = []
        fitEq = lambda t,vm,vs,phivm,phivs : self.dtForwardToNextPeak(t,vm,vs,phivm,phivs)*scale
        for _ in range(2):
            if vs_fit is not None:
                guess[1::2] = vs_fit[0]
                if alpha_vs is None:
                    bound[0][1] = guess[1]*0.5 #if guess[1]*0.75 > 0 else 0
                    bound[1][1] = guess[1]*1.5
                    bound[0][3] = guess[3]-np.pi/4
                    bound[1][3] = guess[3]+np.pi/4
                else:
                    if (alpha_vs<0) or (alpha_vs>1):
                        raise ValueError('Invalid alpha_vs in preprocess.ModulationSegment.fitPeakDt(). {} not between 0 and 1.'.format(alpha_vs))
                    ci_lower, ci_upper = self.calcCI(vs_fit,alpha=alpha_vs)
                    bound[0][1::2] = ci_lower
                    bound[1][1::2] = ci_upper
            print('vm_est={} phivm_est={}'.format(guess[0],guess[2]))
            popt, pcov = curve_fit(fitEq,df['t'].values,df['dt'].values*scale,p0=guess,bounds=bound)
            print('vm_fit={} phivm_fit={}'.format(popt[0],popt[2]))
            popt[-2:] = np.mod(popt[-2:],2*np.pi)
            popt_list.append(popt)
            pcov_list.append(pcov)
            SSE_list.append(np.sum((self.dtForwardToNextPeak(df['t'].values,*popt)-df['dt'].values)**2))
            # Change guess and bounds to check for better fit at opposite phase
            guess[2] = np.mod(phivm_est + np.pi, 2*np.pi)
            bound[0][2] = guess[2] - np.pi/2
            bound[1][2] = guess[2] + np.pi/2
        i_bestfit = np.argmin(SSE_list)
        popt, pcov = popt_list[i_bestfit], pcov_list[i_bestfit]
        if plotflag:
            t = df['t'].values
            xy = [(t,df['dt'].values),(t,self.dtForwardToNextPeak(t,*popt))]
            plotData(xy, linestyles=['ko','r+'], labels=['data','fit'], title='fitPeakDt', xlabel='Time (s)', ylabel='$\Delta$t (s)')
        return popt, pcov

    def dtForwardToNextPeak(self,t,vm,vs,phivm,phivs,tolerance=None):
        """ Function to calculate dt to next peak for each input time assuming given wavelength parameters.
        Assumes a peak occurs at all of the given time values and the next peak occurs at v(t_next) = v(t) +/- 2*n*pi with n = -1, 0, or 1.
        Returns dt_forward array. """
        if tolerance is None:
            tolerance = 1/(100*self.SR)     # Set default tolerance if none given
        v_all = lambda x: self.v_all_derivatives(x,vm,vs,phivm,phivs)   # Define lambda function to get v,v',v" at any input time, x
        v, dv, d2v = v_all(t)   # Calculate the value, slope, & curvature at all input times
        t_next_list = list()     # Initialize list of "next peak" times to solve for
        for i, _ in enumerate(t):     # Loop through all input times
            # Calculate initial guess for next t at each possible v_target
            v_target_list = v[i] + np.array([-2*np.pi, 0, 2*np.pi])    # Define v(t) +/- 2*n*pi which v(t) must cross next, n = -1,0,1
            df = pd.DataFrame()
            for v_t in v_target_list:
                df = df.append(pd.DataFrame(data={'dt':np.roots([d2v[i]/2, dv[i], v[i]-v_t]),'v_target':v_t}))  # Calculate all possible initial guesses for time of next peak
            df.reset_index(drop=True,inplace=True)
            df.drop(df[(np.iscomplex(df['dt'])) | (df['dt'].values.real<=0)].index, inplace=True)     # Remove complex, negative, and zero guesses
            t_guess, v_target = df.loc[df['dt'].idxmin()].values.real + np.array([t[i],0])     # Use the minimum remaining dt as the t_next, v_next guess
            # Move the guess until it converges on the correct answer
            case = round((v_target-v[i])/(2*np.pi))
            crossedflag = False
            v_guess, dv_guess, d2v_guess = v_all(t_guess)
            dt = t_guess - t[i]
            if case == -1:  # Case of v_target = v[i] - 2*pi
                if (not crossedflag) & (v_guess<=v_target) & (dv_guess<0): crossedflag = True  # Check if target was crossed
                # Move right (linearly) until past target or past local minimum
                while (v_guess>v_target) & (dv_guess<0):
                    t_guess += dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    if (not crossedflag) & (v_guess<=v_target) & (dv_guess<0): crossedflag = True  # Recheck if target was crossed
                # Move half step back, then bisect until find target or local min
                dt = dt/2
                t_guess -= dt
                v_guess, dv_guess, d2v_guess = v_all(t_guess)
                if (not crossedflag) & (v_guess<=v_target) & (dv_guess<0): crossedflag = True
                while dt > tolerance/2:
                    dt = dt/2
                    if dv_guess >= 0:
                        t_guess -= dt
                    elif v_guess > v_target:
                        t_guess += dt
                    elif v_guess < v_target:
                        t_guess -= dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    if (not crossedflag) & (v_guess<=v_target) & (dv_guess<0): crossedflag = True
                # If target was never crossed, converged to a local min. Update params and find new target
                if not crossedflag:
                    t_min, v_target, dt = t_guess, v[i], t_guess-t[i]
                    t_guess = t_min + dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    while dt > tolerance/2:
                        dt = dt/2
                        t_guess = t_guess+dt if v_guess<v_target else t_guess-dt
                        v_guess, dv_guess, d2v_guess = v_all(t_guess)
            elif case == 1:  # Case of v_target = v[i] + 2*pi
                if (not crossedflag) & (v_guess>=v_target) & (dv_guess>0): crossedflag = True  # Check if target was crossed
                # Move right (linearly) until past target or past local maximum
                while (v_guess<v_target) & (dv_guess>0):
                    t_guess += dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    if (not crossedflag) & (v_guess>=v_target) & (dv_guess>0): crossedflag = True  # Recheck if target was crossed
                # Move half step back, then bisect until find target or local min
                dt = dt/2
                t_guess -= dt
                v_guess, dv_guess, d2v_guess = v_all(t_guess)
                if (not crossedflag) & (v_guess>=v_target) & (dv_guess>0): crossedflag = True
                while dt > tolerance/2:
                    dt = dt/2
                    if dv_guess <= 0:
                        t_guess -= dt
                    elif v_guess < v_target:
                        t_guess += dt
                    elif v_guess > v_target:
                        t_guess -= dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    if (not crossedflag) & (v_guess>=v_target) & (dv_guess>0): crossedflag = True
                # If target was never crossed, converged to a local min. Update params and find new target
                if not crossedflag:
                    t_max, v_target, dt = t_guess, v[i], t_guess-t[i]
                    t_guess = t_max + dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    while dt > tolerance/2:
                        dt = dt/2
                        t_guess = t_guess+dt if v_guess>v_target else t_guess-dt
                        v_guess, dv_guess, d2v_guess = v_all(t_guess)
            else:  # Case of v_target = v[i]
                if d2v[i] >= 0:     # Subcase of positive curvature at t[i]
                    # Check if local min was crossed, or if overshot past next local max
                    if dv_guess < 0:
                        if d2v_guess > 0:   # If guessed too low, move right until past local min
                            while dv_guess < 0:
                                t_guess += dt
                                v_guess, dv_guess, d2v_guess = v_all(t_guess)
                        else:   # If guessed too far (past next local max), move left
                            dt = dt/10
                            while dv_guess < 0:
                                t_guess -= dt
                                v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    # Find the local minimum by bisection
                    dt = t_guess - t[i]
                    dt = dt/2
                    t_guess -= dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    while dt > tolerance/2:
                        dt = dt/2
                        t_guess = t_guess+dt if dv_guess<0 else t_guess-dt
                        v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    t_min, v_min = t_guess, v_guess
                    v_target = v[i]-2*np.pi if v_min<v[i]-2*np.pi else v_target     # If local min is less than v(t)-2*pi, then correct the target
                    # Find the target
                    dt = t_min - t[i]
                    t_guess = t_min + dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    while dt > tolerance/2:
                        dt = dt/2
                        t_guess = t_guess+dt if v_guess<v_target else t_guess-dt
                        v_guess, dv_guess, d2v_guess = v_all(t_guess)
                else:   # Subcase of negative curvature at t[i]
                    # Check if local max was crossed, or if overshot past next local min
                    if dv_guess > 0:
                        if d2v_guess < 0:   # If guessed too low, move right until past local max
                            while dv_guess > 0:
                                t_guess += dt
                                v_guess, dv_guess, d2v_guess = v_all(t_guess)
                        else:   # If guessed too far (past next local min), move left
                            dt = dt/10
                            while dv_guess > 0:
                                t_guess -= dt
                                v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    # Find the local maximum by bisection
                    dt = t_guess - t[i]
                    dt = dt/2
                    t_guess -= dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    while dt > tolerance/2:
                        dt = dt/2
                        t_guess = t_guess+dt if dv_guess>0 else t_guess-dt
                        v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    t_max, v_max = t_guess, v_guess
                    v_target = v[i]+2*np.pi if v_max>v[i]+2*np.pi else v_target     # If local min is less than v(t)-2*pi, then correct the target
                    # Find the target
                    dt = t_max - t[i]
                    t_guess = t_max + dt
                    v_guess, dv_guess, d2v_guess = v_all(t_guess)
                    while dt > tolerance/2:
                        dt = dt/2
                        t_guess = t_guess+dt if v_guess>v_target else t_guess-dt
                        v_guess, dv_guess, d2v_guess = v_all(t_guess)
            t_next_list.append(t_guess)
        return np.array(t_next_list) - t
