#!/usr/bin/python3
# -*- coding: utf-8 -*-

# RespInPeace -- Process and analyse breathing belt (RIP) data.
# Copyright (C) 2018 Marcin Włodarczak
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from operator import itemgetter
import csv
import math
import warnings

from scipy.io import wavfile
import numpy as np
import pandas as pd
import scipy.signal

from peakdetect import peakdetect
import tgt

# Make sure Pandas uses the bottleneck and numexpr libraries
# (in they are installed).
pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)

__all__ = ['RIP']


class RIP:

    def __init__(self, resp_data, samp_freq, cycles=None, speech=None):

        self.resp = resp_data
        self.samp_freq = samp_freq

        self.t = np.arange(len(self.resp)) / self.samp_freq
        self.dur = len(self.resp) / self.samp_freq

        self.rel = None
        self.range = None
        self.range_bot = None
        self.range_top = None

        if cycles is not None:
            self._troughs, self._peaks = self._read_cycles(cycles)
        else:
            self._peaks, self._troughs = None, None

        if speech is not None and not isinstance(speech, tgt.IntervalTier):
            raise ValueError(
                'Wrong format of speech segmentation: {}.'.format(
                    type(speech).__name__))
        else:
            self.speech = None
            

        self._holds = None

    # == Alternative initializers ==

    @classmethod
    def from_wav(cls, fname, cycles=None, speech=None):
        """Read respiratory data from a WAV file."""
        samp_freq, resp = wavfile.read(fname)
        return cls(resp, samp_freq, cycles)

    @classmethod
    def from_csv(cls, fname, samp_freq=None, delimiter=',',
                 cycles=None, speech=None):
        """Read respiratory data from a CSV file.

        If `samp_freq` is not specified, the CSV file should have two
        columns: the first column should list time stamps and the second
        column should list respiratory values.
        """

        tbl = np.loadtxt(fname, delimiter=delimiter)

        if tbl.ndim == 1:
            if samp_freq is None:
                raise ValueError('Unable to infer sampling frequency.')
            else:
                return cls(tbl, samp_freq, cycles)
        elif tbl.shape[1] == 2:
            if samp_freq is not None:
                warnings.warn('Ignoring the timestamp column, assuming the '
                              'sampling frequency of {}'.format(samp_freq))
                return cls(tbl[:, 1], samp_freq)
            else:
                samp_freq = np.mean(np.diff(tbl[:, 0]))
                return cls(tbl[:, 1], samp_freq, cycles)
        else:
            raise ValueError('Input data has {} columns'
                             'expected 2.'.format(tbl.shape[1]))

    def __getitem__(self, key):
        return self.resp[key]

    @property
    def idt(self):
        """Return an indexer to access samples by time stamp values."""
        return TimeIndexer(self.resp, self.samp_freq)

    def __iter__(self):
        return iter(self.resp)

    def __len__(self):
        """Return the number of samples."""
        return len(self.resp)

    def __repr__(self):
        return '{}(samp_freq={}, nsamples={})'.format(
            type(self).__name__, self.samp_freq, len(self))

    def resample(self, resamp_freq):

        # Number of samples in the resampled signal
        resamp_num = math.floor(self.dur * resamp_freq)
        self.resp, self.t = scipy.signal.resample(
            self.resp, num=resamp_num, t=self.t)
        self.samp_freq = resamp_freq

    def detrend(self, type='linear'):
        """Remove linear trend from the data.

        If `type == 'linear'` (default), a linear fit is subtracted.
        Otherwise, if `type == 'constant'`, the mean is taken out.
        """

        self.resp = scipy.signal.detrend(self.resp, type=type)

    def remove_baseline(self, win_len=60):
        """Remove low-frequency baseline fluctuation.

        By default, a 60-second (rectangular) window is used.
        """

        low_passed = self._fft_smooth(win_len * self.samp_freq)
        self.resp = self.resp - low_passed

    def scale(self):
        """Scale the signal by subtracting the mean and dividing
        the standard deviation.

        The resulting signal has a mean of 0 and a standard deviation
        of 1.
        """

        self.resp = (self.resp - np.mean(self.resp)) / np.std(self.resp)

    def find_cycles(self, win_len=10, delta=1, lookahead=1,
                    include_holds=True, **kwargs):
        """Locate peaks and troughs in the signal."""

        resp_scaled = self._move_zscore(win_len * self.samp_freq)
        peaks, troughs = peakdetect(resp_scaled, delta=delta,
                                    lookahead=lookahead)

        # Make sure we start with an inhalation and end with an exhalation.
        if peaks[0][0] < troughs[0][0]:
            peaks = peaks[1:]
        if peaks[-1][0] > troughs[-1][0]:
            peaks = peaks[:-1]

        assert len(peaks) == len(troughs) - 1, \
            'Expected {} peaks, got {}'.format(len(troughs) - 1, len(peaks))

        self._peaks = np.array(peaks)[:, 0].astype('int')
        self._troughs = np.array(troughs)[:, 0].astype('int')

        if include_holds:
            # Pass kwargs to find_holds.
            self.find_holds(**kwargs)

    def _find_holds_within_interval(self, start, end, peak_prominence,
                                    bins=100):
        """Find respiratory holds within the respiratory interval
        delimited by start and end."""

        intr_resp = self.resp[start:end]
        bin_vals, bin_edges = np.histogram(intr_resp, bins)

        # Normalise the histogram.
        bin_vals = bin_vals / sum(bin_vals)

        # Get peaks whose prominence exceeds `peak_prominence`.
        peaks = scipy.signal.argrelmax(bin_vals)[0]
        peaks_prom = scipy.signal.peak_prominences(
            bin_vals, peaks, wlen=5)[0]
        peaks = peaks[peaks_prom > peak_prominence]

        if len(peaks) == 0:
            return

        peaks_prom = peaks_prom[peaks_prom > peak_prominence]
        peaks = peaks[peaks_prom.argsort()]

        # Calculate peak ranges
        *_, lo, hi = scipy.signal.peak_widths(bin_vals, peaks,
                                              rel_height=0.8)
        hold_top = bin_edges[np.round(hi).astype(int)]
        hold_bot = bin_edges[np.round(lo).astype(int)]

        # Find the corresponding time interval.
        holds = []
        for l, h in zip(hold_bot, hold_top):
            within_hold_region = np.logical_and(
                intr_resp >= min(l, h), intr_resp <= max(l, h)).astype(np.int)
            hold_cand = self._find_islands(within_hold_region, 0)
            hold_cand_durs = np.array([x[1] - x[0] for x in hold_cand])
            holds.append(hold_cand[np.argmax(hold_cand_durs)])

        # Merge overlapping hold regions.
        holds_merged = []
        prev_hold = None
        for h in sorted(holds, key=itemgetter(0)):
            if prev_hold is None:
                prev_hold = h
            elif h[0] <= prev_hold[1]:
                prev_hold = (prev_hold[0], h[1])
            else:
                holds_merged.append(prev_hold)
                prev_hold = h
        holds_merged.append(prev_hold)
        return holds_merged

    def find_holds(self, min_hold_dur=0.25, min_hold_gap=0.15,
                   peak_prominence=0.05, bins=100):

        # Identify inhalations and exhalation if not present.
        if self.cycles is None:
            self.find_cycles()

        hold_cand = []
        seg_samp = np.concatenate(
            (np.stack([self._troughs[:-1], self._peaks], axis=1),
             np.stack([self._peaks, self._troughs[1:]], axis=1)))

        for lo, hi in seg_samp[seg_samp[:, 1].argsort()]:
            intr_holds = self._find_holds_within_interval(
                lo, hi, peak_prominence, bins)

            if intr_holds is not None:
                hold_cand += [(lo + h[0],  lo + h[1]) for h in intr_holds]

        if not hold_cand:
            return

        # Merge holds which lie closer than min_hold_gap and
        # exclude holds shorter than min_hold_dur.
        holds = []
        prev_hold = None

        for h in hold_cand:
            if prev_hold is None:
                prev_hold = h
            elif h[0] - prev_hold[1] < min_hold_gap * self.samp_freq:
                prev_hold = (prev_hold[0], h[1])
            else:
                if prev_hold[1] - prev_hold[0] >= min_hold_dur * self.samp_freq:
                    holds.append(prev_hold)
                prev_hold = h
        holds.append(prev_hold)
        self._holds = np.array(holds)

    @property
    def cycles(self):
        """Start and end times (in seconds) of respiratory cycles"""
        cycl_samp = np.stack([self._troughs[:-1], self._troughs[1:]], axis=1)
        return cycl_samp / self.samp_freq

    @property
    def inhalations(self):
        """Start and end times (in seconds) of inhalations."""
        inh_samp = np.stack([self._troughs[:-1], self._peaks], axis=1)
        return inh_samp / self.samp_freq

    @property
    def exhalations(self):
        """Start and end times (in seconds) of exhalations"""
        exh_samp = np.stack([self._peaks, self._troughs[1:]], axis=1)
        return exh_samp / self.samp_freq

    @property
    def segments(self):
        """Start and end times (in seconds) of respiratory intervals
        (inhalations and exhalations) sorted by time in increasing
        order.
        """
        seg_samp = np.concatenate(
            (np.stack([self._troughs[:-1], self._peaks], axis=1),
             np.stack([self._peaks, self._troughs[1:]], axis=1)))
        return seg_samp[seg_samp[:, 1].argsort()] / self.samp_freq

    @property
    def holds(self):

        if self._holds is not None:
            return self._holds / self.samp_freq

    def find_laughters(self):
        raise NotImplementedError

    def _check_cycles(self):
        # TODO: if the speech annotation is available, make sure
        # no inhalation coincides with speech intervals.
        raise NotImplementedError

    def classify_cycles(self):
        """Classify respiratory cycles (intervals from inhalation onsets to
        exhalation offsets) as "speech", "silence" or "VSU".
        """
        raise NotImplementedError

    def estimate_range(self, bot=5, top=95):
        """Calculate respiratory range.

        In order to exclude outlying observations, only include peaks
        and troughs lying inside the percentile range specified by
        `bot` and `top` (5th and 95th percentile by default).
        """

        self.range_bot = np.percentile(self.resp[self._troughs], bot)
        self.range_top = np.percentile(self.resp[self._peaks], top)
        self.range = self.range_top - self.range_bot

    def estimate_rel(self, lookbehind, min_len=1):
        """Estimate REL (resting expiratory level).

        Since REL is to a large extent influenced by posture shifts,
        REL is evaluated in a dynamic fashion as the median signal
        level at respiratory troughs in the preceding interval of
        length `lookbehind`.
        """

        lookbehind_samp = lookbehind * self.samp_freq
        rel = np.zeros(len(self._troughs) - 1)

        for i, trough in enumerate(self._troughs[:-1]):

            prev_troughs = self._troughs[np.logical_and(
                self._troughs < trough,
                self._troughs > trough - lookbehind_samp)]

            if len(prev_troughs):
                rel[i] = np.median(self.resp[prev_troughs])
            else:
                rel[i] = np.nan

        self.rel = rel

    def rel_at_time(self, t):

        samp_ind = round(t * self.samp_freq)
        cycle_offset = self._troughs[:-1] - samp_ind
        inh_ind = len(cycle_offset[cycle_offset <= 0]) - 1
        if inh_ind >= 0:
            return self.rel[inh_ind]
        else:
            return None

    # == Feature extraction ==

    def extract_amplitude(self, start, end, norm=True):

        resp = self.idt[start:end]

        if norm:
            return (resp[-1] - resp[0]) / self.range
        else:
            return resp[-1] - resp[0]

    def extract_slope(self, start, end, norm=True):
        
        dur = end - start
        return self.extract_amplitude(start, end, norm) / dur

    def extract_level(self, t, norm=True):

        if norm:
            return (self.idt[t] - self.rel_at_time(t)) / self.range
        else:
            return (self.idt[t] - self.rel_at_time(t))

    # == Saving results to file ==

    def save_resp(self, filename, filetype='wav'):
        """Save respiratory data to file."""

        if filetype == 'wav':
            wavfile.write(filename, data=self.resp, rate=self.samp_freq)
        elif filetype == 'table':
            warnings.warn('Saving to a plain-text table. Only time stamps'
                          'and respiratory values will be saved')

            with open(filename, 'w') as fout:
                csv_out = csv.writer(fout)
                csv_out.writerows(zip(self.t, self.resp))
        else:
            raise ValueError('Unsupported filetype: {}.'.format(filetype))

    def save_annotations(self, filename, tiers=['cycles', 'holds'],
                         filetype='textgrid', merge_holds=False):
        """Save annotations to file."""

        if filetype not in ['textgrid', 'eaf', 'table']:
            raise ValueError('Unsupported file type: {}'.format(filetype))

        tg = tgt.TextGrid()

        if 'holds' in tiers or merge_holds:
            holds = tgt.IntervalTier(name='holds')
            for start, end in self.holds:
                holds.add_interval(tgt.Interval(start, end, 'hold'))
            if not merge_holds:
                tg.add_tier(holds)

        if 'cycles' in tiers:
            cycles = tgt.IntervalTier(name='cycles')
            for inh, exh in zip(self.inhalations, self.exhalations):
                cycles.add_intervals(
                    [tgt.Interval(inh[0], inh[1], 'in'),
                     tgt.Interval(exh[0], exh[1], 'out')])
            if merge_holds:
                tg.add_tier(self.merge_holds(cycles, holds))
            else:
                tg.add_tier(cycles)

        if len(tg.tiers):
            filetype = 'short' if filetype == 'textgrid' else filetype
            tgt.write_to_file(tg, filename, format=filetype)

    # == Private methods ==

    @staticmethod
    def _merge_holds(cycles, holds):
        """Merge respiratory holds with the inhalation and exhalation
        boundaries."""

        i, j = 0, 0
        cycles = tgt.IntervalTier()
        cur_intr = None
        while i < len(cycles) and j < len(holds):

            if cycles:
                c_start = max(cycles[-1].end_time, cycles[i].start_time)
            else:
                c_start = cycles[i].start_time
            c_end = min(cycles[i].end_time, holds[j].start_time),
            cur_intr = tgt.Interval(c_start, c_end, cycles[i].text)

            if cur_intr.start_time < holds[j].start_time:
                cycles.add_interval(cur_intr)
            if cycles[i].end_time > holds[j].start_time:
                cycles.add_interval(holds[j])
                j += 1
            if cycles[i].end_time <= cycles[-1].end_time:
                i += 1

        return cycles

    @staticmethod
    def _find_islands(a, min_gap):

        a_diff = np.diff(np.pad(a, 1, 'constant'))
        onsets = np.where(a_diff == 1)[0]
        offsets = np.where(a_diff == -1)[0]

        # Close short gaps
        short_gaps = np.nonzero((onsets[1:] - offsets[:-1]) < min_gap)
        if short_gaps:
            onsets = np.delete(onsets, short_gaps[0] + 1)
            offsets = np.delete(offsets, short_gaps[0])

        return list(zip(onsets, offsets))

    def _move_zscore(self, win_len, noise_level=0):
        """Calculate z-score of the signal in a moving window
        of size `win_size`.
        """

        resp_rolling = pd.Series(self.resp).rolling(win_len, center=True)
        window_mean = resp_rolling.mean().values
        window_std = resp_rolling.std().values
        return (self.resp - window_mean) / (window_std + noise_level)

    def _fft_smooth(self, win_len):
        """Zero-phase low-pass filter using FFT convolution.

        Original MATLAB implementation from Breathmetrics
        (https://github.com/zelanolab/breathmetrics). See also: Noto
        T, Zhou G, Schuele S, Templer J, & Zelano C (2018) Automated
        analysis of breathing waveforms using BreathMetrics: a
        respiratory signal processing toolbox. Chemical Senses (in
        press).
        """

        l = len(self.resp)
        win = np.zeros(l)
        mar_left = math.floor((l - win_len + 1) / 2)
        mar_right = math.floor((l + win_len) / 2)
        win[mar_left: mar_right] = 1
        return scipy.signal.fftconvolve(self.resp, win, mode='same') / win_len

    def _read_cycles(self, cycles):
        """Read an external cycles annotation and return boundary indices."""

        # Check if respiratory segmentation is in the correct format.
        if not isinstance(cycles, tgt.IntervalTier):
            raise ValueError('Wrong speech segmentation format: {}'.format(
                type(cycles).__name__))
        cycle_labs = set(i.text for i in cycles)
        if cycle_labs != {'in', 'out'}:
            extra_labs = cycle_labs - {'in', 'out'}
            raise ValueError('Unrecognised respiratory labels: {}.'.format(
                    ', '.join(extra_labs)))
        if cycles[0].text != 'in':
            raise ValueError('Cycle annotation must start with an inhalation.')
        if cycles[-1].text != 'out':
            raise ValueError('Cycle annotation must end with an exhalation.')
        gaps = [cycles[i + 1].start_time - cycles[i].end_time > 0
                for i in range(len(cycles) - 1)]
        if any(gaps):
            raise ValueError('No gaps allowed in between cycles.')

        bounds = np.round(i.start_time * self.samp_freq for i in cycles).astype(np.int)
        troughs, peaks = bounds[::2], bounds[1::2]

        return troughs, peaks


class TimeIndexer:
    """An indexer to access samples by time stamps."""

    def __init__(self, resp, samp_freq):

        self.resp = resp
        self.samp_freq = samp_freq

    def __getitem__(self, key):
        if (isinstance(key, int) or isinstance(key, float)
            or isinstance(key, np.ndarray)):
            idx = self._time_to_sample(key, method='nearest')
            return self.resp[idx]
        elif isinstance(key, slice):
            start = self._time_to_sample(key.start, method='ceil')
            end = self._time_to_sample(key.stop, method='floor')
            if key.step is not None:
                step = self._time_to_sample(key.step)
            else:
                step = key.step
            return self.resp[start:end:step]
        else:
            raise IndexError

    def _time_to_sample(self, t, method='nearest'):
        """Convert time stamp to sample index using the
        specified rounding method. By default the nearest
        sample is returned."""

        if method == 'nearest':
            return np.round(t * self.samp_freq).astype(np.int)
        elif method == 'ceil':
            return np.ceil(t * self.samp_freq).astype(np.int)
        elif method == 'floor':
            return np.floor(t * self.samp_freq).astype(np.int)
        else:
            raise ValueError('Unknown method: {}'.format(method))
