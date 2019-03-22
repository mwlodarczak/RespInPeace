#!/usr/bin/python3
# -*- coding: utf-8 -*-

# RespInPeace -- Process and analyse breathing belt (RIP) data.
# Copyright (C) 2018-2019 Marcin Włodarczak
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

    def __init__(self, resp_data, samp_freq, cycles=None, speech=None,
                 holds=None):

        self.resp = resp_data
        self.samp_freq = samp_freq

        # TODO: Do we actually need self.t? They can be always computed on the fly.
        self.t = np.arange(len(self.resp)) / self.samp_freq
        self.dur = len(self.resp) / self.samp_freq

        self.rel = None
        self.range = None
        self.range_bot = None
        self.range_top = None

        # if cycles is not None:
        #     self._troughs, self._peaks = self._read_cycles(cycles)
        # else:
        #     self._peaks, self._troughs = None, None

        self.segments = cycles

        if speech is not None and not isinstance(speech, tgt.IntervalTier):
            raise ValueError(
                'Wrong format of speech segmentation: {}.'.format(
                    type(speech).__name__))
        else:
            self.speech = speech
            

        if holds is not None and not isinstance(holds, tgt.IntervalTier):
            raise ValueError(
                'Wrong format of hold segmentation: {}.'.format(
                    type(holds).__name__))
        else:
            self.holds = holds

    # == Alternative initializers ==

    @classmethod
    def from_wav(cls, fname, channel=-1, cycles=None, speech=None, holds=None):
        """Read respiratory data from a WAV file."""
        samp_freq, resp = wavfile.read(fname)
        if resp.ndim == 1:
            return cls(resp, samp_freq, cycles, speech, holds)
        else:
            return cls(resp[:, channel], samp_freq, cycles, speech, holds)

    @classmethod
    def from_csv(cls, fname, samp_freq=None, delimiter=',',
                 cycles=None, speech=None, holds=None):
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
                return cls(tbl, samp_freq, cycles, speech, holds)
        elif tbl.shape[1] == 2:
            if samp_freq is not None:
                warnings.warn('Ignoring the timestamp column, assuming the '
                              'sampling frequency of {}'.format(samp_freq))
                return cls(tbl[:, 1], samp_freq)
            else:
                samp_freq = np.mean(np.diff(tbl[:, 0]))
                return cls(tbl[:, 1], samp_freq, cycles, speech, holds)
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
        if peaks[0] < troughs[0]:
            peaks = peaks[1:]
        if peaks[-1] > troughs[-1]:
            peaks = peaks[:-1]

        assert len(peaks) == len(troughs) - 1, \
            'Expected {} peaks, got {}'.format(len(troughs) - 1, len(peaks))

        # self._peaks = np.array(peaks)[:, 0].astype('int')
        # self._troughs = np.array(troughs)[:, 0].astype('int')

        # Store the results in an IntervalTier.
        inhalations = zip(troughs[:-1], peaks)
        exhalations = zip(peaks, troughs[1:])

        segments = tgt.IntervalTier(name='resp')
        for inh, exh in zip(inhalations, exhalations):
            inh_onset = inh[0] / self.samp_freq
            inh_offset = inh[1] / self.samp_freq
            exh_offset = exh[1] / self.samp_freq

            segments.add_interval(tgt.Interval(inh_onset, inh_offset, 'in'))
            segments.add_interval(tgt.Interval(inh_offset, exh_offset, 'out'))

        self.segments = segments

        if include_holds:
            # Pass kwargs to find_holds.
            self.find_holds(**kwargs)

    def _find_holds_within_interval(self, start, end, peak_prominence,
                                    bins=100):
        """Find respiratory holds within the respiratory interval
        delimited by start and end."""

        intr_resp = self._filt[start:end]

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

        self._filt = self._butter_lowpass(self.resp, highcut=3,
                                          fs=self.samp_freq, order=8)
        plt.plot(self.resp)
        plt.plot(self.filt)
        plt.show()

        # Identify inhalations and exhalation if not present.
        if self.segments is None:
            self.find_cycles()

        hold_cand = []

        for intr in self.segments:

            lo = round(intr.start_time * self.samp_freq)
            hi = round(intr.end_time * self.samp_freq)

            intr_holds = self._find_holds_within_interval(
                lo, hi, peak_prominence, bins)

            if intr_holds is not None:
                hold_cand += [(lo + h[0],  lo + h[1]) for h in intr_holds]

        # Repeat with boundaries shifted by half of a segment.
        shifts = [i.duration() / 2 for i in self.segments]
        intr_shifted = []

        for i in range(len(self.segments) - 1):
            intr_shifted.append(tgt.Interval(self.segments[i].start_time + shifts[i],
                                             self.segments[i].end_time + shifts[i + 1], ''))
                        # for i, s in zip(self.segments, shifts)]

        hold_cand2 = []

        for intr in intr_shifted:

            lo = round(intr.start_time * self.samp_freq)
            hi = round(intr.end_time * self.samp_freq)

            intr_holds = self._find_holds_within_interval(
                lo, hi, peak_prominence, bins)

            if intr_holds is not None:
                hold_cand2 += [(lo + h[0],  lo + h[1]) for h in intr_holds]

        hh = hold_cand + hold_cand2

        hh.sort(key=lambda x: x[1])

        c = []
        temp = hh[0]
        for i in range(1, len(hh)):
            cur = hh[i]
            if temp[1] > cur[1]:
                temp[1] = max(cur[1], temp[1])
            else:
                c.append(temp)
                temp = cur
            i += 1
        c.append(temp)

        if not c:
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
        if prev_hold[1] - prev_hold[0] >= min_hold_dur * self.samp_freq:
            holds.append(prev_hold)

        # Build a holds tier.
        holds_tier = tgt.IntervalTier(name='holds')
        for lo, hi in  holds:
            start = lo / self.samp_freq
            end = hi / self.samp_freq
            # Filter out holds overlapping with speech.
            if (self.speech is not None and
                self.speech.get_annotations_between_timepoints(
                    start, end, True, True)):
                continue
            holds_tier.add_interval(tgt.Interval(start, end, 'hold'))
        self.holds = holds_tier

    @property
    def inhalations(self):
        """Start and end times (in seconds) of inhalations."""

        return self.segments.get_annotations_with_matching_text('in')
        # inh_samp = np.stack([self._troughs[:-1], self._peaks], axis=1)
        # return inh_samp / self.samp_freq

    @property
    def exhalations(self):
        """Start and end times (in seconds) of exhalations"""

        return self.segments.get_annotations_with_matching_text('out')

        # exh_samp = np.stack([self._peaks, self._troughs[1:]], axis=1)
        # return exh_samp / self.samp_freq

    @property
    def troughs(self):
        return np.array([i.start_time for i in self.segments if i.text == 'in']
                        + [self.segments[-1].end_time])

    @property
    def peaks(self):
        return np.array([i.start_time for i in self.segments
                         if i.text == 'out'])

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

        self.range_bot = np.percentile(self.idt[self.troughs], bot)
        self.range_top = np.percentile(self.idt[self.peaks], top)
        self.range = self.range_top - self.range_bot

    def estimate_rel(self, dynamic=False, lookbehind=60, min_len=1):
        """Estimate REL (resting expiratory level).

        Since REL is to a large extent influenced by posture shifts,
        REL is evaluated in a dynamic fashion as the median signal
        level at respiratory troughs in the preceding interval of
        length `lookbehind`.
        """

        if dynamic:
            # lookbehind_samp = lookbehind * self.samp_freq
            rel = np.zeros(len(self.troughs) - 1)

            for i, trough in enumerate(self.troughs[:-1]):

                prev_troughs = self.troughs[np.logical_and(
                    self.troughs < trough,
                    self.troughs > trough - lookbehind)]

                if len(prev_troughs):
                    rel[i] = np.median(self.idt[prev_troughs])
                else:
                    rel[i] = np.nan

            self.rel = rel

        else:
            self.rel =  np.median(self.idt[self.troughs])


    def rel_at_time(self, t):

        if isinstance(self.rel, np.ndarray):
            cycle_offset = self.troughs[:-1] - t
            inh_ind = len(cycle_offset[cycle_offset <= 0]) - 1
            if inh_ind >= 0:
                return self.rel[inh_ind]
            else:
                return None
        else:
            return self.rel

    # == Feature extraction ==

    def extract_amplitude(self, start, end, norm=True):

        if norm:
            return (self.idt[end] - self.idt[start]) / self.range
        else:
            return self.idt[end] - self.idt[end]

    def extract_slope(self, start, end, norm=True):

        dur = end - start
        return self.extract_amplitude(start, end, norm) / dur

    def extract_level(self, t, norm=True):

        if norm:
            if self.rel_at_time(t) is not None:
                return (self.idt[t] - self.rel_at_time(t)) / self.range
            else:
                return None
        else:
            return (self.idt[t] / self.range)

    def extract_features(self, start, end, norm):
        """Extract all features for the given interval."""

        features = {'duration': end - start,
                    'amplitude': self.extract_amplitude(start, end, norm),
                    'slope': self.extract_slope(start, end, norm),
                    'onset_level': self.extract_level(start, norm),
                    'offset_level': self.extract_level(end, norm)}
        return features

    # == Calibration methods ==

    def calibrate_vc(self, vol, tmin=None, tmax=None, tstart=None, tend=None):
        """Calibrate respiratory signal in absolute units of volume given a
        vital capacity (VC) manoeuvre. The locations of measurement
        points need to be identified either by start and end times (in
        which case the range in the interval is calculated) or by
        time points of minimum and maximum lung volumes.

        """

        if tmin is not None and tmax is not None:
            vc_bot = self.resp.idt[tmin]
            vc_top = self.resp.idt[tmax]
        elif tstart is not None and tend is not None:
            resp_vc = self.resp.idt[tstart:tend]
            vc_bot = np.min(resp_vc)
            vc_top = np.max(resp_vc)
        else:
            raise ValueError('Missing argument: tmin and tmax or tstart and'
                             'tend need to be specified.')

        self.resp = (self.resp - vc_bot) / (vc_top - vc_bot)

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
            # holds = tgt.IntervalTier(name='holds')
            # for start, end in self.holds:
            #     holds.add_interval(tgt.Interval(start, end, 'hold'))
            if not merge_holds:
                tg.add_tier(self.holds)

        if 'cycles' in tiers:
            if merge_holds:
                segments_merged= tgt.IntervalTier(name='cycles')
                tg.add_tier(self.merge_holds(self.segments, self.holds))
            else:
                tg.add_tier(self.segments)

        if len(tg.tiers):
            filetype = 'short' if filetype == 'textgrid' else filetype
            tgt.write_to_file(tg, filename, format=filetype)

    # == Private methods ==

    @staticmethod
    def _butter_lowpass(data, cutoff, fs, order):
        """A Butterworth low-pass filter."""

        nyq = 0.5 * fs
        b, a = scipy.signal.butter(order, cutoff / nyq, btype='low')
        return scipy.signal.filtfilt(b, a, data)

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

    # def _read_cycles(self, cycles):
    #     """Read an external cycles annotation and return boundary indices."""

    #     # Check if respiratory segmentation is in the correct format.
    #     if not isinstance(cycles, tgt.IntervalTier):
    #         raise ValueError('Wrong speech segmentation format: {}'.format(
    #             type(cycles).__name__))
    #     cycle_labs = set(i.text for i in cycles)
    #     if cycle_labs != {'in', 'out'}:
    #         extra_labs = cycle_labs - {'in', 'out'}
    #         raise ValueError('Unrecognised respiratory labels: {}.'.format(
    #                 ', '.join(extra_labs)))
    #     if cycles[0].text != 'in':
    #         raise ValueError('Cycle annotation must start with an inhalation.')
    #     if cycles[-1].text != 'out':
    #         raise ValueError('Cycle annotation must end with an exhalation.')
    #     gaps = [cycles[i + 1].start_time - cycles[i].end_time > 0
    #             for i in range(len(cycles) - 1)]
    #     if any(gaps):
    #         raise ValueError('No gaps allowed in between cycles.')

    #     bounds = np.round([i.start_time * self.samp_freq for i in cycles]).astype(np.int)
    #     troughs, peaks = bounds[::2], bounds[1::2]

    #     return troughs, peaks


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
            return np.round(t * self.samp_freq).astype(np.int) - 1
        elif method == 'ceil':
            return np.ceil(t * self.samp_freq).astype(np.int) - 1
        elif method == 'floor':
            return np.floor(t * self.samp_freq).astype(np.int) - 1
        else:
            raise ValueError('Unknown method: {}'.format(method))
