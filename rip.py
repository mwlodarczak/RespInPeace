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
import math
import warnings

from scipy.io import wavfile
import numpy as np
import pandas as pd
import scipy.signal
import scipy.sparse
import scipy.sparse.linalg
from scipy.interpolate import UnivariateSpline

from peakdetect import peakdetect
import tgt

# Make sure Pandas uses the bottleneck and numexpr libraries
# (in they are installed).
pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)

__all__ = ['Resp', 'Sampled', 'TimeIndexer']


class Sampled:
    '''A sampled signal.'''

    def __init__(self, data, samp_freq):

        self.samples = data
        self.samp_freq = samp_freq
        self.t = np.arange(len(self)) / self.samp_freq
        self.dur = len(self) / self.samp_freq

    def __getitem__(self, key):
        return self.samples[key]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        """Return the number of samples."""
        return len(self.samples)

    def __repr__(self):
        return '{}(samp_freq={}, nsamples={})'.format(
            type(self).__name__, self.samp_freq, len(self))

    @property
    def idt(self):
        """Return an indexer to access samples by time stamp values."""
        return TimeIndexer(self.samples, self.samp_freq)


class Resp(Sampled):
    '''A respiratory signal'''

    def __init__(self, resp_data, samp_freq, cycles=None, speech=None,
                 holds=None):

        super(Resp, self).__init__(resp_data, samp_freq)

        self.range = None
        self.range_bot = None
        self.range_top = None

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

    # == Detrending and baseline-related methods

    def detrend(self, type='linear'):
        """Remove linear trend from the data.

        If `type == 'linear'` (default), a linear fit is subtracted.
        Otherwise, if `type == 'constant'`, the mean is taken out.
        """

        self.samples = scipy.signal.detrend(self.samples, type=type)

    def remove_baseline_square(self, win_len=60):
        """Remove low-frequency baseline fluctuation using a rectangular
        window
        """
        baseline = self._fft_smooth(win_len * self.samp_freq)
        self.samples = self.samples - baseline

    def remove_baseline_savgol(self, win_len=60, order=3):
        """Remove low-frequency baseline fluctuation using a Savitzky-Golay
        filter.
        """
        win = win_len * self.samp_freq
        if win % 2 == 0:
            win += 1
        baseline = scipy.signal.savgol_filter(self.samples, win, order)
        self.samples = self.samples - baseline

    def remove_baseline_als(self, lam=1e10, p=0.01, niter=10):
        """Remove baseline fluctuation using Asymmetric Least Squares
        Smoothing. The default values of `lam` (smoothness) and `p`
        (assymetry) might have to be adjusted.

        Source: https://stackoverflow.com/a/50160920 by Torne
        (https://stackoverflow.com/users/12345/torne)
        """
        L = len(self.samples)
        D = scipy.sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        for i in range(niter):
            W = scipy.sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = scipy.sparse.linalg.spsolve(Z, w * self.samples)
            w = p * (self.samples > z) + (1 - p) * (self.samples < z)
        self.samples = self.samples - z

    def scale(self):
        """Scale the signal by subtracting the mean and dividing
        the standard deviation.

        The resulting signal has a mean of 0 and a standard deviation
        of 1.
        """

        mean, sd = np.mean(self.samples), np.std(self.samples)
        self.samples = (self.samples - mean) / sd

    def filter_lowpass(self, cutoff, order, inplace=True):
        """A Butterworth low-pass filter."""

        nyq = 0.5 * self.samp_freq
        b, a = scipy.signal.butter(order, cutoff / nyq, btype='low')
        resp_filt = scipy.signal.filtfilt(b, a, self.samples)
        if inplace:
            self.samples = resp_filt
        else:
            return resp_filt

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

        self._filt = self.filter_lowpass(cutoff=3, order=8, inplace=False)
        # self._filt = self.res

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
        for lo, hi in holds:
            start = lo / self.samp_freq
            end = hi / self.samp_freq
            # Filter out holds overlapping with speech or inhalation:
            if (self.overlaps_speech(start, end)
                    or self.overlaps_inhalation(start, end)):
                continue
            holds_tier.add_interval(tgt.Interval(start, end, 'hold'))
        self.holds = holds_tier

    def overlaps_speech(self, start, end):
        """Check if the interval between `start` and `end` coincides with
        a speech segment."""

        if self.speech is None:
            return
        else:
            return bool(self.speech.get_annotations_between_timepoints(
                start, end, left_overlap=True, right_overlap=True))

    def overlaps_inhalation(self, start, end):
        """Check if the interval between `start` and `end` coincides with
        an inhalatory segment."""

        if self.segments is None:
            return
        else:
            coinc = self.segments.get_annotations_between_timepoints(
                start, end, left_overlap=True, right_overlap=True)
            return any(i.text == 'in' for i in coinc)

    @property
    def inhalations(self):
        """Start and end times (in seconds) of inhalations."""

        return self.segments.get_annotations_with_matching_text('in')

    @property
    def exhalations(self):
        """Start and end times (in seconds) of exhalations"""

        return self.segments.get_annotations_with_matching_text('out')

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

    def estimate_range(self, bot=5, top=95):
        """Calculate respiratory range.

        In order to exclude outlying observations, only include peaks
        and troughs lying inside the percentile range specified by
        `bot` and `top` (5th and 95th percentile by default).
        """

        self.range_bot = np.percentile(self.idt[self.troughs], bot)
        self.range_top = np.percentile(self.idt[self.peaks], top)
        self.range = self.range_top - self.range_bot

    def estimate_rel(self, dynamic=False, win_len=11):
        """Estimate REL (resting expiratory level).

        If `dynamic==False`, REL is calculated as the median value of
        all troughs in the resiratory signal. Otherwise, REL is
        estimated in a dynamic fashion to allow for posture shifts.
        This is done by calculating the median level of all troughs in
        a window of specified size (by default, `win_len=11`, i.e. REL
        is calcualte as the median level of five preceding and five
        following troughs).
        """

        if self.troughs is None:
            raise ValueError('The value self.troughs is None.')

        if dynamic:
            if win_len % 2 != 1:
                raise ValueError('Window length must be odd.')

            troughs_med = scipy.signal.medfilt(self.idt[self.troughs], win_len)
            interp = UnivariateSpline(self.troughs, troughs_med, k=3, s=0)
            rel = interp(np.linspace(0, self.t[-1], len(self)))
        else:
            rel = np.median(self.idt[self.troughs])

        self.samples = self.samples - rel

    # == Feature extraction ==

    def extract_amplitude(self, start, end, norm=True):

        if norm:
            return (self.idt[end] - self.idt[start]) / self.range
        else:
            return self.idt[end] - self.idt[end]

    def extract_slope(self, start, end, norm=True):

        dur = end - start
        return self.extract_amplitude(start, end, norm) / dur

    def extract_level_norm(self, t):

        return self.idt[t] / self.range

    def extract_features(self, start, end, norm=True):
        """Extract all features for the given interval."""

        features = {'duration': end - start,
                    'amplitude': self.extract_amplitude(start, end, norm),
                    'slope': self.extract_slope(start, end, norm),
                    'onset_level': self.extract_level_norm(start),
                    'offset_level': self.extract_level_norm(end)}
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
            vc_bot = self.samples.idt[tmin]
            vc_top = self.samples.idt[tmax]
        elif tstart is not None and tend is not None:
            resp_vc = self.samples.idt[tstart:tend]
            vc_bot = np.min(resp_vc)
            vc_top = np.max(resp_vc)
        else:
            raise ValueError('Missing argument: tmin and tmax or tstart and'
                             'tend need to be specified.')

        self.samples = (self.samples - vc_bot) / (vc_top - vc_bot)

    # == Saving results to file ==

    def save_resp(self, filename, filetype='wav'):
        """Save respiratory data to file."""

        if filetype == 'wav':
            wavfile.write(filename, data=self.samples, rate=self.samp_freq)
        elif filetype == 'table':
            warnings.warn('Saving to a plain-text table. Only time stamps'
                          'and respiratory values will be saved')

            with open(filename, 'w') as fout:
                csv_out = csv.writer(fout)
                csv_out.writerows(zip(self.t, self.samples))
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
                tg.add_tier(self.merge_holds(self.segments, self.holds))
            else:
                tg.add_tier(self.segments)

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

        resp_rolling = pd.Series(self.samples).rolling(win_len, center=True)
        window_mean = resp_rolling.mean().values
        window_std = resp_rolling.std().values
        return (self.samples - window_mean) / (window_std + noise_level)

    def _fft_smooth(self, win_len):
        """Zero-phase low-pass filter using FFT convolution.

        Original MATLAB implementation from Breathmetrics
        (https://github.com/zelanolab/breathmetrics). See also: Noto
        T, Zhou G, Schuele S, Templer J, & Zelano C (2018) Automated
        analysis of breathing waveforms using BreathMetrics: a
        respiratory signal processing toolbox. Chemical Senses (in
        press).
        """

        nsamples = len(self)
        win = np.zeros(nsamples)
        mar_left = math.floor((nsamples - win_len + 1) / 2)
        mar_right = math.floor((nsamples + win_len) / 2)
        win[mar_left: mar_right] = 1
        return scipy.signal.fftconvolve(self.samples, win, mode='same') / win_len

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
    '''An indexer to access samples by time stamps.'''

    def __init__(self, resp, samp_freq):

        self.samples = resp
        self.samp_freq = samp_freq

    def __getitem__(self, key):
        if (isinstance(key, int) or isinstance(key, float)
                or isinstance(key, np.ndarray)):
            idx = self._time_to_sample(key, method='nearest')
            return self.samples[idx]
        elif isinstance(key, slice):
            start = self._time_to_sample(key.start, method='ceil')
            end = self._time_to_sample(key.stop, method='floor')
            if key.step is not None:
                step = self._time_to_sample(key.step)
            else:
                step = key.step
            return self.samples[start:end:step]
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
