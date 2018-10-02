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

from collections import deque
import math

from scipy.io import wavfile
import numpy as np
import pandas as pd
import scipy.signal

from peakdetect import peakdetect

__all__ = ['RIP']


class RIP:

    def __init__(self, wav_path):

        self.samp_freq, self.resp = wavfile.read(wav_path)
        self.t = np.arange(len(self.resp)) / self.samp_freq
        self.dur = len(self.resp) / self.samp_freq

        self._peaks = None
        self._troughs = None

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

        low_passed = self._fft_smooth(60 * self.samp_freq)
        self.resp = self.resp - low_passed

    def scale(self):
        """Scale the signal by subtracting the mean and dividing
        the standard deviation.

        The resulting signal has a mean of 0 and a standard deviation
        of 1.
        """

        self.resp = (self.resp - np.mean(self.resp)) / np.std(self.resp)

    def find_cycles(self, win_len=10, delta=1, lookahead=1):
        """Locate peaks and troughs in the signal.

        """

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

    @property
    def inhalations(self):
        return np.stack([self._troughs[:-1], self._peaks],
                        axis=1)

    @property
    def exhalations(self):
        return np.stack([self._peaks, self._troughs[1:]],
                        axis=1)

    @property
    def cycles(self):
        return np.stack([self._troughs[:-1], self._troughs[1:]],
                        axis=1)

    def find_holds(self):
        pass

    def find_laughters(self):
        pass

    def estimate_range(self, lo=5, hi=95):
        """Calculate respiratory range.

        In order to exclude outlying observations, only include peaks
        and troughs lying inside the percentile range specified by
        `lo` and `hi` (5th and 95th percentile by default).
        """

        peak_vals = self.resp[self._peaks]
        trough_vals = self.resp[self._troughs]

        return np.percentile(trough_vals, lo), np.percentile(peak_vals, hi)

    def estimate_rel(self, lookbehind, min_len=1):
        """Estimate REL (resting expiratory level).

        Since REL is to a large extent influenced by posture shifts,
        REL is evaluated in a dynamic fashion as the median signal
        level at respiratory troughs in the preceding interval of
        length `lookbehind`.
        """

        cycle_durs = self.cycles[:, 1] - self.cycles[:, 0]

        prev_durs = deque()
        prev_troughs = deque()
        rel = np.zeros(len(cycle_durs))

        # for dur, trough in np.nditer([cycle_durs, self._troughs[:-1]]):
        for i, (dur, trough) in enumerate(zip(cycle_durs, self._troughs[:-1])):

            if len(prev_durs) < min_len:
                rel[i] = np.nan
            else:
                # TODO: Skip the following loop. Perhaps use simple
                # slicing or np.where instead of deques.
                while sum(prev_durs) > lookbehind * self.samp_freq:
                    prev_durs.popleft()
                    prev_troughs.popleft()

                rel[i] = np.median(self.resp[prev_troughs])

            prev_durs.append(dur)
            prev_troughs.append(trough)

        self.rel = rel


    def save_resp(self, filename, samp_freq, filetype='wav'):
        """Save respiratory data to file."""

        if filetype == 'wav':
            wavfile.write(filename, data=self.resp, rate=self.samp_freq)
        else:
            raise ValueError('Illegal filetype: {}.'.format(filetype))

    # def save_annotations(self, tiers=['cycles'], filetype='textgrid'):
    #     """Save annotations to file."""

    #     for bnd in zip(troughs, peaks):
    #         # Convert samples to timestamps
    #         bnd_start_time = bnd[0][0] / self.samp_freq
    #         bnd_end_time = bnd[1][0] / self.samp_freq
    #         # Add exhalation if needed
    #         if prev_inh_offset is not None:
    #             cycles.add_interval(
    #                 tgt.Interval(prev_inh_offset, bnd_start_time, 'out'))
    #         # Add inhalation
    #         cycles.add_interval(
    #             tgt.Interval(bnd_start_time, bnd_end_time, 'in'))
    #         prev_inh_offset = bnd_end_time

    #     self.cycles = cycles

    # == Parameter estimation ==

    def estimate_feature_in_interval(self):
        # The idea would be have a function which evaluates a function
        # over an arbitrary parameter That parameter could of course
        # be inh or exh!
        pass

    def slope(self):
        pass

    def amplitude(self):
        pass

    def speech_delay(self):
        pass

    # == Private methods ==

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

# == TODO ==
# 1. feature estimation
