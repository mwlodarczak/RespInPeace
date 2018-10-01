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

import math

from scipy.io import wavfile
import numpy as np
import scipy.signal

__all__ = ['RIP']


class RIP:

    def __init__(self, wav_path):

        self.samp_freq, self.resp = wavfile.read(wav_path)
        self.t = np.arange(len(self.resp)) / self.samp_freq
        self.dur = len(self.resp) / self.samp_freq

    # Alternative initializers
    @classmethod
    def from_npy():
        pass

    @classmethod
    def from_csv():
        pass

    def resample(self, resamp_freq):

        # Number of samples in the resampled signal
        resamp_num = math.floor(self.dur * resamp_freq)
        self.t, self.resp = scipy.signal.resample(
            self.resp, num=resamp_num, t=self.t)
        self.samp_freq = resamp_freq

    def detrend(self, type='linear'):

        self.resp = scipy.signal.detrend(self.resp, type=type)

    def scale(self, win_len=None, type='rolling'):

        if type == 'rolling':
            mov_mean = np.mean(self._mov_window(win_len), 1)
            mov_std = np.std(self._mov_window(win_len), 1)
            self.resp = (self.resp - mov_mean) / mov_std
        elif type == 'constant':
            self.resp = (self.resp - np.mean(self.resp)) / np.std(self.resp)
        else:
            raise Exception(
                "Unknown type. {}. Must be 'rolling' or constant".format(type))

    def find_peaks(self, delta=1):
        # TODO: Next
        pass

    def find_holds(self):
        pass

    def find_laughters(self):
        pass

    def _mov_window(self, window):
        shape = self.resp.shape[:-1] + (self.resp.shape[-1] - window + 1, window)
        strides = self.resp.strides + (self.resp.strides[-1],)
        return np.lib.stride_tricks.as_strided(
            self.resp, shape=shape, strides=strides)

    def _fft_smooth(self, win_len):
        '''Zero-phase low-pass filter using FFT convolution.

        Originally proposed in https://github.com/zelanolab/breathmetrics.
        See also: Noto T, Zhou G, Schuele S, Templer J, & Zelano C (2018)
        Automated analysis of breathing waveforms using BreathMetrics: a
        respiratory signal processing toolbox. Chemical Senses (in press)

        '''

        l = len(self.resp)
        win = np.zeros(l)
        win[math.floor((l - win_len + 1) / 2): math.floor((l + win_len) / 2)] = 1
        return scipy.signal.fftconvolve(self.resp, win, mode='same') / win_len
