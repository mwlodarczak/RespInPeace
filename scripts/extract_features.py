#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# extract_features - Extract exhalatory features.
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

import csv
import os
import argparse

from rip import Resp


def parse_arguments():
    '''Parse command-line arguments'''

    argparser = argparse.ArgumentParser(
        description='Analyse a respiratory recording..')
    argparser.add_argument('wav_path', type=str,
                           help='Path to input WAV file')
    argparser.add_argument('out_dir', type=str,
                           help='Path to output directory')
    return argparser.parse_args()


def main(wav_path, out_dir):

    # Read the respiratory signal, detrend it, identify cycles and
    # holds.
    resp = Resp.from_wav(wav_path)
    resp.remove_baseline(method='als')
    resp.find_cycles(include_holds=True)
    resp.samples = resp.samples - resp.estimate_rel(method='dynamic')
    resp.estimate_range()

    fname = os.path.splitext(os.path.basename(wav_path))[0]
    resp.save_annotations(os.path.join(out_dir, fname + '_resp.TextGrid'))

    features = []
    for i, seg in enumerate(resp.segments):

        start = seg.start_time
        end = seg.end_time
        label = seg.text

        features_seg = {'file': fname, 'start': start, 'end': end,
                        'segment': label}
        features_seg.update(resp.extract_features(start, end))
        if label == 'out':
            cycle_start = resp.segments[i - 1].start_time
            features_seg['duty_cycle'] = (end - start) / (end - cycle_start)
        else:
            cycle_end = resp.segments[i + 1].end_time
            features_seg['duty_cycle'] = (end - start) / (cycle_end - start)
        holds = resp.holds.get_annotations_between_timepoints(
            start, end, left_overlap=True, right_overlap=True)
        features_seg['nholds'] = len(holds)

        if len(holds) > 0:
            holds_dur = sum(h.end_time - h.start_time for h in holds)
            holds_dur -= max(0, start - holds[0].start_time)
            holds_dur -= max(0, holds[-1].end_time - end)
        else:
            holds_dur = 0

        features.append(features_seg)

    with open(os.path.join(out_dir, fname + '_feat.csv'), 'w') as fout:
        csv_out = csv.DictWriter(fout, fieldnames=features[0].keys())
        csv_out.writeheader()
        csv_out.writerows(features)


if __name__ == '__main__':
    args = parse_arguments()
    main(args.wav_path, args.out_dir)
