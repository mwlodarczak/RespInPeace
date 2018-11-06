#!/usr/bin/env python3

import csv
import os
import sys

import rip


def main(wav_path, outfile_path):

    # Read the respiratory signal, detrend it, identify cycles and
    # holds.
    resp = rip.RIP.from_wav(wav_path)
    resp.remove_baseline()
    resp.find_cycles(include_holds=True)
    resp.estimate_range()
    resp.estimate_rel(30)
    resp.save_annotations('breath.TextGrid')

    fname = os.path.splitext(os.path.basename(wav_path))[0]

    # For each inhalation and exhalation, extract the respiratory
    # cycles.

    features = []
    for i, (start, end) in enumerate(resp.segments):
        features_seg = {'file': fname, 'start': start, 'end': end}
        # Odd-numbered rows correspond to inhalations and even-numbered
        # rows correspond to exhalations.
        if i % 2:
            prev_inh = resp.segments[i - 1,:]
            features_seg['duty_cycle'] = (end - start) / (end - prev_inh[0])
            features_seg['segment'] = 'exhalation'
        else:
            next_exh = resp.segments[i + 1,:]
            features_seg['duty_cycle'] = (end - start) / (next_exh[1] - start)
            features_seg['segment'] = 'inhalation'
        features_seg['duration'] = end - start
        features_seg['slope'] = resp.extract_slope(start, end)
        features_seg['amplitude'] = resp.extract_amplitude(start, end)
        features_seg['vol_start'] = resp.extract_level(start)
        features_seg['vol_end'] = resp.extract_level(end)

        features.append(features_seg)

    with open(outfile_path, 'w') as fout:
        csv_out = csv.DictWriter(fout, fieldnames=features[0].keys())
        csv_out.writeheader()
        csv_out.writerows(features)


if __name__ == '__main__':
    main('../data/resp.wav', 'features_cycles.csv')
