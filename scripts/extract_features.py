#!/usr/bin/env python3

import csv
import os

import tgt
import rip


def main(wav_path, outfile_path, speech_path=None,
         speech_tier_name=None):

    if speech_path is not None and speech_path is not None:
        tg = tgt.read_textgrid(speech_path)
        speech = tg.get_tier_by_name(speech_tier_name)
    else:
        speech = None

    # Read the respiratory signal, detrend it, identify cycles and
    # holds.
    resp = rip.RIP.from_wav(wav_path, speech=speech)
    resp.remove_baseline()
    resp.find_cycles(include_holds=True)
    resp.find_holds()
    resp.estimate_range()
    resp.estimate_rel(30)
    resp.save_annotations('breath.TextGrid')

    fname = os.path.splitext(os.path.basename(wav_path))[0]

    # For each inhalation and exhalation, extract the respiratory
    # cycles.

    features = []
    for i, seg in enumerate(resp.segments):

        start = seg.start_time
        end = seg.end_time
        label = seg.text

        features_seg = {'file': fname, 'start': start, 'end': end,
                        'segment': label}
        # Odd-numbered rows correspond to inhalations and even-numbered
        # rows correspond to exhalations.
        if label == 'out':
            cycle_start = resp.segments[i - 1].start_time
            features_seg['duty_cycle'] = (end - start) / (end - cycle_start)
        else:
            cycle_end = resp.segments[i + 1].end_time
            features_seg['duty_cycle'] = (end - start) / (cycle_end - start)
        features_seg['duration'] = end - start
        features_seg['slope'] = resp.extract_slope(start, end)
        features_seg['amplitude'] = resp.extract_amplitude(start, end)
        features_seg['vol_start'] = resp.extract_level(start)
        features_seg['vol_end'] = resp.extract_level(end)
        holds = resp.holds.get_annotations_between_timepoints(
            start, end, left_overlap=True, right_overlap=True)
        features_seg['nholds'] = len(holds)

        if len(holds):
            holds_dur = sum(h.end_time - h.start_time for h in holds)
            holds_dur -= max(0, start - holds[0].start_time)
            holds_dur -= max(0, holds[-1].end_time - end)
        else:
            holds_dur = 0

        if speech is not None:

            # Extract: time lag, interval before and after: duration,
            # start, end level, slope
            pass

        features.append(features_seg)

    with open(outfile_path, 'w') as fout:
        csv_out = csv.DictWriter(fout, fieldnames=features[0].keys())
        csv_out.writeheader()
        csv_out.writerows(features)


if __name__ == '__main__':
    main('../data/resp.wav', 'features_cycles.csv')
