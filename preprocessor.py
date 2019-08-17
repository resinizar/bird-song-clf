##################################################################################
#    Gabriel Cano                                                                #
##################################################################################
#    A script to chop up or pad audio clips to prepare them for a net.           #
#                                                                                #
#                                                                                #
##################################################################################
import csv
from math import floor
import librosa
import pandas as pd
import numpy as np
import os.path
import pandas as pd
from random import randint
from os import path
import yaml


def chop(clip, displacement, splice_size, stride, fp, tag):
    if clip is None:
        return []
    if splice_size > len(clip):  # if shorter than duration, need padding
        return []
    i = 0
    new_rows = []
    while True:
        chop_start = round(i * stride)
        chop_end = chop_start + splice_size
        
        if chop_end <= len(clip):  # if haven't reached end
            new_rows.append([fp, displacement + chop_start, displacement + chop_end, tag])
            i += 1
        else:  # if I am at the end
            overhang = len(clip) - chop_end
            if overhang / splice_size <= .5:
                new_rows.append([fp, displacement + len(clip) - splice_size, displacement + len(clip), tag])  # add chop at end
            return new_rows


def main(df, sr, dur, overlap, out_fp):
    splice_size = round(dur * sr)
    stride = round((1 - overlap) * splice_size)

    rows = []
    rows.append(['fp', 'start', 'end', 'tag'])
    for fp in df.fp.unique():  # for every file in dataframe
        print('processing ', fp)

        clip, _ = librosa.load(fp, sr=sr)  # read clip in desired sample rate
        curr_clip_ind = 0  # start at beginning of each file
        subset = df[df.fp.str.contains(fp)].sort_values('start')  # get the rows
        for _, (_, s_ind, e_ind, recorded_sr, tag) in subset.iterrows():
            s_ind, e_ind, recorded_sr = int(s_ind), int(e_ind), int(recorded_sr)
            if sr > recorded_sr:
                raise ValueError('''Desired samplerate {} is higher than the recorded samplerate 
                                    {} of file {}.'''.format(sr, recorded_sr, fp))
            bg_clip = clip[curr_clip_ind:s_ind] if curr_clip_ind < s_ind else None
            selection = clip[s_ind : e_ind]

            rows.extend(chop(bg_clip, curr_clip_ind, splice_size, stride, fp, 'bg'))
            rows.extend(chop(selection, s_ind, splice_size, stride, fp, tag))
            curr_clip_ind = e_ind
        rows.extend(chop(clip[curr_clip_ind:], curr_clip_ind, splice_size, stride, fp, 'bg'))

    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.to_csv(os.path.join(out_fp), index=False)

    # save metadata
    d = {
        'sr': sr,
        'dur': dur,
        'overlap': overlap
    }
    with open(path.splitext(out_fp)[0] + '.yaml', 'w') as file:
        yaml.dump(d, file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str,
        help='full path to the csvfile where the info is')
    parser.add_argument('samplerate', type=int,
        help='the sample rate to read files')
    parser.add_argument('dur', type=float, 
        help='time in seconds that each training clip should last')
    parser.add_argument('overlap', type=float,
        help='fraction of overlap, e.g. .5 for 50%% overlap')

    parser.add_argument('-o', '--output', dest='out_fp', type=str, default='train.csv',
        help='name of csv file to save new data to')
    
    args = parser.parse_args()

    # raise exceptions if given illegal arguments
    if '.csv' not in args.datapath:
        raise Exception('datapath must be a csvfile (got {})'.format(args.datapath))
    if args.samplerate < 0:
        raise ValueError('samplerate must be a positive integer (got {})'.format(args.samplerate))
    if args.overlap > 1 or args.overlap < 0:
        raise ValueError('overlap must be between 0 and 1 (got {})'.format(args.overlap))
    if args.out_fp:
        if '.csv' not in args.out_fp:
            raise Exception('output file path must be a csvfile (got {})'.format(args.out_fp))
    
    df = pd.read_csv(args.datapath)
    if not np.array_equal(df.columns.values, ['fp', 'start', 'end', 'sr', 'tag']):
        raise Exception('the given csv file must have the following columns: [\'fp\' \'start\' \'end\' \'sr\' \'tag\'] (got {})'.format(df.columns.values))
    df = df.dropna()  # remove anything with unknown values

    main(df, args.samplerate, args.dur, args.overlap, args.out_fp)
