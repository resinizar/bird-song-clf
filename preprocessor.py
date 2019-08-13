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


def main(df, sr, dur, overlap, bg_amount, out_fp):
    new_rows = []
    new_rows.append(['fp', 'start', 'end', 'tag'])

    skipped = 0
    
    for _, (fp, s_ind, e_ind, recorded_sr, tag) in df.iterrows(): # for each row
        if sr > recorded_sr:
            raise ValueError('Desired samplerate {} is higher than the recorded samplerate {} of file {}.'.format(sr, recorded_sr, fp))
        whole_clip, _ = librosa.load(fp, sr=sr)  # read clip in desired sample rate
        mini_clip = whole_clip[int(s_ind) : int(e_ind)]

        splice_size = round(dur * sr)
        stride = round((1 - overlap) * splice_size)

        if splice_size > len(mini_clip):  # if shorter than duration, need padding
            skipped += 1

        else:
            i = 0
            while True:
                chop_start = round(i * stride)
                chop_end = chop_start + splice_size
                
                if chop_end > len(mini_clip):
                    overhang = (len(mini_clip) - chop_end) / splice_size
                    if overhang <= .5:
                        new_rows.append([fp, len(mini_clip) - splice_size, len(mini_clip), tag])  # add chop at end
                    break
                else:
                    new_rows.append([fp, s_ind + chop_start, s_ind + chop_end, tag])
                    i += 1

    print('too short: ', skipped)

    if bg_amount:
        for i in range(round(len(new_rows) * bg_amount) - 1):
            new_entry = []
            while not new_entry:
                row = df.sample()  # choose a random row
                fp, sr = row.fp.values[0], int(row.sr.values[0])
                whole_clip, _ = librosa.load(fp, sr=sr)
                splice_size = round(dur * sr)
                splice_start = randint(0, len(whole_clip) - splice_size)
                splice_end = splice_start + splice_size
                for _, (_, s_ind, e_ind, _, _) in df[df['fp'].str.contains(fp)].iterrows():
                    if splice_start > s_ind and splice_start < e_ind:
                        break
                else: # (no break encountered)
                    new_entry = [fp, splice_start, splice_end, 'bg']
            new_rows.append(new_entry)


    df = pd.DataFrame(new_rows[1:], columns=new_rows[0])
    # tags = [df.tag.unique()]
    tags = ['bg', 'cotr']
    ids = np.arange(0, len(tags), 1)
    tag_to_id = pd.DataFrame(list(zip(tags, ids)), columns=['tag', 'id'])
    new_col = df['tag'].apply(lambda x: tag_to_id.loc[tag_to_id['tag'] == x].id.values[0])
    df['id'] = pd.DataFrame(new_col)
    df.to_csv(os.path.join(out_fp), index=False)


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

    parser.add_argument('-b', '--bg-amount', dest='bg_amount', type=float, default=None,
        help='amount of bg data as a fraction compared to not bg data')
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
    if args.bg_amount:
        if args.bg_amount < 0:
            raise ValueError('bg amount must be a positive integer (got {})'.format(args.bg_amount))
    if args.out_fp:
        if '.csv' not in args.out_fp:
            raise Exception('output file path must be a csvfile (got {})'.format(args.out_fp))
    
    df = pd.read_csv(args.datapath)
    if not np.array_equal(df.columns.values, ['fp', 'start', 'end', 'sr', 'tag']):
        raise Exception('the given csv file must have the following columns: [\'fp\' \'start\' \'end\' \'sr\' \'tag\'] (got {})'.format(df.columns.values))
    df = df.dropna()  # remove anything with unknown values

    main(df, args.samplerate, args.dur, args.overlap, args.bg_amount, args.out_fp)
