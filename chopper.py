##################################################################################
#    Gabriel Cano                                                                #
##################################################################################
#    A script to chop up or pad audio clips to prepare them for a net.           #
#                                                                                #
#                                                                                #
##################################################################################
from scipy.io import wavfile
import csv
from math import floor
import pandas as pd
import numpy as np
import os.path



def main(datapath, dur, overlap):
    datafile = open(datapath, 'r')
    rows = csv.reader(datafile)

    new_rows = []
    new_rows.append(['fp', 'start', 'end', 'tag'])
    
    next(rows)  # skip header
    for filepath, tag in rows: 
        sr, clip = wavfile.read(filepath)
        splice_size = round(dur * sr)
        stride = round((1 - overlap) * splice_size)

        if splice_size > len(clip):  # if shorter than duration, need padding
            pass

        else:
            i = 0
            while True:
                chop_start = round(i * stride)
                chop_end = chop_start + splice_size
                
                if chop_end > len(clip):
                    break
                else:
                    new_rows.append([filepath, chop_start, chop_end, tag])
                    i += 1
                    
            new_rows.append([filepath, len(clip) - splice_size, len(clip), tag])  # add chop at end
    datafile.close()

    # add 'id' column which is a number corresponding to a tag
    df = pd.DataFrame(new_rows[1:], columns=new_rows[0])
    tags = df.tag.unique()
    ids = np.arange(0, len(tags), 1)

    tag_to_id = pd.DataFrame(list(zip(tags, ids)), columns=['tag', 'id'])
    tag_to_id.to_csv(os.path.join(os.path.split(datapath)[0], 'tag_to_id.csv'), index=False)

    new_col = df['tag'].apply(lambda x: tag_to_id.loc[tag_to_id['tag'] == x].id.values[0])
    df['id'] = pd.DataFrame(new_col)

    # once chopping is finished make classes equal
    g = df.groupby('tag')
    df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    df.to_csv(os.path.join(os.path.split(datapath)[0], 'chopped.csv'), index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str,
        help='Full path to the csvfile where the info is.')
    parser.add_argument('dur', type=float, 
        help='The time in seconds that each clip should last.')
    parser.add_argument('overlap', type=float,
        help='The fraction of overlap, e.g. .5 for 50%% overlap.')
    
    args = parser.parse_args()

    # raise exceptions if given illegal arguments
    if '.csv' not in args.datapath:
        raise Exception('datapath must be a csvfile (got {})'.format(args.datapath))
    if args.overlap > 1 or args.overlap < 0:
        raise ValueError('overlap must be between 0 and 1 (got {})'.format(args.overlap))

    main(args.datapath, args.dur, args.overlap)
