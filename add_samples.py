from os import path

import pandas as pd
import librosa
import yaml

from preprocessor import chop



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str,
        help='list of files in txt doc')
    parser.add_argument('annopath', type=str,
        help='csv file to add to')
    parser.add_argument('-t', '--tag', dest='tag', type=str, default='bg',
        help='specify how the generated data should be tagged')

    args = parser.parse_args()

    if '.txt' not in args.datapath:
        raise Exception('datapath must be a txt file (got {})'.format(args.datapath))
    if '.csv' not in args.annopath:
        raise Exception('annopath must be a csv file (got {})'.format(args.annopath))

    base_path, fn = path.split(args.annopath)
    only_fn, ext = path.splitext(fn)

    with open(args.datapath, 'r') as f:
        fps = f.read().split('\n')

    with open(path.splitext(args.annopath)[0] + '.yaml', 'r') as file:
        metadata = yaml.load(file, Loader=yaml.BaseLoader)
        sr = int(metadata['sr'])
        dur = float(metadata['dur'])
        overlap = float(metadata['overlap'])

    rows = []
    for fp in fps[:-1]: 
        clip, _ = librosa.load(fp, sr=sr)  # read clip in desired sample rate
        curr_clip_ind = 0  # start at beginning of each file
        splice_size = round(dur * sr)
        stride = round((1 - overlap) * splice_size)
        rows.extend(chop(clip, curr_clip_ind, splice_size, stride, fp, args.tag))

    old_rows = pd.read_csv(args.annopath)
    new_rows = pd.DataFrame(rows, columns=old_rows.columns)
    new_rows.to_csv('./annos/extra_bg.csv', index=False)
    savepath = path.join(base_path, only_fn + '_{}'.format(len(new_rows)) + ext)
    pd.concat([new_rows, old_rows]).to_csv(savepath, index=False)
