import pandas as pd
from os import path



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str,
        help='old csv file with old paths')
    parser.add_argument('savepath', type=str,
        help='new csv file with updated paths')
    parser.add_argument('new_data_fp', type=str,
        help='the path to the directory where the audio files are')

    args = parser.parse_args()

    if '.csv' not in args.datapath:
        raise Exception('datapath must be a csvfile (got {})'.format(args.datapath))
    if '.csv' not in args.savepath:
        raise Exception('savepath must be a csvfile (got {})'.format(args.savepath))

    def change_path(ori_path):
        fp, fn = path.split(ori_path)
        new_path = path.join(args.new_data_fp, fn)
        return new_path

    df = pd.read_csv(args.datapath).dropna()
    df['fp'] = df['fp'].apply(lambda x: change_path(x))
    df.to_csv(args.savepath, index=False)
