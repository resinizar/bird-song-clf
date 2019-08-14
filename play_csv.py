import sounddevice as sd
import soundfile as sf 
import pandas as pd 
from time import sleep


def main(df):
    for i, row in df.iterrows():
        print(row)
        fp, start, end, sr, tag = row
        data, sr = sf.read(fp, start=int(start), stop=int(end))
        dur_in_sec = len(data) / sr
        print('dur: ', dur_in_sec)
        print()
        sd.play(data, sr)
        sleep(dur_in_sec)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str,
        help='full path to the csvfile where the info is')  
    parser.add_argument('-c', '--col-names', dest='cols', default=[], nargs='+', type=str,
        help='specific cols to include')
    args = parser.parse_args()

    if '.csv' not in args.datapath:
        raise Exception('datapath must be a csvfile (got {})'.format(args.datapath))

    df = pd.read_csv(args.datapath)

    if args.cols:
        df = df.loc[df.tag.isin(args.cols)]

    main(df)
