import sounddevice as sd
import soundfile as sf 
import pandas as pd 
from time import sleep
import librosa
import numpy as np
import matplotlib.pyplot as plt

def norm(spec):
    min_num = np.amin(spec)
    max_num = np.amax(spec)
    return np.divide(np.add(spec, -min_num), max_num-min_num)

def get_spec(data):
    stft = librosa.stft(data, n_fft=512, hop_length=512//2+1)  # TODO: how do I pick this number?
    rstft, _ = librosa.magphase(stft)
    spec = librosa.amplitude_to_db(rstft)
    return (norm(spec) * 255).astype('uint8')


def main(df):
    for i, row in df.iterrows():
        print(row)
        fp, tag, start, end, sr = row
        pad = .2 * sr
        data, sr = sf.read(fp, start=int(start - pad), stop=int(end + pad))
        dur_in_sec = len(data) / sr
        print('dur: ', dur_in_sec)
        print()
        sd.play(data, sr)
        # sleep(dur_in_sec)
        savename = input('Save as:')
        if savename != '':
            sf.write('{}.wav'.format(savename), data, sr)
            plt.axis('off')
            plt.imshow(get_spec(data), cmap='magma')
            plt.savefig('{}.png'.format(savename))




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
