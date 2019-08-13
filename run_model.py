import sys
import csv
from os import path
from copy import deepcopy as cpy

import numpy as np
import sounddevice as sd
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd

from train_model import Model



def norm(spec):
    """
    spec - 2d numpy array - representation of data in frequency domain
    return - 2d numpy array - same shape, normalized between 0 and 1
    """
    min_num = np.amin(spec)
    max_num = np.amax(spec)
    return np.divide(np.add(spec, -min_num), max_num-min_num)


def make_pred(model, data):
    data = torch.Tensor(data)
    zs = model(data.view(1, 1, -1))
    _, pred = torch.max(zs, dim=1)
    return pred.item(), zs.detach().numpy()[0]

def get_spec(data):
    stft = librosa.stft(data, n_fft=1024, hop_length=1024//2+1)
    rstft, _ = librosa.magphase(stft)
    spec = librosa.amplitude_to_db(rstft)
    spec = norm(spec)  # norm between 0 and 1
    spec = spec[np.where(np.sum(spec, axis=1) > 1)]
    spec = np.flipud(spec)  # flip so low sounds are on bottom
    return spec


def create_vis(data, preds, chunk_size, fig_h, out_fp, gt_fp=None, wavfile=None):
    to_stack = []

    spec = get_spec(data)
    to_stack.append(spec)

    h, w = spec.shape
    pred_vis = np.zeros((int(h * .1), w))
    for i, pred in enumerate(preds):
        start_spec = int(i * chunk_size / len(data) * w)
        end_spec = int((i + 1) * chunk_size / len(data) * w)
        pred_vis[:, start_spec : end_spec] = pred
    to_stack.append(norm(pred_vis))

    if gt_fp:
        gt_vis = np.zeros((int(h * .1), w))
        _, fn = path.split(wavfile)
        df = pd.read_csv(gt_fp)
        df = df.dropna()
        df = df[df['fp'].str.contains(fn)]

        for index, row in df.iterrows():
            _, start, end, _, _ = row
            start_spec = int(start / len(data) * w)
            end_spec =   int(end /   len(data) * w)
            gt_vis[:, start_spec : end_spec] = 1
        to_stack.append(gt_vis)
        
    img = np.vstack((to_stack))
    h, w = img.shape
    fig_w = max(round(fig_h * w / h), 1)
    
    plt.figure(figsize=(fig_w, fig_h))
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    plt.savefig(out_fp, bbox_inches='tight')
    

def live_record(model, block_dur, sr, fig_height, out_fp):
    data = []
    preds = []
    chunk_size = round(block_dur * sr)

    def callback(indata, frames, time, status):
        if status:
            print(status)
        if any(indata):
            data.append(cpy(indata))
            pred, zs = make_pred(model, indata)
            preds.append(zs[0])
            print('z: {}\t{}'.format(zs, pred))

        else:
            print('no input')

    try:
        with sd.InputStream(device=0, channels=1, callback=callback, blocksize=chunk_size, samplerate=sr, dtype='int16'):
            print('press control C to stop recording')
            while True:
                response = input()

    except KeyboardInterrupt:
        create_vis(np.array(data).reshape(-1), preds, chunk_size, fig_height, out_fp)

        fn, ext = path.splitext(out_fp)
        wav_save  = fn + '.wav'
        sf.write(wav_save, np.array(data).reshape(-1), sr)
        sys.exit()

    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))
        sys.exit()


def use_wavfile(model, wavfile, gt_fp, block_dur, sr, fig_height, out_fp):
    data, read_sr = sf.read(wavfile, dtype='int16')
    data = data.astype(np.float32)
    assert sr == read_sr
    preds = []
    chunk_size = round(block_dur * sr)

    for i in range(0, len(data), chunk_size):
        block = data[i : i + chunk_size]
        if len(block) == chunk_size:
            pred, zs = make_pred(model, block)
            preds.append(pred)

    create_vis(data, preds, chunk_size, fig_height, out_fp, gt_fp, wavfile)


def use_gt(model, gt_fp, inds, block_dur, sr, fig_height, out_fp):
    df = pd.read_csv(gt_fp)
    df = df.dropna()

    wavs = df['fp'].unique()

    if len(inds) == 0:
        inds = [0, len(wavs)]

    for i, wav in enumerate(wavs[inds[0]:inds[1]]):
        print('Starting wav file, ', wav)
        fn, ext = path.splitext(out_fp)
        new_out_fp = '{}_{}{}'.format(fn, i, ext)
        print('Save name is: ', new_out_fp)
        use_wavfile(model, wav, gt_fp, block_dur, sr, fig_height, new_out_fp)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=str,
        help='file path to model to load')
    parser.add_argument('-w', '--wavfile', dest='wavfile', type=str,
        help='file path to wavfile to load')
    parser.add_argument('-g', '--ground-truth', dest='gt', type=str,
        help='file path to csv with gt')
    parser.add_argument('-i', '--inds', dest='inds', default=[], nargs='+', type=int,
        help='start and end index for gt files to use')
    parser.add_argument('-b', '--block-dur', dest='block_dur', type=float, default=1.,
        help='duration in seconds to give net at a time')
    parser.add_argument('-s', '--sr', dest='sr', type=int, default=16000,
        help='sampling rate to load file at')
    parser.add_argument('-f', '--fig-height', dest='fig_height', type=int, default=2,
        help='sampling rate to load file at')
    parser.add_argument('-o', '--output', dest='out_fp', type=str, default='./results/test.png',
        help='filepath of visualization (saved by matplotlib)')

    args = parser.parse_args()

    model = Model(2)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    if args.gt and args.wavfile:
        use_wavfile(model, args.wavfile, args.gt, args.block_dur, args.sr, args.fig_height, args.out_fp)
    elif args.gt and not args.wavfile:
        use_gt(model, args.gt, args.inds, args.block_dur, args.sr, args.fig_height, args.out_fp)
    else:
        live_record(model, args.block_dur, args.sr, args.fig_height, args.out_fp)
