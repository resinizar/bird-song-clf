import sys
import csv
from os import path
from copy import deepcopy as cpy
from math import ceil

import numpy as np
import sounddevice as sd
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from tinytag import TinyTag
from PIL import Image

from train_model import Model, PretrainedModel, transformation

FALSE_POS = 1
FALSE_NEG = -1



def metadata(fp):
    """
    Parses the comment provided in recordings by audio moth.
    fp - str - full file path
    returns - a dictionary of the provided information
    """
    comment = TinyTag.get(fp).comment

    if comment:
        split_up = comment.split(' ')
        time = split_up[2]
        date = split_up[3]
        timezone = split_up[4]
        am_id = split_up[7]
        gain_setting = split_up[11]
        battery_level = split_up[16]

        return {
            'time': time,
            'date': date,
            'timezone': timezone,
            'id': am_id,
            'gain': gain_setting,
            'battery': battery_level
        }
    else:   # HACK FIX
        return {
            'time': '',
            'date': '',
            'timezone': '',
            'id': '',
            'gain': '',
            'battery': ''
        }


def score(pred, gt, chunk_size):
    diff = np.subtract(pred, gt)

    counts = {
        FALSE_POS: 0,
        FALSE_NEG: 0,
    }

    contig = 1
    prev = diff[0]
    for curr in diff[1:]:
        if curr == prev:
            contig += 1
        else:
            if contig >= chunk_size:
                if prev != 0:
                    counts[prev] += contig
            contig = 1
        prev = cpy(curr)
    if contig >= chunk_size:
        if prev != 0:
            counts[prev] += contig

    return {
        'false pos': round(counts[FALSE_POS] / len(diff) * 100, 1),
        'false neg': round(counts[FALSE_NEG] / len(diff) * 100, 1)
    }


def norm(spec):
    """
    spec - 2d numpy array - representation of data in frequency domain
    return - 2d numpy array - same shape, normalized between 0 and 1
    """
    min_num = np.amin(spec)
    max_num = np.amax(spec)
    return np.divide(np.add(spec, -min_num), max_num-min_num)


def get_spec(data):
    stft = librosa.stft(data, n_fft=1024, hop_length=1024//2+1)
    rstft, _ = librosa.magphase(stft)
    spec = librosa.amplitude_to_db(rstft)
    spec = norm(spec)  # norm between 0 and 1
    spec = spec[np.where(np.sum(spec, axis=1) > 1)]
    spec = np.flipud(spec)  # flip so low sounds are on bottom
    return spec


def get_img(data):
    stft = librosa.stft(data, n_fft=256, hop_length=256//2+1)
    rstft, _ = librosa.magphase(stft)
    spec = librosa.amplitude_to_db(rstft)
    spec = (norm(spec) * 255).astype('uint8')
    img = Image.fromarray(spec).convert('RGB')
    return transformation(img)


def create_vis(data, preds, chunk_size, gt_df=None):
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

    if gt_df is not None:
        gt_vis = np.zeros((int(h * .1), w))

        for index, row in gt_df.iterrows():
            _, start, end, _, _ = row
            start_spec = int(start / len(data) * w)
            end_spec =   int(end /   len(data) * w)
            gt_vis[:, start_spec : end_spec] = 1
        to_stack.append(gt_vis)
        
    return np.vstack((to_stack)), score(pred_vis[0], gt_vis[0], chunk_size * w / len(data))


def save_vis(img, score, fig_h):
    h, w = img.shape
    fig_w = max(round(fig_h * w / h), 1)
    plt.figure(figsize=(fig_w, fig_h))
    plt.axis('off')
    plt.set_title(score, fontsize=20)
    plt.imshow(img, cmap='gray')
    plt.savefig(out_fp, bbox_inches='tight')


def make_pred(model, data, use_img):
    if use_img:
        data = get_img(data)
        c, h, w = data.shape
        zs = model(data.view(-1, c, h, w))
    else:
        data = torch.Tensor(data)
        zs = model(data.view(1, 1, -1))

    _, pred = torch.max(zs, dim=1)
    return pred.item(), zs.detach().numpy()[0]
    

def live_record(model, use_img, block_dur, sr, fig_h, out_fp):
    data = []
    preds = []
    chunk_size = round(block_dur * sr)

    def callback(indata, frames, time, status):
        if status:
            print(status)
        if any(indata):
            data.append(cpy(indata))
            pred, zs = make_pred(model, indata, use_img)
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
        img, score = create_vis(np.array(data).reshape(-1), preds, chunk_size)
        save_vis(img, score, fig_h)

        # save wav file 
        fn, ext = path.splitext(out_fp)
        wav_save  = fn + '.wav'
        sf.write(wav_save, np.array(data).reshape(-1), sr)
        sys.exit()

    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))
        sys.exit()


def use_wavfile(model, use_img, wav_fp, gt_df, block_dur, sr):
    data, read_sr = sf.read(wav_fp, dtype='int16')
    data = data.astype(np.float32)
    assert sr == read_sr

    preds = []
    chunk_size = round(block_dur * sr)
    for i in range(0, len(data), chunk_size):
        block = data[i : i + chunk_size]
        if len(block) == chunk_size:
            pred, zs = make_pred(model, block, use_img)
            preds.append(pred)

    _, fn = path.split(wav_fp)
    df = gt_df[gt_df['fp'].str.contains(fn)]

    return create_vis(data, preds, chunk_size, df)


def use_gt(model, use_img, gt_fp, num, block_dur, sr, fig_height, out_fp):
    gt_df = pd.read_csv(gt_fp)
    gt_df = gt_df.dropna()

    if not num:
        fps = gt_df.fp.unique()
    else:
        fps = np.random.choice(gt_df.fp.unique(), num)

    f, axarray = plt.subplots(ncols=2, nrows=ceil(len(fps)/2))
    f.set_size_inches(10*fig_height, len(fps)*fig_height)
    f.tight_layout()
    false_pos = []
    false_neg = []
    for fp, ax in zip(fps, axarray.ravel()):
        img, score = use_wavfile(model, use_img, fp, gt_df, block_dur, sr)
        false_pos.append(score['false pos'])
        false_neg.append(score['false neg'])
        info = metadata(fp)
        ax.axis('off')
        ax.set_title('{} {} {} {} {}'.format(
            path.split(fp)[1], info['time'], info['timezone'], info['date'], score), fontsize=20)
        ax.imshow(img, cmap='gray')
    f.suptitle('mean false pos {} mean false neg {}'.format(np.mean(false_pos), np.mean(false_neg)), fontsize=30)
    plt.savefig(out_fp)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=str,
        help='file path to model to load')
    parser.add_argument('-w', '--wavfile', dest='wavfile', type=str,
        help='file path to wavfile to load')
    parser.add_argument('-g', '--ground-truth', dest='gt', type=str,
        help='file path to csv with gt')
    parser.add_argument('-r', '--num-rand', dest='rand', default=None, type=int,
        help='number of random files to select from ground truth csv, if not given does all')
    parser.add_argument('-b', '--block-dur', dest='block_dur', type=float, default=1.,
        help='duration in seconds to give net at a time')
    parser.add_argument('-s', '--sr', dest='sr', type=int, default=16000,
        help='sampling rate to load file at')
    parser.add_argument('-f', '--fig-height', dest='fig_height', type=int, default=3,
        help='sampling rate to load file at')
    parser.add_argument('-o', '--output', dest='out_fp', type=str, default='./results/test.png',
        help='filepath of visualization (saved by matplotlib)')
    parser.add_argument('-i', '--use-img', dest='use_img', type=bool, default=False,
        help='filepath of visualization (saved by matplotlib)')

    args = parser.parse_args()

    if args.use_img:
        model = PretrainedModel(2)
    else:
        model = Model(2)
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    model.eval()

    if args.gt and args.wavfile:
        img = use_wavfile(model, args.use_img, args.wavfile, args.gt, args.block_dur, args.sr, args.fig_height, args.out_fp)
        save_vis(img, score, args.fig_height)
    elif args.gt and not args.wavfile:
        use_gt(model, args.use_img, args.gt, args.rand, args.block_dur, args.sr, args.fig_height, args.out_fp)
    else:
        live_record(model, args.use_img, args.block_dur, args.sr, args.fig_height, args.out_fp)
