from os import path
from math import inf
from time import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision
from torchvision import transforms
import librosa
from PIL import Image

from Trainer import Trainer



class AudioBits(Dataset):
    def __init__(self, annotations):
        super(AudioBits, self).__init__()
        self.df = annotations

    def __getitem__(self, ind):
        fp, start, end, tag = self.df.iloc[ind]  # the fourth value is a number now

        clip, sr = sf.read(fp, dtype='int16')
        bit = clip[int(start):int(end)]
        bit = torch.Tensor(bit.reshape((1, -1)))
        return bit, tag

    def __len__(self):
        return len(self.df)


class Spec(Dataset):
    def __init__(self, data, trans):
        super(Spec, self).__init__()
        self.trans = trans
        self.df = data

    def norm(self, spec):
        min_num = np.amin(spec)
        max_num = np.amax(spec)
        return np.divide(np.add(spec, -min_num), max_num-min_num)

    def get_spec(self, data):
        stft = librosa.stft(data, n_fft=256, hop_length=256//2+1)  # TODO: how do I pick this number?
        rstft, _ = librosa.magphase(stft)
        spec = librosa.amplitude_to_db(rstft)
        return (self.norm(spec) * 255).astype('uint8')

    def __getitem__(self, ind):
        fp, start, end, tag = self.df.iloc[ind]

        clip, sr = librosa.load(fp, sr=16000)
        bit = clip[int(start):int(end)]

        img = Image.fromarray(self.get_spec(bit)).convert('RGB')

        if self.trans:
            img = self.trans(img)  # do transforms on PIL image

        return img, tag

    def __len__(self):
        return len(self.df)

transformation = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class PretrainedModel(torch.nn.Module):

    def __init__(self, num_classes):
        super(PretrainedModel, self).__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, num_classes) 
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d((1,1)) 

    def forward(self, x, **kwargs):
        return self.model(x)


class Model(torch.nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels=1, 
            out_channels=16,
            kernel_size=64,
            stride=2
        )

        self.conv2 = torch.nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=32,
            stride=2
        )

        self.conv3 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=64,  
            kernel_size=16,
            stride=2
        )

        # self.conv4 = torch.nn.Conv1d(
        #     in_channels=64,
        #     out_channels=128,  
        #     kernel_size=8,
        #     stride=2
        # )

        # self.conv5 = torch.nn.Conv1d(
        #     in_channels=128,
        #     out_channels=256,  
        #     kernel_size=8,
        #     stride=2
        # )

        self.fc1 = torch.nn.Linear(928, 128)  # for .5 second stride 2
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        f = F.relu(self.conv1(x))
        # print('after conv1:   ', f.size())
        x = F.max_pool1d(f, kernel_size=8, stride=8)
        # print('after maxpool: ', x.size())
        f = F.relu(self.conv2(x))
        # print('after conv2:   ', f.size())
        x = F.max_pool1d(f, kernel_size=8, stride=8)
        # print('after maxpool: ', x.size())
        f = F.relu(self.conv3(x))
        # print('after conv3:   ', f.size())
        # x = F.max_pool1d(f, kernel_size=2, stride=2)
        # print('after maxpool: ', x.size())
        # f = F.relu(self.conv4(x))
        # print('after conv4:   ', f.size())
        # x = F.max_pool1d(f, kernel_size=2, stride=2)
        # print('after maxpool: ', x.size())
        # f = F.relu(self.conv5(x))
        # print('after conv5:   ', f.size())
        # x = F.max_pool1d(f, kernel_size=2, stride=2)
        # print('after maxpool: ', x.size())
        x = x.reshape(x.size()[0], -1)
        # print('after convolutional layers: ', x.size())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc1(x)
        return x


def strat_test(df, use_imgs, b_size, o_type, lr, m, num_e):
    start_time = time()

    # add an id column for training with numbers
    tags = list(df.tag.unique())
    if 'bg' in tags: 
        tags = np.roll(tags, -1 * tags.index('bg'))
    ids = list(range(len(tags)))
    tag_to_id = dict(zip(tags, ids))
    new_col = df['tag'].apply(lambda t: tag_to_id[t])
    df['id'] = pd.DataFrame(new_col)

    # stratified train test split  
    X, y = df[['fp', 'start', 'end']], df['id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y)
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_test, y_test], axis=1)

    # get data loaders
    if use_imgs:
        train_data = Spec(train_df, transformation)
        valid_data = Spec(valid_df, transformation)
    else:
        train_data = AudioBits(train_df)
        valid_data = AudioBits(valid_df)
    train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=b_size, shuffle=True, num_workers=4)
    print('Loaded data in {:.2f} seconds.'.format(time()-start_time))

    # get net
    if use_imgs:
        net = PretrainedModel(len(tags))
    else:
        net = Model(len(tags))
    print(net)

    if o_type == 'sgd':
        opt = torch.optim.SGD(net.parameters(), lr, momentum=m)
    else:
        opt = torch.optim.Adam(net.parameters(), lr)

    wts = pd.DataFrame(df['id'].value_counts().sort_index()).id.apply(lambda x: 1/x).values
    loss_fun = torch.nn.CrossEntropyLoss(torch.Tensor(wts).to('cpu'))

    sch = None
    trainer = Trainer(net, train_loader, valid_loader, len(tags), opt, sch, loss_fun, 'cpu')
    trainer.train(num_e, num_e)
    trainer.graph_loss()
    torch.save(trainer.best_net.state_dict(), 'img_model.pth')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str,
        help='Full path including filename of csvfile.')

    parser.add_argument('-b', dest='b_size', default=16, type=int,
        help='batch size.')
    parser.add_argument('-o', dest='opt', default='adam', type=str,
        help='optimizer you want to use (sgd | adam)')
    parser.add_argument('-l', dest='lr', default=1e-4, type=float,
        help='learning rate')
    parser.add_argument('-m', dest='momentum', default=.9, type=float,
        help='momentum (only used for -o sgd)')
    parser.add_argument('-e', dest='num_epochs', default=10, type=int,
        help='number of epochs to run net')
    parser.add_argument('-i', dest='use_imgs', default=False, type=bool,
        help='whether or not to train with spectrograms')
    
    args = parser.parse_args()

    # raise exceptions if given illegal arguments
    if '.csv' not in args.datapath:
        raise Exception('datapath must be a csvfile (got {})'.format(args.datapath))
    df = pd.read_csv(args.datapath)

    # if not np.array_equal(df.columns.values, ['fp', 'start', 'end', 'tag', 'id']):
    #     raise Exception('the given csv file must have the following columns: [\'fp\' \'start\' \'end\' \'tag\'] (got {})'.format(df.columns.values))
    df = df.dropna()  # remove anything with unknown values

    if not (args.opt == 'adam' or args.opt == 'sgd'):
        raise Exception('optimizer must be \'adam\' or \'sgd\' (got {})'.format(args.opt))
    if args.momentum < 0 or args.momentum > 1:
        raise ValueError('momentum must be between 0 and 1 (got {})'.format(args.momentum))
    if args.lr < 0 or args.lr > 1:
        raise ValueError('learning rate must be between 0 and 1 (got {})'.format(args.lr))
    if args.num_epochs < 0:
        raise ValueError('number of epochs must be a positive integer (got {})'.format(args.num_epochs))

    print(df.tag.value_counts())
    strat_test(df, args.use_imgs, args.b_size, args.opt, args.lr, args.momentum, args.num_epochs)
