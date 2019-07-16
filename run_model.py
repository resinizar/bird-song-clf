from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from os import path
from scipy.io import wavfile
from math import inf
from sklearn.model_selection import train_test_split
from time import time
from Trainer import Trainer
import torch.nn.functional as F



class AudioBits(Dataset):
    def __init__(self, annotations):
        """
        annotations - str: full filepath to csv file - or pd.DataFrame
        """
        super(AudioBits, self).__init__()

        if isinstance(annotations, str):
            self.df = pd.read_csv(path.join(datafolder, annotations))
        elif isinstance(annotations, pd.DataFrame):
            self.df = annotations
        else:
            raise Exception('Input to Audiobits must be a path (string) or a pandas.DataFrame')

    def __getitem__(self, ind):
        fp, start, end, tag = self.df.iloc[ind]

        sr, clip = wavfile.read(fp)
        bit = clip[start:end]
        bit = torch.Tensor(bit.reshape((1, -1)))
        return bit, tag

    def __len__(self):
        return len(self.df)


class Model(torch.nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv1d(
            in_channels=1, 
            out_channels=16,
            kernel_size=64,
            stride=4
            )

        self.conv2 = torch.nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=32,
            stride=4
            )

        self.conv3 = torch.nn.Conv1d(
            in_channels=32,
            out_channels=64,  
            kernel_size=16,
            stride=4
            )

        self.fc1 = torch.nn.Linear(1216, num_classes)
        # self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        f = F.relu(self.conv1(x))
        # print('after conv1:   ', f.size())
        x = F.max_pool1d(f, kernel_size=2, stride=2)
        # print('after maxpool: ', x.size())
        f = F.relu(self.conv2(x))
        # print('after conv2:   ', f.size())
        x = F.max_pool1d(f, kernel_size=2, stride=2)
        # print('after maxpool: ', x.size())
        f = F.relu(self.conv3(x))
        # print('after conv3:   ', f.size())
        x = F.max_pool1d(f, kernel_size=2, stride=2)
        # print('after maxpool: ', x.size())
        x = x.reshape(x.size()[0], -1)
        # print('after convolutional layers: ', x.size())
        x = self.fc1(x)
        # print('after fc1:     ', x.size())

        # x = self.fc2(x)
        # print('after fc2:     ', x.size())
        
        return x


def strat_test(datafile, b_size, o_type, lr, m):
    start_time = time()

    # stratified train test split
    df = pd.read_csv(datafile)
    num_classes = len(df.tag.unique())

    X, y = df[['fp', 'start', 'end']], df['id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y)
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_test, y_test], axis=1)

    train_data = AudioBits(train_df)
    valid_data = AudioBits(valid_df)

    train_loader = DataLoader(train_data, batch_size=b_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=b_size, shuffle=True, num_workers=4)

    print('Loaded data in {:.2f} seconds.'.format(time()-start_time))

    net = Model(num_classes)

    if o_type == 'sgd':
        opt = torch.optim.SGD(net.parameters(), lr, momentum=m)
    else:
        opt = torch.optim.Adam(net.parameters(), lr)

    loss_fun = torch.nn.CrossEntropyLoss()
    sch = None
    trainer = Trainer(net, train_loader, valid_loader, num_classes, opt, sch, loss_fun, 'cpu')
    trainer.train(10, 10)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath', type=str,
        help='Full path including filename of csvfile.')

    parser.add_argument('-b', dest='b_size', default=2, type=int,
        help='Specify batch size.')
    parser.add_argument('-o', dest='opt', default='adam', type=str,
        help='Specify which optimizer you want to use (sgd | adam).')
    parser.add_argument('-l', dest='lr', default=1e-4, type=float,
        help='Specify learning rate.')
    parser.add_argument('-m', dest='momentum', default=.9, type=float,
        help='Specify momentum (only used for -o sgd).')
    
    args = parser.parse_args()

    # raise exceptions if given illegal arguments
    if '.csv' not in args.datapath:
        raise Exception('datapath must be a csvfile (got {})'.format(args.datapath))
    if not (args.opt == 'adam' or args.opt == 'sgd'):
        raise Exception('optimizer must be \'adam\' or \'sgd\' (got {})'.format(args.opt))
    if args.momentum < 0 or args.momentum > 1:
        raise ValueError('momentum must be between 0 and 1 (got {})'.format(args.momentum))
    if args.lr < 0 or args.lr > 1:
        raise ValueError('learning rate must be between 0 and 1 (got {})'.format(args.lr))

    strat_test(args.datapath, args.b_size, args.opt, args.lr, args.momentum)

