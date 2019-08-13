##################################################################################
#    Gabriel Cano                                                                #
##################################################################################
#    Class that handles training and testing for a pytorch net.                  #
#                                                                                #
#                                                                                #
##################################################################################
import torch
import matplotlib.pyplot as plt
from copy import deepcopy as cpy
from time import time
import os
import numpy as np



class Trainer:
    def __init__(self, net, tr_data, vs_data, num_classes, opt, sch, loss_fun, device='cpu'):
        self.net = net
        self.tr_data = tr_data
        self.vs_data = vs_data
        self.opt = opt
        self.sch = sch
        self.loss_fun = loss_fun
        self.num_classes = num_classes
        self.tr_loss = []
        self.vs_loss = []
        self.device = device
        self.best_net = net
        self.epoch_num = 0

    def make_predictions(self, ts_data, net):
        net.eval()
        pass
        
    def test_accuracy(self, data, net):
        net.eval()

        total = 0
        correct = 0
        running_loss = 0.
        with torch.no_grad():
            for i, (xs, ts) in enumerate(data):
                xs, ts = xs.to(self.device), ts.to(self.device)
                output_energies = net(xs)
                _, pred = torch.max(output_energies, dim=1)
                total += ts.size(0)
                correct += (pred == ts).sum().item()

        return correct / total
            
    def graph_loss(self):
        plt.plot(list(range(len(self.tr_loss))), self.tr_loss, '#4d8978')  # dark color
        plt.plot(list(range(len(self.vs_loss))), self.vs_loss, '#85c1a8')
        plt.xlabel('# epochs')
        plt.ylabel('loss')
        plt.savefig('./graphs/loss.png'.format(), dpi=300)
        # plt.show()
        plt.clf()
        
    def class_accuracy(self, data):
        self.best_net.eval()
        correct = [0 for i in range(self.num_classes)]
        total = [0 for i in range(self.num_classes)]
        with torch.no_grad():
            for (xs, ts) in data:
                xs, ts = xs.to(self.device), ts.to(self.device)
                output_energies = self.best_net(xs)
                _, pred = torch.max(output_energies, dim=1)
                c = (pred == ts).squeeze()
                for i in range(len(xs)):
                    t = ts[i]
                    correct[t] += c[i].item()
                    total[t] += 1
        for i in range(self.num_classes):
            if total[i]:
                print('{}:\t{}/{}\t{}'.format(i, correct[i], total[i], round(100*correct[i]/total[i], 1)))
            else:
                print('{}:\tnone in testing'.format(i))

    def train(self, max_epochs, patience):
        print('{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}'.format('epoch', 'tr_acc', 'vs_acc', 'tr_loss', 'vs_loss', 'time'))
        self.net.to(self.device)
        
        count_no_change = 0
        best_val_acc = 0
        
        for epoch in range(max_epochs):
            self.epoch_num += 1
            start_time = time()
            self.net.train()
            if self.sch:
                self.sch.step()

            # preds = []  # delete

            running_loss = 0.
            for i, (xs, ts) in enumerate(self.tr_data):
                xs, ts = xs.to(self.device), ts.to(self.device)
                self.opt.zero_grad()
                zs = self.net(xs)
                loss = self.loss_fun(zs, ts)

                # _, pred = torch.max(zs, dim=1)  # delete
                # preds.extend(pred.cpu().detach().numpy())  # delete

                loss.backward()
                self.opt.step()
                running_loss += loss.item()/len(xs)
            curr_tr_loss = round(running_loss/(i+1), 4)
            curr_tr_acc = round(self.test_accuracy(self.tr_data, self.net), 4)
            self.tr_loss.append(curr_tr_loss) 

            # print('% of 1s predicted: ', round(np.sum(preds) / len(preds), 3))

            self.net.eval()
            running_loss = 0.
            for i, (xs, ts) in enumerate(self.vs_data):
                xs, ts = xs.to(self.device), ts.to(self.device)
                zs = self.net(xs)
                _, pred = torch.max(zs, dim=1)
                loss = self.loss_fun(zs, ts)
                running_loss += loss.item()/len(xs)
            curr_vs_loss = round(running_loss/(i+1), 4)
            curr_vs_acc = round(self.test_accuracy(self.vs_data, self.net), 4)
            self.vs_loss.append(curr_vs_loss)

            epoch_time = round(time() - start_time, 1)
            print('{:8}{:8}{:8}{:8}{:8}{:8}'.format(self.epoch_num, curr_tr_acc, curr_vs_acc, curr_tr_loss, 
                curr_vs_loss, epoch_time))           
            
            if curr_vs_acc > best_val_acc:
                best_val_acc = cpy(curr_vs_acc)
                self.best_net = cpy(self.net)
                count_no_change = 0
            else:
                count_no_change += 1
            
            if count_no_change == patience:
                break  
        print('Finished Training')

    def save_checkpoint(self, fp=None):
        state = {'epoch': self.epoch_num, 
        'best_net': self.best_net.state_dict(),
        'curr_net': self.net.state_dict(), 
        'opt': self.opt.state_dict()
        }

        if fp is None:
            highest = -1
            for filename in os.listdir('./checkpoints'):
                if '.pth.tar' in filename:
                    ind = int(filename.split('.')[0].split('-')[-1])
                    if ind > highest:
                        highest = cpy(ind)

            torch.save(state, 'checkpoint-{}.pth.tar'.format(highest + 1))
        else:
            torch.save(state, fp)

    def load_checkpoint(self, filename=None):
        if not filename:
            filename = os.listdir('./checkpoints')[-1]
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['curr_net'])
        self.best_net.load_state_dict(checkpoint['best_net'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.epoch_num = checkpoint['epoch_num']
