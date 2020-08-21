import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import *
from utils import save, load
from trainer import Trainer
from data_read import Data_read
from adamp import AdamP
from torch.utils.tensorboard import SummaryWriter
import torchsummary

import argparse

parser = argparse.ArgumentParser(description="Main", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-4, type=float, dest='lr')
parser.add_argument("--batch_size", default=4, type=int, dest='batch_size')
parser.add_argument("--num_epoch", default=100, type=int, dest='num_epoch')
parser.add_argument("--raw_data_dir", default='./180285_405369_bundle_archive', type=str, dest='raw_data_dir')
parser.add_argument("--data_dir", default='./Dataset', type=str, dest='data_dir')
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest='ckpt_dir')
parser.add_argument("--log_dir", default='./log_dir', type=str, dest='log_dir')
parser.add_argument("--mode", default='train', type=str, dest='mode')
parser.add_argument("--data_reading", default=False, type=bool, dest='data_reading')


args = parser.parse_args()

##parameters
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
raw_data_dir = args.raw_data_dir
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
mode = args.mode
data_reading = args.data_reading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Ensemble(MnasNet(), Efficientnet_lite(), MobileNet()).to(device)

fn_loss = nn.CrossEntropyLoss().to(device)
# optim = AdamP(net.parameters(), lr=lr * batch_size/256,  betas=(0.9, 0.999), weight_decay=1e-2)
optim = torch.optim.Adam(net.parameters(), lr=lr * batch_size/256, weight_decay=1e-2)

def main():
    if data_reading:
        if mode == 'train':
            Data_read(raw_data_dir,data_dir) #for first train
            # torchsummary.summary(net, (3, 128, 128))
            Trainer(mode, device, data_dir, net, batch_size, fn_loss, optim, lr, num_epoch, ckpt_dir, log_dir)
        else:
            raise NameError('Check data_reading and mode parser')
    else:
        pytorch_total_params = sum(p.numel() for p in net.parameters())
        # torchsummary.summary(net, (3, 256, 256))
        print(pytorch_total_params)
        Trainer(mode, device, data_dir, net, batch_size, fn_loss, optim, lr, num_epoch, ckpt_dir, log_dir)

if __name__ == '__main__':
    main()








