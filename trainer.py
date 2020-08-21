import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_read import Data_read

from utils import *

def Trainer(mode, device, data_dir, net, batch_size, fn_loss, optim, lr, num_epoch, ckpt_dir, log_dir):

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.RandomCrop(224),
         transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_inference = transforms.Compose(
        [transforms.Resize(256), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    Trainset = torchvision.datasets.ImageFolder(root=data_dir+'/'+'train', transform=transform)
    Valset = torchvision.datasets.ImageFolder(root=data_dir+'/'+'val', transform=transform)
    Testset = torchvision.datasets.ImageFolder(root=data_dir + '/' + 'test', transform=transform_inference)
    TrainLoader = DataLoader(Trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    ValLoader = DataLoader(Valset, batch_size=batch_size, shuffle=False, num_workers=4)
    TestLoader = DataLoader(Testset, batch_size=batch_size, shuffle=False, num_workers=4)

    if mode == 'train':
        for epoch in range(1, num_epoch+1):
            net.train()
            loss_arr=[]

            for batch, data in enumerate(TrainLoader,0):

                input, label = data
                input, label = input.to(device), label.to(device)

                optim.zero_grad()
                output = net(input)
                loss = fn_loss(output,label)
                loss.backward()
                optim.step()

                loss_arr += [loss.item()]

                print("TRAIN: EPOCH %04d / %04d | BATCH %04d  | LOSS %.4f" %
                      (epoch, num_epoch, batch, np.mean(loss_arr)))

            writer_train(log_dir).add_scalar('loss', np.mean(loss_arr), epoch)

            with torch.no_grad():
                net.eval()
                loss_arr=[]

                for batch, data in enumerate(ValLoader,0):
                    input, label = data
                    input, label = input.to(device), label.to(device)

                    output = net(input)

                    loss = fn_loss(output, label)
                    loss_arr += [loss.item()]

                    print("VALID: EPOCH %04d / %04d | BATCH %04d  | LOSS %.4f" %
                          (epoch, num_epoch, batch, np.mean(loss_arr)))

                writer_val(log_dir).add_scalar('loss', np.mean(loss_arr), epoch)
                if epoch % 2 == 0:
                    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

        writer_train.close()
        writer_val.close()

    else:
        #inference
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

        with torch.no_grad():
            net.eval()
            inference_img = Testset[0][0]
            imshow(inference_img, data_dir)
            input = inference_img.unsqueeze(0).to(device)
            result = net(input).detach().cpu()
            class_idx = torch.max(result, 1)[1].numpy()
            print(Testset.classes[class_idx[0]])





