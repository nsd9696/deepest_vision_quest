import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False)
        self.fc1 = nn.Linear(1000, 64, bias=True)
        self.fc2 = nn.Linear(64, 32, bias=True)
        self.fc3 = nn.Linear(32, 20, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.mobilenet(x)
        x = x.view(-1,1000)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SpnasNet(nn.Module):
    def __init__(self):
        super(SpnasNet,self).__init__()
        self.spnasnet = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'spnasnet_100', pretrained=False)
        self.fc1 = nn.Linear(1000, 64, bias=True)
        self.fc2 = nn.Linear(64, 32, bias=True)
        self.fc3 = nn.Linear(32, 20, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.spnasnet(x)
        x = x.view(-1,1000)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class MnasNet(nn.Module):
    def __init__(self):
        super(MnasNet, self).__init__()
        self.mnasnet = models.mnasnet1_0(pretrained=False)
        self.fc1 = nn.Linear(1000, 64, bias=True)
        self.fc2 = nn.Linear(64, 32, bias=True)
        self.fc3 = nn.Linear(32, 20, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.mnasnet(x)
        x = x.view(-1,1000)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Efficientnet_lite(nn.Module):
    def __init__(self):
        super(Efficientnet_lite, self).__init__()
        self.efficientnet_lite = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_lite0', pretrained=False)
        self.fc1 = nn.Linear(1000, 40, bias=True)
        self.fc2 = nn.Linear(40, 32, bias=True)
        self.fc3 = nn.Linear(32, 20, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.efficientnet_lite(x)
        x = x.view(-1,1000)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



class Ensemble(nn.Module):
    def __init__(self, model1, model2, model3):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.fc1 = nn.Linear(60, 40, bias=True)
        self.fc2 = nn.Linear(40, 32, bias=True)
        self.fc3 = nn.Linear(32, 20, bias=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x = torch.cat((x1,x2,x3), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x