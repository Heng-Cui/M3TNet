import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class Conv_Block(nn.Module):
    def __init__(self, channel=32, kernel=9, stride=1, padding=4):
        super(Conv_Block, self).__init__()
        self.lay1 = nn.Sequential(
            nn.Conv1d(2, 1, kernel, stride, padding, bias=False),
            nn.BatchNorm1d(1),
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )
        self.lay2 = nn.Sequential(
            nn.Conv1d(2, 32, kernel, stride, padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Conv1d(channel, channel, kernel, stride, padding),
            # nn.BatchNorm1d(channel),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            #nn.Conv1d(channel, channel // 2, kernel_size=1, stride=1),
            nn.Conv1d(32, 1, kernel, stride, padding),
            nn.BatchNorm1d(1),
            #nn.Dropout(0.5),
            nn.Sigmoid(),
        )

        self.lay3 = nn.Sequential(
            nn.Linear(channel*2, channel),
            nn.BatchNorm1d(channel),
            #nn.Dropout(0.5),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.lay3(x)


class Interaction_Block(nn.Module):
    def __init__(self, inchannel=128,k=4):
        super(Interaction_Block, self).__init__()
        self.Conv_o2e = Conv_Block(inchannel)  # TODO 这里千万要注意，因为这里是两条路，一个n2s，一个s2n，所以是连个独立的卷积层
        self.Conv_e2o = Conv_Block(inchannel)  # TODO 1*1卷积核是不会学习到任何特征的，它只能起调整通道数的作用
        self.lay_e = nn.Sequential(
            nn.Linear(inchannel, inchannel//k),
            nn.ReLU(),
            nn.BatchNorm1d(inchannel//k),
            #nn.Dropout(0.5),
        )
        self.lay_o = nn.Sequential(
            nn.Linear(inchannel, inchannel // k),
            nn.ReLU(),
            nn.BatchNorm1d(inchannel // k),
            #nn.Dropout(0.5),
        )
    def forward(self, F_e, F_o):  # F_RA：[B, T, F', C]
        # F_e = F_e.unsqueeze(1)
        # F_o = F_o.unsqueeze(1)
        F_cat = torch.cat((F_e, F_o), dim=1)  # F_cat: [B, T, F', 2*C]

        Mask_o = self.Conv_o2e(F_cat)    # [B, C, T, F']
        Mask_e = self.Conv_e2o(F_cat)    # [B, C, T, F']

        H_o2e = F_o * Mask_o   # [B, T, F', C]
        H_e2o = F_e * Mask_e   # [B, T, F', C]

        # F_e_new = F_e.squeeze() + H_o2e.squeeze()   # [B, T, F', C]
        # F_o_new = F_o.squeeze() + H_e2o.squeeze()   # [B, T, F', C]


        H_o2e = H_o2e.squeeze()
        H_e2o = H_e2o.squeeze()

        H_o2e = self.lay_e(H_o2e)
        H_e2o = self.lay_o(H_e2o)
        F_e_new = torch.cat((F_e.squeeze(), H_o2e), dim = 1)   # [B, T, F', C]
        F_o_new = torch.cat((F_o.squeeze(), H_e2o), dim = 1) # [B, T, F', C]

        return F_e_new, F_o_new



class M3TNet_SEEDIV(nn.Module):
    def __init__(self,k=4):
        super(M3TNet_SEEDIV, self).__init__()
        self.k = k
        self.f_fc1 = nn.Linear(310, 128)
        self.f_relu1 = nn.ReLU(True)
        self.f_bn1 = nn.BatchNorm1d(128)
        self.f_drop1 = nn.Dropout(0.5)

        self.f_fc2 = nn.Linear(128, 64)
        self.f_relu2 = nn.ReLU(True)
        self.f_bn2 = nn.BatchNorm1d(64)
        self.f_drop2 = nn.Dropout(0.5)

        self.f_fc3 = nn.Linear(64+64//self.k, 32)
        #self.f_fc3 = nn.Linear(64, 32)
        self.f_relu3 = nn.ReLU(True)
        self.f_bn3 = nn.BatchNorm1d(32)

        self.ef_fc1 = nn.Linear(31, 128)
        self.ef_relu1 = nn.ReLU(True)
        self.ef_bn1 = nn.BatchNorm1d(128)
        self.ef_drop1 = nn.Dropout(0.5)

        self.ef_fc2 = nn.Linear(128, 64)
        self.ef_relu2 = nn.ReLU(True)
        self.ef_bn2 = nn.BatchNorm1d(64)
        self.ef_drop2 = nn.Dropout(0.5)

        self.ef_fc3 = nn.Linear(64+64//self.k, 32)
        #self.ef_fc3 = nn.Linear(64, 32)
        self.ef_relu3 = nn.ReLU(True)
        self.ef_bn3 = nn.BatchNorm1d(32)

        self.i1 = Interaction_Block(128, self.k)
        self.i2 = Interaction_Block(64, self.k)
        self.i3 = Interaction_Block(32, self.k)

        self.c = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 4),
        )
        self.ec = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 4),
        )

        self.c_all = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 4),
        )


    def forward(self, eeg, eye):

        e = self.f_fc1(eeg)
        e = self.f_relu1(e)
        e = self.f_bn1(e)
        e = self.f_drop1(e)
        o = self.ef_fc1(eye)
        o = self.ef_relu1(o)
        o = self.ef_bn1(o)
        o = self.ef_drop1(o)

        #e, o = self.i1(e, o)

        e = self.f_fc2(e)
        e = self.f_relu2(e)
        e = self.f_bn2(e)
        #e = self.f_drop2(e)

        o = self.ef_fc2(o)
        o = self.ef_relu2(o)
        o = self.ef_bn2(o)
        #o = self.ef_drop2(o)

        e, o = self.i2(e, o)

        e = self.f_drop2(e)
        o = self.ef_drop2(o)

        e = self.f_fc3(e)
        e = self.f_relu3(e)
        e = self.f_bn3(e)

        o = self.ef_fc3(o)
        o = self.ef_relu3(o)
        o = self.ef_bn3(o)

        #e, o = self.i3(e, o)

        classifier_eeg = self.c(e)
        classifier_eye = self.ec(o)
        classifier = self.c_all(torch.cat((e, o), dim=-1))

        return classifier, e, o, classifier_eeg, classifier_eye

class M3TNet_SEED(nn.Module):
    def __init__(self,k=4):
        super(M3TNet_SEED, self).__init__()
        self.k = k
        self.f_fc1 = nn.Linear(310, 128)
        self.f_relu1 = nn.ReLU(True)
        self.f_bn1 = nn.BatchNorm1d(128)
        self.f_drop1 = nn.Dropout(0.5)

        self.f_fc2 = nn.Linear(128, 64)
        self.f_relu2 = nn.ReLU(True)
        self.f_bn2 = nn.BatchNorm1d(64)
        self.f_drop2 = nn.Dropout(0.5)

        #self.f_fc3 = nn.Linear(64+64//self.k, 32)
        self.f_fc3 = nn.Linear(64, 32)
        self.f_relu3 = nn.ReLU(True)
        self.f_bn3 = nn.BatchNorm1d(32)

        self.ef_fc1 = nn.Linear(33, 128)
        self.ef_relu1 = nn.ReLU(True)
        self.ef_bn1 = nn.BatchNorm1d(128)
        self.ef_drop1 = nn.Dropout(0.5)

        self.ef_fc2 = nn.Linear(128, 64)
        self.ef_relu2 = nn.ReLU(True)
        self.ef_bn2 = nn.BatchNorm1d(64)
        self.ef_drop2 = nn.Dropout(0.5)

        #self.ef_fc3 = nn.Linear(64+64//self.k, 32)
        self.ef_fc3 = nn.Linear(64, 32)
        self.ef_relu3 = nn.ReLU(True)
        self.ef_bn3 = nn.BatchNorm1d(32)

        self.i1 = Interaction_Block(128, self.k)
        self.i2 = Interaction_Block(64, self.k)
        self.i3 = Interaction_Block(32, self.k)

        self.c = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32+32//self.k, 3),
        )
        self.ec = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32+32//self.k, 3),
        )

        self.c_all = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64+64//self.k, 3),
        )


    def forward(self, eeg, eye):

        e = self.f_fc1(eeg)
        e = self.f_relu1(e)
        e = self.f_bn1(e)
        e = self.f_drop1(e)
        o = self.ef_fc1(eye)
        o = self.ef_relu1(o)
        o = self.ef_bn1(o)
        o = self.ef_drop1(o)

        #e, o = self.i1(e, o)

        e = self.f_fc2(e)
        e = self.f_relu2(e)
        e = self.f_bn2(e)
        #e = self.f_drop2(e)

        o = self.ef_fc2(o)
        o = self.ef_relu2(o)
        o = self.ef_bn2(o)
        #o = self.ef_drop2(o)

        #e, o = self.i2(e, o)

        e = self.f_drop2(e)
        o = self.ef_drop2(o)

        e = self.f_fc3(e)
        e = self.f_relu3(e)
        e = self.f_bn3(e)

        o = self.ef_fc3(o)
        o = self.ef_relu3(o)
        o = self.ef_bn3(o)

        e, o = self.i3(e, o)

        classifier_eeg = self.c(e)
        classifier_eye = self.ec(o)
        classifier = self.c_all(torch.cat((e, o), dim=-1))

        return classifier, e, o, classifier_eeg, classifier_eye




if __name__ == '__main__':
    a = torch.tensor([2,9,5,7,3,5,91,3,6,4])
    _,c = a.topk(4, largest=False, sorted=False)
    print(c)
    # s = torch.randn(3, 1, 512)
    # n = torch.randn(2, 64, 128)
    # model = TSIMNet()
    # S, _, N = model(s)
    # print(S.shape, N.shape)