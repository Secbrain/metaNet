import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class HSU_Net(nn.Module):
    def __init__(self, img1_ch=1,img2_ch=5, output_ch=5, t=2):
        super(HSU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img1_ch, ch_out=64, t=t)
        self.RRCNN1_2 = RRCNN_block(ch_in=img2_ch, ch_out=64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=256, ch_out=256, t=t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t)

        # self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.fcl1 = nn.Linear(200704, 256)
        self.fcl10 = nn.Linear(256,128)
        self.fcl2 = nn.Linear(128, 64)
        self.fcl3 = nn.Linear(64, 32)
        self.fcl4 = nn.Linear(32, output_ch)

    def forward(self, x, y):
        # encoding path
        x1 = self.RRCNN1(x) #E-Res1-2
        y1 = self.RRCNN1_2(y) #E-Res1-1

        x2 = self.Maxpool(x1) #E-Mp2-2
        x2 = self.RRCNN2(x2) #E-Res2-2
        y2 = self.Maxpool(y1) #E-Mp2-1
        y2 = self.RRCNN2(y2)  #E-Res2-1

        x3 = self.Maxpool(x2) #E-Mp3-2
        #y3 = self.Maxpool(y2) #E-Mp3-1
        #print(y3.shape,x3.shape)
        x3 = torch.cat((y2,x3),dim=1)
        #print(x3.shape)
        x3 = self.RRCNN3(x3) #E-Res3-1

        d3 = self.Up3(x3) #D-UC2-1
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3) #D-Res2-1
        # print("d3:",d3.shape)

        d2 = self.Up2(d3) #D-UC1-1
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2) #D-Res1-1
        # print("d2:",d2.shape)
        # d1 = self.Conv_1x1(d2)

        d2 = d2.view(d2.size()[0], -1)
        #print(d2.shape)
        f1 = F.relu(self.fcl1(d2))
        # print(f1)
        f10 = F.relu(self.fcl10(f1))
        f2 = F.relu(self.fcl2(f10))
        # print(f2)
        f3 = F.relu(self.fcl3(f2))
        f4 = F.relu(self.fcl4(f3))
        # print("f1:",f1.shape)
        return f4

