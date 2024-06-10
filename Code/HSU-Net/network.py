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

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        #print("-",x[0])
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print("u-",x)
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t)#,
            # Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3,),stride=(1,),padding=(1,),bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        #self.Conv3 = conv_block(ch_in=128,ch_out=256)
        #self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=128,ch_out=256)

        self.Up5 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        #self.Up4 = up_conv(ch_in=512,ch_out=256)
        #self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        #self.Up3 = up_conv(ch_in=256,ch_out=128)
        #self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

'''
class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=5,t=2):
        super(R2AttU_Net,self).__init__()

        self.channel1 = 5
        self.channel2 = 10
        # 输入数据25*320*1
        self.conv1 = nn.Conv2d(1, self.channel1, 3)  # 输入通道数为1，输出通道数为5
        self.conv2 = nn.Conv2d(self.channel1, self.channel2, 3)  # 输入通道数为5，输出通道数为10
        self.fc1 = nn.Linear(144 * self.channel2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_ch)

    def forward(self,x):
        # encoding path
        #print("x:",x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # 输入x -> conv2 -> relu -> 2x2窗口的最大池化
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(x.size()[0], -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=1,output_ch=5,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        #self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.fcl1 = nn.Linear(200704,output_ch)
        # self.fcl2 = nn.Linear(128,64)
        # self.fcl3 = nn.Linear(64, 32 )
        # self.fcl4 = nn.Linear(32 ,output_ch)

    def forward(self,x):
        # encoding path
        #print("x:",x.shape)
        x1 = self.RRCNN1(x)
        #print("x1:", x1.shape)
        ''''''
        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        #print("x2:", x2.shape)
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        #print("x3:", x3.shape)
        

        d3 = self.Up3(x3)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)
        #print("d3:",d3.shape)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)
        #print("d2:",d2.shape)
        #d1 = self.Conv_1x1(d2)

        d2 = d2.view(d2.size()[0], -1)
        f1 = F.relu(self.fcl1(d2))
        # print(f1)
        f2 = F.relu(self.fcl2(f1))
        #print(f2)
        f3 = F.relu(self.fcl3(f2))
        f4 = F.relu(self.fcl4(f3))
        # print("f1:",f1.shape)
        return f4


# class HSU_Net(nn.Module):
#     def __init__(self, img1_ch=1,img2_ch=5, output_ch=5, t=2):
#         super(HSU_Net, self).__init__()

#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Upsample = nn.Upsample(scale_factor=2)

#         self.RRCNN1 = RRCNN_block(ch_in=img1_ch, ch_out=16, t=t) #64
#         self.RRCNN1_2 = RRCNN_block(ch_in=img2_ch, ch_out=16, t=t) #64
#         self.RRCNN2 = RRCNN_block(ch_in=16, ch_out=32, t=t) #64 128

#         self.RRCNN3 = RRCNN_block(ch_in=64, ch_out=64, t=t) #256 256



#         self.Up3 = up_conv(ch_in=64, ch_out=32) #256 128
#         self.Att3 = Attention_block(F_g=32, F_l=32, F_int=16) #128 128 64
#         self.Up_RRCNN3 = RRCNN_block(ch_in=64, ch_out=32, t=t) #256 128

#         self.Up2 = up_conv(ch_in=32, ch_out=16) #128 64
#         self.Att2 = Attention_block(F_g=16, F_l=16, F_int=8) #64 64 32
#         self.Up_RRCNN2 = RRCNN_block(ch_in=32, ch_out=16, t=t) #128 64

#         # self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
#         self.fcl1 = nn.Linear(200704, 1024)
#         self.fcl11 = nn.Linear(512, 64)
#         # self.fcl10 = nn.Linear(256,128)
#         # self.fcl2 = nn.Linear(128, 64)
#         # self.fcl3 = nn.Linear(64, 32)
#         # self.fcl4 = nn.Linear(32, 16)
#         # self.fcl5 = nn.Linear(16, 8)
#         # self.fcl6 = nn.Linear(8, 4)
#         # self.fcl7 = nn.Linear(4, output_ch)
#         self.fcl8 = nn.Linear(1024, 64)
#         self.fc20 = nn.Linear(64, output_ch)
#         self.fcl9 = nn.Linear(50176, output_ch) #200704

#     def forward(self, x, y):
#         # encoding path
#         x1 = self.RRCNN1(x) #E-Res1-2
#         y1 = self.RRCNN1_2(y) #E-Res1-1

#         x2 = self.Maxpool(x1) #E-Mp2-2
#         x2 = self.RRCNN2(x2) #E-Res2-2
#         y2 = self.Maxpool(y1) #E-Mp2-1
#         y2 = self.RRCNN2(y2)  #E-Res2-1

#         x3 = self.Maxpool(x2) #E-Mp3-2
#         #y3 = self.Maxpool(y2) #E-Mp3-1
#         #print(y3.shape,x3.shape)
#         x3 = torch.cat((y2,x3),dim=1)
#         #print(x3.shape)
#         x3 = self.RRCNN3(x3) #E-Res3-1


#         d3 = self.Up3(x3) #D-UC2-1
#         x2 = self.Att3(g=d3, x=x2)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_RRCNN3(d3) #D-Res2-1
#         # print("d3:",d3.shape)

#         d2 = self.Up2(d3) #D-UC1-1
#         x1 = self.Att2(g=d2, x=x1)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_RRCNN2(d2) #D-Res1-1
#         # print("d2:",d2.shape)
#         # d1 = self.Conv_1x1(d2)

#         d2 = d2.view(d2.size()[0], -1)
#         #print(d2.shape)
#         # f1 = self.fcl9(d2)
#         # print(f1)
#         # f1 = self.fcl1(d2)
#         # f1 = F.relu(self.fcl1(d2))
#         # f10 = F.relu(self.fcl10(f11))
#         # f2 = F.relu(self.fcl2(f10))
#         # # print(f2)
#         # f3 = F.relu(self.fcl3(f2))
#         # f4 = F.relu(self.fcl4(f3))
#         # f5 = F.relu(self.fcl5(f4))
#         # f6 = F.relu(self.fcl6(f5))
#         # f7 = F.relu(self.fcl7(f6))
#         # f2 = self.fcl8(f1)
#         # f7 = F.relu(self.fcl8(f11))
#         # f3 = self.fc20(f2)
#         # print(d2.shape)
#         f3 = self.fcl9(d2)

#         # print(f7.shape)
#         # print("f1:",f1.shape)
#         return f3

class HSU_Net(nn.Module):
    def __init__(self, img1_ch=1,img2_ch=5, output_ch=5, t=2):
        super(HSU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool_1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img2_ch, ch_out=64, t=t) #64
        self.RRCNN1_2 = RRCNN_block(ch_in=img1_ch, ch_out=1, t=t) #64
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t) #64 128
        self.RRCNN2_2 = RRCNN_block(ch_in=1, ch_out=1, t=t) #64 128

        self.RRCNN3 = RRCNN_block(ch_in=129, ch_out=256, t=t) #256 256

        self.Up3 = up_conv(ch_in=256, ch_out=128) #256 128
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64) #128 128 64
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=t) #256 128

        self.Up2 = up_conv(ch_in=128, ch_out=64) #128 64
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32) #64 64 32
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=t) #128 64

        # self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.fcl1 = nn.Linear(50176, 1024)
        # self.fcl11 = nn.Linear(512, 64)
        # self.fcl10 = nn.Linear(256,128)
        # self.fcl2 = nn.Linear(128, 64)
        # self.fcl3 = nn.Linear(64, 32)
        # self.fcl4 = nn.Linear(32, 16)
        # self.fcl5 = nn.Linear(16, 8)
        # self.fcl6 = nn.Linear(8, 4)
        # self.fcl7 = nn.Linear(4, output_ch)
        # self.fcl8 = nn.Linear(1024, 64)
        self.fc20 = nn.Linear(1024, output_ch)
        self.fcl9 = nn.Linear(200704, output_ch) #200704

        self.fc21 = nn.Linear(1024, output_ch) #200704

    def forward(self, x, y):
        # encoding path
        x1 = self.RRCNN1(y) #E-Res1-2
        # print(x.shape)
        y1 = self.RRCNN1_2(x) #E-Res1-1

        x2 = self.Maxpool(x1) #E-Mp2-2
        x2 = self.RRCNN2(x2) #E-Res2-2
        y2 = self.Maxpool_1(y1) #E-Mp2-1
        y2 = self.RRCNN2_2(y2)  #E-Res2-1

        x3 = self.Maxpool(x2) #E-Mp3-2
        y3 = self.Maxpool_1(y2) #E-Mp3-1
        #print(y3.shape,x3.shape)
        # print("y2:{}".format(y2.shape))
        # print("y3:{}".format(y3.shape))
        # print("x3:{}".format(x3.shape))
        x3 = torch.cat((x3,y3),dim=1)
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
        # print(d2.shape)
        # f1 = self.fcl9(d2)
        # print(f1)
        f1 = self.fcl1(d2)
        # f1 = F.relu(self.fcl1(d2))
        # f10 = F.relu(self.fcl10(f11))
        # f2 = F.relu(self.fcl2(f10))
        # # print(f2)
        # f3 = F.relu(self.fcl3(f2))
        # f4 = F.relu(self.fcl4(f3))
        # f5 = F.relu(self.fcl5(f4))
        # f6 = F.relu(self.fcl6(f5))
        # f7 = F.relu(self.fcl7(f6))
        # f2 = self.fcl8(f1)
        # f7 = F.relu(self.fcl8(f11))
        f3 = self.fc20(f1)
        # print(d2.shape)
        # f3 = self.fcl9(d2)

        # f3 = self.fc21(f1)


        # print(f7.shape)
        # print("f3:",f3.shape)
        return f3