from os import X_OK
import torch
from torch import nn, optim
from torch.nn import functional as F

class LnReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LnReLU, self).__init__()
        layers=[
            nn.Linear(in_ch, out_ch, bias=True),
            nn.ReLU()
        ]
        self.ln = nn.Sequential(*layers)
    def forward(self, x):
        x = self.ln(x)
        return x


class LnSELU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LnSELU, self).__init__()
        layers=[
            nn.Linear(in_ch, out_ch, bias=True),
            nn.SELU()
        ]
        self.ln = nn.Sequential(*layers)
    def forward(self, x):
        x = self.ln(x)
        return x
        
'''
class Build(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,32)
        self.ln3 = LnReLU(16,64)

        self.lnb1_1= LnReLU(64,128)
        self.lnb1_2 = LnReLU(128,128)

        self.lnb2_1 = LnSELU(64, 128)
        self.lnb2_2 = LnSELU(128,128)

        self.lnb3 = nn.Linear(128, 1)

    def forward(self,x):
        l1 = self.ln1(x)
        # l2 = self.ln2(l1)
        l3 = self.ln3(l1)

        hi_1 = self.lnb1_1(l3)
        hi_2 = self.lnb1_2(hi_1)
        hi_3 = self.lnb3(hi_2)

        fnls_1 = self.lnb2_1(l3)
        fnls_2 = self.lnb2_2(fnls_1)
        fnls_3 = self.lnb3(fnls_2)

        return hi_3, fnls_3

class Build_SELU5(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(Build_SELU5, self).__init__()
        self.conv1 = LnSELU(in_ch, 16)
        self.conv2 = LnSELU(16,64)
        self.conv3 = LnSELU(64,128)
        self.conv4 = LnSELU(128, 128)
        self.conv5 = nn.Linear(128, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv5(c4)
        return c5, c6

class Build2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build2, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,32)
        self.ln3 = LnReLU(32,64)

        self.lnb1_1= LnSELU(64,128)

        self.lnb2_1 = LnReLU(64, 128)

        self.lnb3 = nn.Linear(128, 1)

    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)
        l3 = self.ln3(l2)

        hi_1 = self.lnb1_1(l3)
        hi_3 = self.lnb3(hi_1)

        fnls_1 = self.lnb2_1(l3)
        fnls_3 = self.lnb3(fnls_1)

        return hi_3, fnls_3


class Build2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build2, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,32)
        self.ln3 = LnReLU(16,64)

        self.lnb1_1= LnReLU(64,128)

        self.lnb2_1 = LnSELU(64, 128)

        self.lnb3 = nn.Linear(128, 1)

    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)
        l3 = self.ln3(l1)

        hi_1 = self.lnb1_1(l3)
        hi_3 = self.lnb3(hi_1)

        fnls_1 = self.lnb2_1(l3)
        fnls_3 = self.lnb3(fnls_1)

        return hi_3, fnls_3



class Build3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build3, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.ln3 = LnReLU(64,128)


        # self.lnb1_1= LnReLU(128,128)

        self.lnb2_1 = LnSELU(128, 64)

        self.lnb2_2 = LnReLU(64,64)
        self.lnb3 = nn.Linear(64, 1, bias = False)

        # self.lnb4 = nn.Linear(128,1, bias=False)

    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)
        l3 = self.ln3(l2)

        hi_1 = self.lnb2_1(l3)
        hi_2 = self.lnb3(hi_1)

        fnls_1 = self.lnb2_2(hi_1)
        # print(fnls_1.shape)
        fnls_2 = self.lnb3(fnls_1)

        return hi_2, fnls_2


class Build4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build4, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.ln3 = LnReLU(64,128)


        self.lnb1_1= LnReLU(128,128)

        self.lnb2_1 = LnSELU(128, 64)

        self.lnb2_2 = LnReLU(64,64)
        self.lnb3 = nn.Linear(64, 1, bias = False)

        # self.lnb4 = nn.Linear(128,1, bias=False)

    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)
        l3 = self.ln3(l2)

        hi_1 = self.lnb1_1(l3)
        hi_2 = self.lnb2_1(hi_1)
        hi_3= self.lnb3(hi_2)

        fnls_1 = self.lnb2_2(hi_2)
        # print(fnls_1.shape)
        fnls_2 = self.lnb3(fnls_1)

        return hi_3, fnls_2


class Build5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build5, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.ln3 = LnReLU(64,128)


        self.lnb1_1= LnReLU(128,128)

        self.lnb2_1 = LnSELU(128, 64)

        self.lnb2_2 = LnReLU(64,64)
        self.lnb3 = nn.Linear(64, 1, bias = False)
        
        self.dp = nn.Dropout(p=0.3)

    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.dp(self.ln2(l1))
        l3 = self.dp(self.ln3(l2))

        hi_1 = self.dp(self.lnb1_1(l3))
        hi_2 = self.dp(self.lnb2_1(hi_1))
        hi_3= self.lnb3(hi_2)

        fnls_1 = self.dp(self.lnb2_2(hi_2))
        fnls_2 = self.lnb3(fnls_1)

        return hi_3, fnls_2


class Build6(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build6, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.ln3 = LnReLU(64,128)


        self.lnb1_1= LnReLU(128,128)
        self.lnb2_1 = LnSELU(128, 64)

        self.lnb2_2 = LnReLU(64,64)
        self.lnb3 = nn.Linear(64, 1, bias = False)


    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)
        l3 = self.ln3(l2)

        hi_1 = self.lnb1_1(l3)
        hi_2 = self.lnb2_1(hi_1)
        hi_2_2 = self.lnb2_2(hi_2)
        hi_3= self.lnb3(hi_2_2)

        fnls_1 = self.lnb2_2(hi_2)
        fnls_2 = self.lnb3(fnls_1)

        return hi_3, fnls_2



class Build7(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build7, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.lnb1_1= LnReLU(64,64)

        self.lnb2_2 = LnReLU(64,64)
        self.lnb3 = nn.Linear(64, 1, bias = False)


    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)

        hi_1 = self.lnb1_1(l2)
        hi_2 = self.lnb3(hi_1)

        fnls_1 = self.lnb2_2(hi_1)
        fnls_2 = self.lnb3(fnls_1)

        return hi_2, fnls_2




class Build8(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build8, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.lnb1_1= LnReLU(64,64)

        self.lnb2_2 = LnReLU(64,64)
        self.lnb3 = nn.Linear(64, 1, bias = False)


    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)

        hi_1 = self.lnb1_1(l2)
        hi_2 = self.lnb2_2(hi_1)
        hi_3 = self.lnb2_2(hi_2)
        hi_4 = self.lnb3(hi_3)

        fnls_1 = self.lnb2_2(hi_3)
        fnls_2 = self.lnb2_2(fnls_1)
        fnls_3= self.lnb3(fnls_2)

        return hi_4, fnls_3


class Build9(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build9, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.ln3 = LnReLU(64,128)


        self.lnb2_2 = LnSELU(128,64)
        self.lnb2_1 = LnReLU(64,64)
        self.lnb3 = nn.Linear(64, 1, bias = False)

        # self.lnb4 = nn.Linear(128,1, bias=False)

    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)
        l3 = self.ln3(l2)

        hi_1 = self.lnb2_2(l3)
        hi_2 = self.lnb3(hi_1)


        fnls_1 = self.lnb2_1(hi_1)
        fnls_2 = self.lnb3(fnls_1)


        return hi_2, fnls_2


class Build10(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build10, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.ln3 = LnReLU(64,64)


        self.lnb2_2 = LnSELU(64,64)
        self.lnb2_1 = LnReLU(64,64)
        self.lnb3 = nn.Linear(64, 1, bias = False)

        # self.lnb4 = nn.Linear(128,1, bias=False)

    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)
        l3 = self.ln3(l2)

        hi_1 = self.lnb2_2(l3)
        hi_2 = self.lnb3(hi_1)


        fnls_1 = self.lnb2_1(hi_1)
        fnls_2 = self.lnb3(fnls_1)


        return hi_2, fnls_2



class Build11(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build11, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.ln3 = LnReLU(64,64)


        self.lnb2_2 = LnSELU(64,32)
        self.lnb2_1 = LnReLU(32,32)
        self.lnb3 = nn.Linear(32, 1, bias = False)

        # self.lnb4 = nn.Linear(128,1, bias=False)

    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)
        l3 = self.ln3(l2)
        l4 = self.ln3(l3)

        hi_1 = self.lnb2_2(l4)
        hi_2 = self.lnb3(hi_1)


        fnls_1 = self.lnb2_1(hi_1)
        fnls_2 = self.lnb3(fnls_1)
        1

        return hi_2, fnls_2

class Build_12(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(Build_12, self).__init__()
        self.conv1 = LnSELU(in_ch, 16)
        self.conv2 = LnSELU(16,64)
        self.conv3 = LnSELU(64,128)
        self.conv4 = LnSELU(128, 128)
        self.conv5 = nn.Linear(128, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv5(c4)
        return c5, c6



class Build13(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build13, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,64)
        self.ln3 = LnReLU(64,128)


        self.lnb2_2 = LnSELU(128,128)
        self.lnb2_3 = LnSELU(128,64)
        # self.lnb2_1 = LnReLU(64,64)


        self.lnb2_1 = LnReLU(128,64)
        self.lnb3 = nn.Linear(64, 1, bias = False)

        # self.lnb4 = nn.Linear(128,1, bias=False)

    def forward(self,x):
        l1 = self.ln1(x)
        l2 = self.ln2(l1)
        l3 = self.ln3(l2)

        hi_1 = self.lnb2_2(l3)
        hi_2 = self.lnb2_3(hi_1)
        hi_3 = self.lnb3(hi_2)


        fnls_1 = self.lnb2_1(l3)
        fnls_2 = fnls_1+hi_2

        fnls_3 = self.lnb3(fnls_2)


        return hi_3, fnls_3



class Build15(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build15, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,32)
        self.ln3 = LnReLU(32,32)

        self.lnb2_2 = LnSELU(32,16)
        self.lnb2_3 = LnSELU(16,16)
        self.lnb2_4 = nn.Linear(16,1)

        self.lnb2_1 = LnReLU(32,16)
        self.lnb3 = nn.Linear(16, 1, bias = False)



    def forward(self,x):
        l1 = self.ln1(x)        #1,16
        l2 = self.ln2(l1)       #16,32
        l3 = self.ln3(l2)    #32,32

        hi_1 = self.lnb2_2(l3)  #32,16
        hi_2 = self.lnb2_3(hi_1)#16,16
        hi_3 = self.lnb3(hi_2)  #16,1

        fnls_1 = self.lnb2_1(l3)                    #32,16
        # fnls_2 = fnls_1+hi_2
        fnls_2 = torch.cat([fnls_1, hi_2], dim=1)   #16+16 ,32
        fnls_3 = self.lnb2_2(fnls_2)                #32,16
        fnls_4 = self.lnb3(fnls_3)                  #16,1

        return hi_3, fnls_4


class Build16(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build16, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,32)
        self.ln3 = LnReLU(32,64)

        self.lnb2_2 = LnSELU(64,32)
        self.lnb2_3 = LnSELU(32,32)
        self.lnb2_4 = LnSELU(32,16)
        self.lnb2_5 = nn.Linear(16,1)
        
        self.lnb3_1 = LnReLU(64,32)
        self.lnb3_2 = LnReLU(48,16)
        self.lnb3 = nn.Linear(16, 1, bias = False)



    def forward(self,x):
        l1 = self.ln1(x)        #1,16
        l2 = self.ln2(l1)       #16,64
        l3 = self.ln3(l2)    #32,64

        hi_1 = self.lnb2_2(l3)  #64,32
        hi_2 = self.lnb2_3(hi_1)#32,32
        hi_3 = self.lnb2_4(hi_2) #32,16
        hi_4 = self.lnb3(hi_3)  #32,1

        fnls_1 = self.lnb3_1(l3)                    #64,32
        fnls_2 = torch.cat([fnls_1, hi_3], dim=1)   #32+32 ,64
        fnls_3 = self.lnb3_2(fnls_2)                #64,32
        fnls_4 = self.lnb3(fnls_3)                  #32,1

        return hi_4, fnls_4
'''

class Build14(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Build14, self).__init__()
        
        self.ln1 = LnReLU(in_ch, 16)
        self.ln2 = LnReLU(16,32)
        self.ln3 = LnReLU(32,64)

        self.lnb2_2 = LnSELU(64,32)
        self.lnb2_3 = LnSELU(32,32)
        self.lnb2_4 = nn.Linear(32,1)

        self.lnb2_1 = LnReLU(64,32)
        self.lnb3 = nn.Linear(32, 1, bias = False)



    def forward(self,x):
        l1 = self.ln1(x)        #7,16
        l2 = self.ln2(l1)       #16,64
        l3 = self.ln3(l2)    #32,64

        hi_1 = self.lnb2_2(l3)  #64,32
        hi_2 = self.lnb2_3(hi_1)#32,32
        hi_3 = self.lnb3(hi_2)  #32,1

        fnls_1 = self.lnb2_1(l3)                    #64,32
        fnls_2 = torch.cat([fnls_1, hi_2], dim=1)   #32+32 ,64
        fnls_3 = self.lnb2_2(fnls_2)                #64,32
        fnls_4 = self.lnb3(fnls_3)                  #32,1

        return hi_3, fnls_4
