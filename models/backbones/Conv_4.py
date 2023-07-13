import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    
    def __init__(self,input_channel,output_channel):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_channel))

    def forward(self,inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self,num_channel=64):
        super().__init__()
        
        self.layers1 = nn.Sequential(
            ConvBlock(3,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layers2 = nn.Sequential(
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layers3 = nn.Sequential(
            ConvBlock(num_channel, num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layers4 = nn.Sequential(
            ConvBlock(num_channel, num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )


    def forward(self,inp):
        l1 = self.layers1(inp)
        l2 = self.layers2(l1)
        l3 = self.layers3(l2)
        l4 = self.layers4(l3)

        return l1,l2,l3,l4