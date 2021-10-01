import math
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
from PIL import Image, ImageDraw
IS_TRAIN = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class FCNDecoder(nn.Module):
    def __init__(self, decode_layers, decode_channels=[], decode_last_stride=8):
        super(FCNDecoder, self).__init__()
        #decode_channels = [512, 512, 256]
        #decode_layers = ["pool5", "pool4", "pool3"]
        self._decode_channels = [512, 256]
        self._out_channel = 64
        self._decode_layers = decode_layers

        self._conv_layers = []
        for _ch in self._decode_channels:
            self._conv_layers.append(nn.Conv2d(_ch, self._out_channel, kernel_size=1, bias=False))

        self._conv_final = nn.Conv2d(self._out_channel, 2, kernel_size=1, bias=False)
        self._deconv = nn.ConvTranspose2d(self._out_channel, self._out_channel, kernel_size=4, stride=2, padding=1,
                                          bias=False)

        self._deconv_final = nn.ConvTranspose2d(self._out_channel, self._out_channel, kernel_size=16,
                                                stride=decode_last_stride,
                                                padding=4, bias=False)

    def forward(self, encode_data):
        ret = {}
        input_tensor = encode_data[0]
        #input_tensor = input_tensor.cuda()
        #print("input_tensor 的type = ",type(input_tensor))
        #print("-------0-------")
        global IS_TRAIN
        GPU_status = False
        if IS_TRAIN:
            #print(" GPU_status = True")
            GPU_status = True
        try:
            score = self._conv_layers[0](input_tensor)
            #print("is train------")
        except: 
            IS_TRAIN = True
            self._conv_layers[0] = self._conv_layers[0].to(device)
            #print("------0------")
            #print(input_tensor.is_cuda)
            score = self._conv_layers[0](input_tensor)
        IS_TRAIN = True
        if IS_TRAIN: input_tensor = input_tensor.to(device)
        for i in range(1,3):        
            #print("-------1-------")
            if GPU_status: score = score.to(device)
            #print(score.is_cuda)
            deconv = self._deconv(score)
            #print(deconv.is_cuda)
            #print("-------2-------")
            
            input_tensor = encode_data[i]
            #print(input_tensor.is_cuda)
            if GPU_status: self._conv_layers[i-1] = self._conv_layers[i-1].to(device)
            score = self._conv_layers[i-1](input_tensor)
            #print("-------3-------")
            if GPU_status: score = score.to(device)
            fused = torch.add(deconv, score)
            score = fused

        deconv_final = self._deconv_final(score)
        score_final = self._conv_final(deconv_final)

        ret['logits'] = score_final
        ret['deconv'] = deconv_final
        return ret