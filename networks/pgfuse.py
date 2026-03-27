from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from .resnet import *
from .mobilenet import *
from .layers import Conv3x3, ConvBlock, upsample, Cube2Equirec, Concat, BiProj, CEELayer

from collections import OrderedDict
from .PSAutils.equisamplingpoint import genSamplingPattern
from .PSAutils.DBAT import *
from einops.layers.torch import Rearrange
from torch import einsum

class FastLeFF(nn.Module):
    
    def __init__(self, dim=256, hidden_dim=512, act_layer=nn.GELU,drop = 0.):
        super().__init__()

        #from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x, hh, ww):
        # bs x hw x c
        bs, hw, c = x.size()

        x = self.linear1(x)

        # spatial restore
        x = x.view(bs, hh, ww, self.hidden_dim)
        x = x.permute(0, 3, 1, 2)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = x.view(bs, self.hidden_dim, hh*ww)
        x = x.permute(0, 2, 1)#Rearrange(x, ' b c h w -> b (h w) c', h = hh, w = ww)

        x = self.linear2(x)

        return x

class PGFuse(nn.Module):
    def __init__(self, num_layers, equi_h, equi_w, pretrained=True, max_depth=10.0,
                 fusion_type="cee", se_in_fusion=True):
        super(PGFuse, self).__init__()

        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.cube_h = equi_h//2

        self.fusion_type = fusion_type
        self.se_in_fusion = se_in_fusion
        #self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        # encoder
        encoder = {2: mobilenet_v2,
                   18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,
                   152: resnet152}

        if num_layers not in encoder:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        self.cube_encoderA = encoder[num_layers](pretrained)
        self.cube_encoderB = self.cube_encoderA

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        if num_layers < 18:
            self.num_ch_enc = np.array([16, 24, 32, 96, 320])

        # decoder
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.equi_dec_convs = OrderedDict()
        self.c2e = {}

        Fusion_dict = {"cat": Concat,
                       "biproj": BiProj,
                       "cee": CEELayer}
        FusionLayer = Fusion_dict[self.fusion_type]


        self.c2e["5"] = Cube2Equirec(self.cube_h // 32, self.equi_h // 32, self.equi_w // 32)

        self.equi_dec_convs["fusion_5"] = FusionLayer(self.num_ch_enc[4], 16,  SE=self.se_in_fusion)
        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])

        self.c2e["4"] = Cube2Equirec(self.cube_h // 16, self.equi_h // 16, self.equi_w // 16)
        self.equi_dec_convs["fusion_4"] = FusionLayer(self.num_ch_enc[3], 32, SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_4"] = ConvBlock(256, self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        self.c2e["3"] = Cube2Equirec(self.cube_h // 8, self.equi_h // 8, self.equi_w // 8)
        self.equi_dec_convs["fusion_3"] = FusionLayer(self.num_ch_enc[2], 64, SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3]+256, self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        self.c2e["2"] = Cube2Equirec(self.cube_h // 4, self.equi_h // 4, self.equi_w // 4)
        self.equi_dec_convs["fusion_2"] = FusionLayer(self.num_ch_enc[1], 128, SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + 256, self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        self.c2e["1"] = Cube2Equirec(self.cube_h // 2, self.equi_h // 2, self.equi_w // 2)
        self.equi_dec_convs["fusion_1"] = FusionLayer(self.num_ch_enc[0], 256, SE=self.se_in_fusion)
        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + 256, self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])

        self.equi_dec_convs["deconv_0"] = Conv3x3(self.num_ch_dec[0], self.num_ch_dec[0])
        self.relu = nn.ReLU(inplace=True)
        
        self.ref_point32x64 = genSamplingPattern(32, 64, 3, 3).cuda()#改大小
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.norm4 = nn.LayerNorm(256)
        self.dattn1 = DeformableHeadAttention(8, 256, k=9, last_feat_height=32, last_feat_width=64, scales=4, dropout=0.0, need_attn=False)
        self.dattn2 = DeformableHeadAttention(8, 256, k=9, last_feat_height=32, last_feat_width=64, scales=1, dropout=0.0, need_attn=False)
        self.mlp1 = FastLeFF()
        self.mlp2 = FastLeFF()

        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
        self.projectors = nn.ModuleList(list(self.c2e.values()))

        self.sigmoid = nn.Sigmoid()

        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)


    def forward(self, input_cube_imageA, input_cube_imageB):
        #print(input_cube_imageB.shape[0])
        

        # cube image encoding
        cube_inputsA = torch.cat(torch.split(input_cube_imageA, self.cube_h, dim=-1), dim=0)

        if self.num_layers < 18:
            cube_enc_featA0, cube_enc_featA1, cube_enc_featA2, cube_enc_featA3, cube_enc_featA4 \
                = self.cube_encoderA(cube_inputsA)
        else:

            x = self.cube_encoderA.conv1(cube_inputsA)
            x = self.cube_encoderA.relu(self.cube_encoderA.bn1(x))
            cube_enc_featA0 = x

            x = self.cube_encoderA.maxpool(x)

            cube_enc_featA1 = self.cube_encoderA.layer1(x)
            cube_enc_featA2 = self.cube_encoderA.layer2(cube_enc_featA1)
            cube_enc_featA3 = self.cube_encoderA.layer3(cube_enc_featA2)
            cube_enc_featA4 = self.cube_encoderA.layer4(cube_enc_featA3)


        # cube image encoding
        cube_inputsB = torch.cat(torch.split(input_cube_imageB, self.cube_h, dim=-1), dim=0)

        if self.num_layers < 18:
            cube_enc_featB0, cube_enc_featB1, cube_enc_featB2, cube_enc_featB3, cube_enc_featB4 \
                = self.cube_encoder(cube_inputsB)
        else:

            x = self.cube_encoderB.conv1(cube_inputsB)
            x = self.cube_encoderB.relu(self.cube_encoderB.bn1(x))
            cube_enc_featB0 = x

            x = self.cube_encoderB.maxpool(x)

            cube_enc_featB1 = self.cube_encoderB.layer1(x)
            cube_enc_featB2 = self.cube_encoderB.layer2(cube_enc_featB1)
            cube_enc_featB3 = self.cube_encoderB.layer3(cube_enc_featB2)
            cube_enc_featB4 = self.cube_encoderB.layer4(cube_enc_featB3)


        # euqi image decoding fused with cubemap features
        outputs = {}
        #fuse4
        cube_enc_featB4 = torch.cat(torch.split(cube_enc_featB4, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featB4 = self.c2e["5"](cube_enc_featB4)
        cube_enc_featA4 = torch.cat(torch.split(cube_enc_featA4, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featA4 = self.c2e["5"](cube_enc_featA4)
        _,_,_,W = c2e_enc_featA4.shape
        shift = W//8
        c2e_enc_featA4 = torch.cat((c2e_enc_featA4[:, :, :, shift:], c2e_enc_featA4[:, :, :, :shift]), dim=3)       
        fused_feat4 = self.equi_dec_convs["fusion_5"](c2e_enc_featB4, c2e_enc_featA4)
        #fuse3
        cube_enc_featB3 = torch.cat(torch.split(cube_enc_featB3, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featB3 = self.c2e["4"](cube_enc_featB3)
        cube_enc_featA3 = torch.cat(torch.split(cube_enc_featA3, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featA3 = self.c2e["4"](cube_enc_featA3)
        _,_,_,W = c2e_enc_featA3.shape
        shift = W//8
        c2e_enc_featA3 = torch.cat((c2e_enc_featA3[:, :, :, shift:], c2e_enc_featA3[:, :, :, :shift]), dim=3)                        
        fused_feat3 = self.equi_dec_convs["fusion_4"](c2e_enc_featB3, c2e_enc_featA3)
        #fuse2
        cube_enc_featB2 = torch.cat(torch.split(cube_enc_featB2, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featB2 = self.c2e["3"](cube_enc_featB2)
        cube_enc_featA2 = torch.cat(torch.split(cube_enc_featA2, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featA2 = self.c2e["3"](cube_enc_featA2)
        _,_,_,W = c2e_enc_featA2.shape
        shift = W//8
        c2e_enc_featA2 = torch.cat((c2e_enc_featA2[:, :, :, shift:], c2e_enc_featA2[:, :, :, :shift]), dim=3)                        
        fused_feat2 = self.equi_dec_convs["fusion_3"](c2e_enc_featB2, c2e_enc_featA2)
        #fuse1
        cube_enc_featB1 = torch.cat(torch.split(cube_enc_featB1, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featB1 = self.c2e["2"](cube_enc_featB1)
        cube_enc_featA1 = torch.cat(torch.split(cube_enc_featA1, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featA1 = self.c2e["2"](cube_enc_featA1)
        _,_,_,W = c2e_enc_featA1.shape
        shift = W//8
        c2e_enc_featA1 = torch.cat((c2e_enc_featA1[:, :, :, shift:], c2e_enc_featA1[:, :, :, :shift]), dim=3)                        
        fused_feat1 = self.equi_dec_convs["fusion_2"](c2e_enc_featB1, c2e_enc_featA1)
        #fuse0
        cube_enc_featB0 = torch.cat(torch.split(cube_enc_featB0, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featB0 = self.c2e["1"](cube_enc_featB0)
        cube_enc_featA0 = torch.cat(torch.split(cube_enc_featA0, input_cube_imageB.shape[0], dim=0), dim=-1)
        c2e_enc_featA0 = self.c2e["1"](cube_enc_featA0)
        _,_,_,W = c2e_enc_featA0.shape
        shift = W//8
        c2e_enc_featA0 = torch.cat((c2e_enc_featA0[:, :, :, shift:], c2e_enc_featA0[:, :, :, :shift]), dim=3)                        
        fused_feat0 = self.equi_dec_convs["fusion_1"](c2e_enc_featB0, c2e_enc_featA0)
        
        b, c, refh, refw = fused_feat2.shape
        #print(fused_feat4.shape)
        #print(fused_feat3.shape)
        #print(fused_feat2.shape)
        #print(fused_feat1.shape)
        #print(fused_feat0.shape)
        fuse_list = []
        fuse_list.append(fused_feat4.permute(0, 2, 3, 1).contiguous())
        fuse_list.append(fused_feat3.permute(0, 2, 3, 1).contiguous())
        fuse_list.append(fused_feat2.permute(0, 2, 3, 1).contiguous())
        fuse_list.append(fused_feat1.permute(0, 2, 3, 1).contiguous())
        #fuse_list.append(fused_feat0.permute(0, 2, 3, 1).contiguous())
        q = fuse_list[-3]
        _, h, w, c = q.shape
        #res1 = q.view(b, c, h, w)
        #res1 = q.permute(0, 3, 1, 2).contiguous()
        q = q.view(b, h*w, c)
        tmp = q
        q = self.norm1(q) 
        q = q.view(b, h, w, c)
        x = self.dattn1(q, fuse_list, self.ref_point32x64.repeat(b, 1, 1, 1, 1))
        x = x.view(b, h*w, c)
        x = x + self.mlp1(self.norm2(x), h, w)
        q = self.norm3(x) 
        q = q.view(b, h, w, c)
        x = self.dattn2(q, q.unsqueeze(0), self.ref_point32x64.repeat(b, 1, 1, 1, 1))
        x = x.view(b, h*w, c)
        x = x + self.mlp2(self.norm4(x), h, w)
        
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(b, c, h, w)
        equi_x = x
        #print(equi_x.shape)
        #x = x + res1
        
        equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))
        
        equi_x = torch.cat([equi_x, fused_feat2], 1)
        equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))

        
        equi_x = torch.cat([equi_x, fused_feat1], 1)
        equi_x = self.equi_dec_convs["deconv_2"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_2"](equi_x))

        
        equi_x = torch.cat([equi_x, fused_feat0], 1)
        equi_x = self.equi_dec_convs["deconv_1"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_1"](equi_x))

        equi_x = self.equi_dec_convs["deconv_0"](equi_x)

        equi_depth = self.equi_dec_convs["depthconv_0"](equi_x)
        outputs["pred_depth"] = equi_depth

        return outputs
