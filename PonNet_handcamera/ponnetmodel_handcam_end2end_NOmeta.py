from time import time
from os import path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import cv2
#import PIL as Image
from PIL import Image
import os
from os import path
#rom torchvision import transforms
import torchvision.transforms.functional as TF
import torch.utils.data as data
from tqdm import tqdm
import copy


import resnet_pre
import resnet_post
import ponnet_dataloader_handcamv2 as dataloader

class attention_branch(nn.Module):
    def __init__(self):
        super(attention_branch, self).__init__()

        self.resnet_pre = resnet_pre.resnet18(pretrained=True)

        self.extractor = nn.Sequential(nn.BatchNorm2d(256), nn.Conv2d(256, 1, 1))
        
        self.attentionbranch2attentionmechanism = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1),  nn.Sigmoid())

        self.perception_branch = nn.Sequential(resnet_post.resnet18(pretrained=True), nn.AvgPool2d(7), nn.Flatten(), nn.BatchNorm1d(512))
        #--------------------------------------
        self.attentionbranch2classifier = nn.Sequential(nn.AvgPool2d(14), nn.Linear(1,2), nn.Softmax(dim=3))


    def forward(self, x):

        #---Feature Extractor(ResNet-pre)---
        h = self.resnet_pre(x)
        identity = h.clone()

        #----buildingmask??--
        h = self.extractor(h)
        identity4 = h.clone()

        #---attention branch(2class outputいくほう)----
        att_outputs = self.attentionbranch2classifier(identity4)

        #---attention branch (attention mechanismに繋がるほうAttention Map)--
        h = self.attentionbranch2attentionmechanism(h)

        #------visualization-attentionmap---------
        self.visualization_att_map = h.clone()

        #---mechanism---
        h = h * identity
        h = h + identity

        #---Perception Branch(ResNet-post)---
        h = self.perception_branch(h)
        outputs = h.clone()

        return outputs, att_outputs, self.visualization_att_map

class parception_branch(nn.Module):
    def __init__(self):
        super(parception_branch, self).__init__()

        num_classes = 2
        self.linear2_1 = nn.Linear(512,256)
        self.linear2_2 = nn.Linear(512,256)
        self.linear2_3 = nn.Linear(512,256)

        self.rgb_balancing   = balancing(256,256)
        self.depth_balancing = balancing(256,256)
        self.hand_balancing = balancing(256,256)

        self.parception1 = nn.Sequential(nn.BatchNorm1d(256),nn.Linear(256,256),nn.ReLU(inplace=True),
                                        nn.Linear(256,16),nn.Linear(16,num_classes),nn.Softmax(dim=1))
    def forward(self, input_rgb, input_depth, input_handcam):

        h_rgb = self.linear2_1(input_rgb)
        h_depth = self.linear2_2(input_depth)
        h_hand = self.linear2_3(input_handcam)
        ##----------------Balancing------------
        e_rgb = self.rgb_balancing(h_rgb)
        e_depth = self.depth_balancing(h_depth)
        e_hand = self.hand_balancing(h_hand)
        exp_sum = torch.exp(e_rgb) + torch.exp(e_depth) + torch.exp(e_hand)
        alpha_rgb = torch.exp(e_rgb) / exp_sum
        alpha_depth = torch.exp(e_depth) / exp_sum
        alpha_hand = torch.exp(e_hand) / exp_sum

        balanced_att = (alpha_rgb * h_rgb) + (alpha_depth * h_depth) + (alpha_hand * h_hand)
        #重みのチェック（評価用）
        #print("alpha rgb === ",sum(alpha_rgb),min(alpha_rgb),max(alpha_rgb))
        #print("alpha depth === ",sum(alpha_depth),min(alpha_depth),max(alpha_depth))
        #print("alpha hand === ",sum(alpha_hand),min(alpha_hand),max(alpha_hand))
        y = self.parception1(balanced_att)
        
        return y

class balancing(nn.Module):
    def __init__(self, input_dim, output_dim):
        '''
        引数:
            input_dim: 入力次元
            output_dim: 出力次元
        パラメータ:
            W: 重み
            b: バイアス
            v: 
        '''
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(np.random.normal(size=(input_dim,output_dim))))
        self.b = nn.Parameter(torch.Tensor(np.zeros(output_dim)))
        self.v = nn.Parameter(torch.Tensor(np.zeros(output_dim)))
        self.tanh = nn.Tanh()
    def forward(self, x):
        return torch.matmul(self.tanh(torch.matmul(x, self.W) + self.b), torch.reshape(torch.t(self.v), (256,1)))

class ponnet(nn.Module):
    def __init__(self):
        super(ponnet, self).__init__()

        self.upsample = nn.Upsample([224,224], mode='bilinear', align_corners=False)
        
        self.rgb_attention_branch = attention_branch()
        self.depth_attention_branch = attention_branch()
        self.hand_attention_branch = attention_branch()

        self.perception_branch = parception_branch()


    def forward(self, input_rgb, input_depth, input_meta, input_handcam):

        input_rgb = self.upsample(input_rgb)
        input_depth = self.upsample(input_depth)

        rgb_out, rgb_att_out, rgb_visual = self.rgb_attention_branch(input_rgb)
        depth_out, depth_att_out, depth_visual = self.depth_attention_branch(input_depth)
        hand_out, hand_att_out, hand_visual = self.hand_attention_branch(input_handcam)

        parception_out = self.perception_branch(rgb_out, depth_out, hand_out)

        rgb_att_outputs = rgb_att_out.squeeze()
        depth_att_outputs = depth_att_out.squeeze()
        hand_att_out = hand_att_out.squeeze()

        #return rgb_att_outputs, depth_att_outputs, y1
        return [rgb_att_outputs, depth_att_outputs, hand_att_out], [parception_out], [rgb_visual, depth_visual, hand_visual]

        
if __name__ == "__main__":
    
    
    use_cuda = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Use device:', device)
    #---kakunin-data---
    batchsize = 2
    #rgb = torch.rand(batchsize,3,192,192,dtype=torch.float32)
    #depth = torch.rand(batchsize,3,192,192,dtype=torch.float32)

    rgb = torch.rand(batchsize,3,224,224,dtype=torch.float32)
    depth = torch.rand(batchsize,3,224,224,dtype=torch.float32)
    metafile = torch.zeros(batchsize, 4, dtype=torch.float32)

    if use_cuda:
        rgb = rgb.cuda()
        depth = depth.cuda()
        metafile = metafile.cuda()
    ponnet_dataset = dataloader.PonNetDataset()
    ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=batchsize)

        
    model = ponnet()
    #model.load_state_dict(torch.load("./logs/20200918_213632/50_model.ckpt"),strict=False)
    #model.load_state_dict(torch.load("./logs/20200925_151859/150_model.ckpt"),strict=False)
    if use_cuda:
        model.cuda()

    for i, (rgb_images, depth_images, meta_datas, y_labels, hand_images, rotation_labels) in tqdm(enumerate(ponnet_loader)):
    #print("mtafile.shape=",metafile.shape)
        hand_images = hand_images.to(device)
        rgb_images, depth_images, meta_datas = rgb_images.to(device), depth_images.to(device), meta_datas.to(device)
        attout, output_y1, visualization_attention = model(rgb_images, depth_images, meta_datas, hand_images)
        break
        if i == 10:
            break

    #print(np.argmax(output_y1))
    #outmax = output_y1[0].max(1)[1].reshape((batchsize,1))
    #print(outmax.shape)
    #print('-------------------------')
    #print("attout.shape",attout[0].shape)
    #print("attout",attout[0])
    #print("depth_attout.shape",attout[1].shape)
    #print("output_y1.shape",output_y1[0].shape)
    #print("output_y1",output_y1[0])

    