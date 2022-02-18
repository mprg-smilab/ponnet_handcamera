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
import argparse

#import ponnetmodel_single_parseption_branch_balancing as ponnet
import ponnetmodel_typeA as ponnet
import ponnetmodel_baseline_NOresblock as ponnet
import ponnet_dataloader as dataloader

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='アテンションマップの可視化を行います')
    parser.add_argument('-b','--batchsize', default='10',type=int)
    parser.add_argument('-f','--path')
    args = parser.parse_args()

    batchsize = args.batchsize

    use_cuda = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Use device:', device)

    ponnet_dataset = dataloader.PonNetDataset(load_mode="valid")
    ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=batchsize)

    model = ponnet.ponnet()

    #model.load_state_dict(torch.load("./logs/20200918_213632/50_model.ckpt"),strict=False)
    model.load_state_dict(torch.load(args.path),strict=False)
    if use_cuda:
        model.cuda()

    for i, (rgb_images, depth_images, meta_datas, y_labels) in tqdm(enumerate(ponnet_loader)):
    #print("mtafile.shape=",metafile.shape)
        rgb_images, depth_images, meta_datas = rgb_images.to(device), depth_images.to(device), meta_datas.to(device)
        attout,depth_attout,output_y1, vizalize_attention = model(rgb_images,depth_images,meta_datas)
        if i == batchsize:
            break
    
    c_att = vizalize_attention[0]
    c_att = c_att.data.cpu()
    c_att = c_att.numpy()
    c_att_depth = vizalize_attention[1]
    c_att_depth = c_att_depth.data.cpu()
    c_att_depth = c_att_depth.numpy()
    d_inputs = depth_images
    d_inputs = d_inputs.data.cpu()
    d_inputs = d_inputs.numpy()
    rgb_inputs = rgb_images
    rgb_inputs = rgb_inputs.data.cpu()
    rgb_inputs = rgb_inputs.numpy()
    count = 0
    in_b, in_c, in_y, in_x = rgb_images.shape
    for item_img,item_depth_img, item_att, item_att_depth in zip(rgb_inputs, d_inputs, c_att, c_att_depth):

        v_img = ((item_img.transpose((1,2,0)) + 0.5 + [0.485, 0.456, 0.406]) * [0.229, 0.224, 0.225])* 256
        v_img = v_img[:, :, ::-1]
        depth_img = ((item_depth_img.transpose((1,2,0)) + 0.5 + [0.485, 0.456, 0.406]) * [0.229, 0.224, 0.225])* 256
        depth_img = depth_img[:, :, ::-1]
        resize_att = cv2.resize(item_att[0], (in_x, in_y))
        resize_att *= 255.
        resize_att_depth = cv2.resize(item_att_depth[0], (in_x, in_y))
        resize_att_depth *= 255.

        cv2.imwrite('stock0.png', depth_img)
        cv2.imwrite('stock1.png', v_img)
        cv2.imwrite('stock2.png', resize_att)
        cv2.imwrite('stock3.png', resize_att_depth)
        depth_img = cv2.imread('stock0.png')
        v_img = cv2.imread('stock1.png')
        vis_map = cv2.imread('stock2.png', 0)
        vis_map_depth = cv2.imread('stock3.png', 0)
        jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
        jet_map_depth = cv2.applyColorMap(vis_map_depth, cv2.COLORMAP_JET)
        print(v_img.shape,jet_map.shape)
        jet_map = cv2.add(v_img, jet_map)
        jet_map = cv2.addWeighted(v_img,0.8, jet_map,0.2,0)
        jet_map_depth = cv2.add(depth_img, jet_map_depth)
        jet_map_depth = cv2.addWeighted(depth_img,0.7, jet_map_depth,0.3,0)

        out_dir = path.join('output')
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        out_path = path.join(out_dir, 'attention', '{0:06d}.png'.format(count))
        cv2.imwrite(out_path, jet_map)
        out_path = path.join(out_dir, 'attention_depth', '{0:06d}.png'.format(count))
        cv2.imwrite(out_path, jet_map_depth)
        out_path = path.join(out_dir, 'raw', '{0:06d}.png'.format(count))
        cv2.imwrite(out_path, v_img)
        out_path = path.join(out_dir, 'raw_depth', '{0:06d}.png'.format(count))
        cv2.imwrite(out_path, depth_img)
        print(out_path)
        count += 1

def visualizing_attention_map(item_img, item_att, output_dir_name='output', count_index=0):
    v_img = ((item_img.transpose((1,2,0)) + 0.5 + [0.485, 0.456, 0.406]) * [0.229, 0.224, 0.225])* 256  
    v_img = v_img[:, :, ::-1]
    in_x, in_y = item_img.shape
    resize_att = cv2.resize(item_att[0], (in_x, in_y))
    resize_att *= 255.

    cv2.imwrite('stock1.png', v_img)
    cv2.imwrite('stock2.png', resize_att)

    v_img = cv2.imread('stock1.png')
    vis_map = cv2.imread('stock2.png', 0)
    
    jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
    jet_map = cv2.add(v_img, jet_map)
    jet_map = cv2.addWeighted(v_img,0.8, jet_map,0.2,0)


    out_dir = path.join(output_dir_name)
    if not path.exists(out_dir):
        os.mkdir(out_dir)
    out_path = path.join(out_dir, 'attention', '{0:06d}.png'.format(count_index))
    cv2.imwrite(out_path, jet_map)
    out_path = path.join(out_dir, 'raw', '{0:06d}.png'.format(count_index))
    cv2.imwrite(out_path, v_img)
    print(out_path)
