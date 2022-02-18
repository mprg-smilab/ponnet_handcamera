#!/usr/bin/env python
# -*- coding: utf-8 -*-
#train_multi_task_model.pyを改造して作った
import torch
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import cv2

#import ponnetmodel_typeA as ponnet_model
#import ponnetmodel_single_parseption_branch_balancing as ponnet_model
#import ponnetmodel_baseline_NOresblock as ponnet_model
import ponnetmodel_baseline_NOmeta as ponnet_model
#import ponnetmodel_multi_parception_branch_balancing as ponnet_model
import ponnet_dataloader_handcam as dataloader

parser = argparse.ArgumentParser(description='PonNetの学習を行います')
parser.add_argument('-b','--batchsize', default='150',type=int)
parser.add_argument('-t','--target_task', default='0',type=int)
parser.add_argument('-e','--eval_vertion', default='test',type=str)
parser.add_argument('-f','--path')
args = parser.parse_args()


if __name__ == "__main__":
    
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    print('Use CUDA:', use_cuda)

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cuda:1"
    print('Use device:', device)
    batch_size = args.batchsize
    target_label = args.target_task

    model = ponnet_model.ponnet()
    #model.load_state_dict(torch.load(args.path),strict=False)
    #model.load_state_dict(torch.load(args.path))
    

    state_dict = torch.load(args.path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    #from collections import OrderedDict
    #new_state_dict = OrderedDict()
    
    #for k, v in state_dict.items():
    #    if 'module' in k:
    #        k = k.replace('module.', '')
    #    new_state_dict[k] = v
    #model.load_state_dict(new_state_dict)


    if use_cuda:
        #model.cuda()
        model.to(device)
        #Multi-GPU
        #model = torch.nn.DataParallel(model, device_ids = [0,1])

    ponnet_dataset = dataloader.PonNetDataset(load_mode=args.eval_vertion, dataset_ver='base')
    ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=batch_size, drop_last=True)
    #---valid--------------------------------
    model.eval
    valid_y_acc1 = 0
    with torch.no_grad():
        for i,(rgb_images, depth_images, meta_datas, y_labels) in tqdm(enumerate(ponnet_loader)):
            rgb_images, depth_images = rgb_images.to(device), depth_images.to(device)
            meta_datas, y_labels = meta_datas.to(device), y_labels.to(device) 

            
            rgb_att_out, depth_att_out, y_label_out1, visualization_attention_map = model(rgb_images,depth_images,meta_datas)
            
            y_labels_acc1 = (y_labels[:,target_label]).long()

            valid_y_acc1  += (y_label_out1.max(1)[1] == y_labels_acc1).sum().item()
            #print(y_label_out1.max(1)[1] == y_labels_acc1)
            

    avg_valid_acc  = valid_y_acc1  / len(ponnet_loader.dataset)

print("タスク0のテスト結果：{acc:.4f}".format(acc=avg_valid_acc))


classes = ('safe', 'out')
fig = plt.figure()
# 画像の表示関数
def imshow(img):
    #img = img / 2 + 0.5     # 正規化を戻す
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.imsave("images.png", np.transpose(npimg, (1, 2, 0)))

# 適当な訓練セットの画像を取得
dataiter = iter(ponnet_loader)
images, depth, meta, labels = dataiter.next()
labels = (labels[:,target_label]).long()
# 画像の表示
imshow(torchvision.utils.make_grid(images))

#print(labels.shape)
# ラベルの表示
#print(' '.join('%5s' % classes[labels[0][j]] for j in range(4)))
