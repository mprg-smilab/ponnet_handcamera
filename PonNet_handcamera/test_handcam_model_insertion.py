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
import os
from os import path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch.nn as nn


import ponnetmodel_handcam_end2end_NOmeta as ponnet_model
import ponnetmodel_handcam_2stage_NOmeta as ponnet_model

import ponnet_dataloader_handcamv2 as dataloader

parser = argparse.ArgumentParser(description='PonNetの学習を行います')
parser.add_argument('-b','--batchsize', default='150',type=int)
parser.add_argument('-t','--target_task', default='4',type=int)
parser.add_argument('-e','--eval_version', default='test',type=str)
parser.add_argument('-f','--path')
parser.add_argument('-m','--model', default='1',type=int)
parser.add_argument('-v','--is_visualizing', default='False',type=bool)
parser.add_argument('-o','--output_dir_name', default='output',type=str)
parser.add_argument('-d','--dataset_name', default='./handcamv8_dataset/',type=str)


args = parser.parse_args()

if args.model == 1:
    import ponnetmodel_handcam_end2end_NOmeta as ponnet_model
    is_hand_att = True
    print("読み込みモデル：end2end_NOmeta")
if args.model == 2:
    import ponnetmodel_handcam_2stage_NOmeta as ponnet_model
    is_hand_att = True
    print("読み込みモデル：2stage_NOmeta")
if args.model == 3:
    import ponnetmodel_handcam_2stage_NOmeta_typeB as ponnet_model
    is_hand_att = True
    print("読み込みモデル：2stage_NOmeta_typeB")    
if args.model == 101:
    import ponnetmodel_baseline_NOmeta as ponnet_model
    is_hand_att = False
    print("読み込みモデル：baseline_NOmet")
if args.model == 100:
    import ponnetmodel_baseline as ponnet_model
    is_hand_att = False
    print("読み込みモデル：baseline")        
if args.model == 5:
    import ponnetmodel_handcam_end2end_typeC as ponnet_model
    is_hand_att = False
    print("読み込みモデル：end2end_typeC")
if args.model == 6:
    import ponnetmodel_handcam_end2end_typeB as ponnet_model
    is_hand_att = False
    print("読み込みモデル：end2end_typeB")
if args.model == 7:
    import ponnetmodel_handcam_end2end_typeD as ponnet_model
    is_hand_att = True
    print("読み込みモデル：end2end_typeD")
if args.model == 8:
    import ponnetmodel_handcam_end2end_typeE as ponnet_model
    is_hand_att = False
    print("読み込みモデル：end2end_typeE")

def visualizing_attention_map(item_img, item_att, output_dir_name='output', count_index=0, image_type='rgb'):
    #print("item_img.shape", item_img.shape)
    #print("item_att.shape", item_att.shape)
    #v_img = ((item_img.transpose((1,2,0)) + 0.5 + [0.485, 0.456, 0.406]) * [0.229, 0.224, 0.225]) * 255
    v_img = item_img.transpose(1,2,0) * 255
    v_img = v_img[:, :, ::-1]
    in_chanel, in_x, in_y = item_img.shape

    #print("item_img.shape", zitem_img.shape)
    #in_x, in_y = 224,224
    resize_att = cv2.resize(item_att[0], (in_x, in_y))
    resize_att *= 255.

    cv2.imwrite('stock1.png', v_img)
    cv2.imwrite('stock2.png', resize_att)

    v_img = cv2.imread('stock1.png')
    vis_map = cv2.imread('stock2.png', 0)
    if image_type == 'depth':
        #v_img, _ = cv2.decolor(v_img)
        #グレースケールに変換
        v_img = cv2.cvtColor(v_img, cv2.COLOR_RGB2GRAY)
        v_img = cv2.cvtColor(v_img, cv2.COLOR_GRAY2RGB)
        #print("vimage shape",v_img.shape)

    
    jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
    #jet_map = cv2.add(v_img, jet_map)
    add_img = cv2.addWeighted(v_img,0.8, jet_map,0.2,0)


    out_dir = path.join(output_dir_name)
    out_dir_att = path.join(out_dir, 'attention')
    out_dir_raw = path.join(out_dir, 'raw')
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(out_dir_att):
        os.makedirs(out_dir_att)
    if not path.exists(out_dir_raw):
        os.makedirs(out_dir_raw)
    
    out_path = path.join(out_dir_att, '{0:05d}_{type}_att.png'.format(count_index, type=image_type))
    cv2.imwrite(out_path, jet_map)
    out_path = path.join(out_dir_att, '{0:05d}_{type}_add.png'.format(count_index, type=image_type))
    cv2.imwrite(out_path, add_img)
    out_path = path.join(out_dir_raw, '{0:05d}_{type}_raw.png'.format(count_index, type=image_type))
    cv2.imwrite(out_path, v_img)
    return jet_map

# 画像の表示関数
def imshow(img,y_label,y_dash_label,save_name):
    #img = img / 2 + 0.5     # 正規化を戻す
    npimg = img.numpy()
    npimg = cv2.cvtColor(npimg[:,], cv2.COLOR_BGR2RGB)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)
    plt.xlabel('Grand Truth' + str(y_label)+ "\n" + 'result label' + str(y_dash_label))
    plt.show()
    plt.savefig("output/handcam/"+ save_name +".png")
    plt.imsave("images.png", np.transpose(npimg, (1, 2, 0)))

def mask(input_images, input_attention, insertion_th):
    images = input_images
    attention =input_attention

    attention=(attention-torch.min(attention))/(torch.max(attention)-torch.min(attention))
    #マスクの作成
    norm_vis = attention
    #print(norm_vis)
    norm_vis[norm_vis >= insertion_th] = 1
    norm_vis[norm_vis != 1.0] = 0
    #入力画像の型を変更
    insertion_inputs = images.data.cpu()
    insertion_inputs = insertion_inputs.numpy()
    insertion_inputs = insertion_inputs.reshape(-1,3,224,224)
    #マスクのupsampling
    norm_vis = F.interpolate(norm_vis,(224, 224),mode='bilinear',align_corners=True)
    norm_vis = norm_vis.data.cpu()
    norm_vis = norm_vis.numpy()
    #入力画像とマスクの乗算
    norm_vis_input = insertion_inputs * norm_vis
    #乗算した画像をモデルに入力できるように変更 
    norm_vis_input_var = torch.from_numpy(norm_vis_input.astype(np.float32)).clone()
    norm_vis_input_var = norm_vis_input_var.cuda()
    norm_vis_input_var = torch.autograd.Variable(norm_vis_input_var, requires_grad=True)
    norm_vis_input_var = norm_vis_input_var.view(-1,3,224,224)
    return norm_vis_input_var

def eval(insertion_th=0.1):
    use_cuda = torch.cuda.is_available()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print('Use device:', device)
    batch_size = args.batchsize
    target_label = args.target_task

    model = ponnet_model.ponnet()

    state_dict = torch.load(args.path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    if use_cuda:
        model.to(device)

    ponnet_dataset = dataloader.PonNetDataset(root=args.dataset_name,load_mode=args.eval_version)
    #ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=batch_size, drop_last=True,shuffle = True)
    ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=batch_size)
    #---valid--------------------------------
    model.eval
    insertion_y_acc1 = 0
    upsample = nn.Upsample([224,224], mode='bilinear', align_corners=False)
    with torch.no_grad():
        for i, (rgb_images, depth_images, meta_datas, y_labels, hand_images, target_rotation_labels) in enumerate(ponnet_loader):
            rgb_images, depth_images = rgb_images.to(device), depth_images.to(device)
            meta_datas, y_labels = meta_datas.to(device), y_labels.to(device)
            hand_images, target_rotation_labels = hand_images.to(device), target_rotation_labels.to(device)

            #使用するラベルを選択
            y_labels_acc1 = (y_labels[:,target_label]).long()

            #PonNet(ハンド画像不使用)
            if args.model >= 100:
                att_out, y_label_out, visual_att_map = model(rgb_images, depth_images, meta_datas)

                #画像サイズを224*224に統一
                rgb_images = upsample(rgb_images)
                depth_images = upsample(depth_images)

                #マスク処理
                rgb_norm_vis_input_var = mask(rgb_images, visual_att_map[0], insertion_th)
                depth_norm_vis_input_var = mask(depth_images, visual_att_map[1], insertion_th)
                #もう一度モデルに入力
                _, insertion_output, visual_att_map = model(rgb_norm_vis_input_var, depth_norm_vis_input_var, meta_datas)
                insertion_y_acc1 += (insertion_output[0].max(1)[1] == y_labels_acc1).sum().item()
            
            #ハンド画像あり
            else:
                att_out, y_label_out, visual_att_map = model(rgb_images, depth_images, meta_datas, hand_images)

                #画像サイズを224*224に統一
                rgb_images = upsample(rgb_images)
                depth_images = upsample(depth_images)

                #マスク処理
                rgb_norm_vis_input_var = mask(rgb_images, visual_att_map[0], insertion_th)
                depth_norm_vis_input_var = mask(depth_images, visual_att_map[1], insertion_th)
                hand_norm_vis_input_var = mask(hand_images, visual_att_map[2], insertion_th)
                #もう一度モデルに入力
                _, insertion_output, visual_att_map = model(rgb_norm_vis_input_var, depth_norm_vis_input_var, meta_datas, hand_norm_vis_input_var)
                insertion_y_acc1 += (insertion_output[0].max(1)[1] == y_labels_acc1).sum().item()
            #使用するlabelのみにする
            
            #insertonとデリーション
                
            if False:
                att_hand = (visual_att_map[2].data.cpu()).numpy()
                img_hand = (hand_images.data.cpu()).numpy()
                #for ( item_img_hand, item_att_hand) in zip(img_hand, att_hand):
                #    res = "insertion"
                #    visualizing_attention_map(item_img_hand, item_att_hand, args.output_dir_name +'/'+str(args.eval_version)+'/' + res, i, image_type='hand')



    avg_insertion_acc  = insertion_y_acc1  / len(ponnet_loader.dataset)

    tqdm.write("insertionの割合{insertion_th}のテスト結果:{acc:.4f}".format(insertion_th=insertion_th,acc=avg_insertion_acc))


if __name__ == "__main__":
    #per = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print("使用データセット{}".format(args.dataset_name))
    per = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    print("per = {}".format(per))
    for i in tqdm(per):
        eval(i)