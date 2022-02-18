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
if args.model == 4:
    import ponnetmodel_baseline_NOmeta as ponnet_model
    is_hand_att = False
    print("読み込みモデル：baseline_NOmet")    
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

    #item_att=(1, 14, 14)
    resize_att = cv2.resize(item_att[0], (in_x, in_y))
    resize_att *= 255.

    cv2.imwrite('stock1.png', v_img)
    cv2.imwrite('stock2.png', resize_att)

    v_img = cv2.imread('stock1.png')
    vis_map = cv2.imread('stock2.png', 0)
    if image_type == 'depth':
        #v_img, _ = cv2.decolor(v_img)
        #グレースケールに変換
        row_depth  = v_img
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
    if image_type == 'depth':
        out_path = path.join(out_dir_raw, '{0:05d}_{type}_raw_normal_vector.png'.format(count_index, type=image_type))
        cv2.imwrite(out_path, row_depth)
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

if __name__ == "__main__":
    
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    #print('Use CUDA:', use_cuda)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = "cuda:1"
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

    ponnet_dataset = dataloader.PonNetDataset(root="./handcamv10_test_dataset/",load_mode=args.eval_version)
    #ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=batch_size, drop_last=True,shuffle = True)
    ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=batch_size)
    #---valid--------------------------------
    model.eval
    valid_y_acc1 = 0
    insertion_y_acc1 = 0
    cm = 0
    pose_coll = 0
    with torch.no_grad():
        for i, (rgb_images, depth_images, meta_datas, y_labels, hand_images, target_rotation_labels) in tqdm(enumerate(ponnet_loader)):
            rgb_images, depth_images = rgb_images.to(device), depth_images.to(device)
            meta_datas, y_labels = meta_datas.to(device), y_labels.to(device)
            hand_images, target_rotation_labels = hand_images.to(device), target_rotation_labels.to(device)

            if args.model == 4:
                att_out, y_label_out, visual_att_map = model(rgb_images, depth_images, meta_datas)
            else:
                att_out, y_label_out, visual_att_map = model(rgb_images, depth_images, meta_datas, hand_images)
            #使用するlabelのみにする
            y_labels_acc1 = (y_labels[:,target_label]).long()
            
            

            #データセットの姿勢と衝突の割合の計算
            #if target_rotation_labels == 1 and y_labels_acc1 == 1:
                #衝突ありで姿勢1なら
            #    pose_coll+=1

            valid_y_acc1 += (y_label_out[0].max(1)[1] == y_labels_acc1).sum().item()
            cm += confusion_matrix(y_labels_acc1.to("cpu"),  y_label_out[0].max(1)[1].to("cpu"))
            
            if args.is_visualizing == True:
                att_rgb = (visual_att_map[0].data.cpu()).numpy()
                att_depth = (visual_att_map[1].data.cpu()).numpy()
                img_rgb = (rgb_images.data.cpu()).numpy()
                img_depth = (depth_images.data.cpu()).numpy()
                i = 0
                for (item_img_rgb, item_img_depth, item_att_rgb, item_att_depth) in zip(img_rgb, img_depth, att_rgb, att_depth):
                    current_true_y = y_labels_acc1[i].data.cpu().numpy()
                    current_pred_y = y_label_out[0].max(1)[1].data.cpu().numpy()
                    current_pred_y = current_pred_y[i]
                    if current_true_y == 1 and current_pred_y == 1:
                        res = "TP"
                    if current_true_y == 1 and current_pred_y == 0:
                        res = "FN"
                    if current_true_y == 0 and current_pred_y == 0:
                        res = "TN"
                    if current_true_y == 0 and current_pred_y == 1:
                        res = "FP" 
                    visualizing_attention_map(item_img_rgb, item_att_rgb, args.output_dir_name +'/'+ str(args.eval_version)+'/' + res, i, image_type='rgb')
                    visualizing_attention_map(item_img_depth, item_att_depth, args.output_dir_name +'/'+str(args.eval_version)+'/'+ res, count_index=i, image_type='depth')
                    i += 1
                if is_hand_att:
                    att_hand = (visual_att_map[2].data.cpu()).numpy()
                    img_hand = (hand_images.data.cpu()).numpy()
                    i = 0
                    for ( item_img_hand, item_att_hand) in zip(img_hand, att_hand):
                        current_true_y = y_labels_acc1[i].data.cpu().numpy()
                        current_pred_y = y_label_out[0].max(1)[1].data.cpu().numpy()
                        current_pred_y = current_pred_y[i]
                        if current_true_y == 1 and current_pred_y == 1:
                            res = "TP"
                        if current_true_y == 1 and current_pred_y == 0:
                            res = "FN"
                        if current_true_y == 0 and current_pred_y == 0:
                            res = "TN"
                        if current_true_y == 0 and current_pred_y == 1:
                            res = "FP" 
                        visualizing_attention_map(item_img_hand, item_att_hand, args.output_dir_name +'/'+str(args.eval_version)+'/' + res, i, image_type='hand')
                        i += 1


            #break
    avg_valid_acc  = valid_y_acc1  / len(ponnet_loader.dataset)
    avg_insertion_acc  = insertion_y_acc1  / len(ponnet_loader.dataset)
    print("衝突の割合",pose_coll)
    print(cm)
    print("タスク{task}のテスト結果：{acc:.4f}".format(task=target_label,acc=avg_valid_acc))
    tn, fp, fn, tp = cm.flatten()
    print("tn, fp, fn, tp",tn, fp, fn, tp)
    print("正解率(accuracy)",(tp+tn)/(tp+tn+fp+fn))
    print("適合率(precison)",(tp)/(tp+fp))
    print("再現率(recall)",(tp)/(tp+fn))
    print("F1値（F1-measure）",(2*tp)/(2*tp+fp+fn))

if False:
    att_rgb = (visual_att_map[0].data.cpu()).numpy()
    att_depth = (visual_att_map[1].data.cpu()).numpy()
    att_hand = (visual_att_map[2].data.cpu()).numpy()

    img_rgb = (rgb_images.data.cpu()).numpy()
    img_depth = (depth_images.data.cpu()).numpy()
    img_hand = (hand_images.data.cpu()).numpy()


    i = 0
    list_att_image_hand = []
    for (item_img_rgb, item_img_depth, item_img_hand, item_att_rgb, item_att_depth, item_att_hand) in zip(img_rgb, img_depth, img_hand, att_rgb, att_depth, att_hand):
        visualizing_attention_map(item_img_rgb, item_att_rgb, 'output/handcam/rgb', i)
        visualizing_attention_map(item_img_depth, item_att_depth, 'output/handcam/depth', i)

        if i < 1:
            att_image_hand = visualizing_attention_map(item_img_hand, item_att_hand, 'output/handcam/hand', i)
            att_image_hand = att_image_hand.transpose(2, 0, 1)
            att_image_hand = torch.from_numpy(att_image_hand)
            att_image_hand = torch.unsqueeze(att_image_hand,0)
            #imshow(torchvision.utils.make_grid(att_image_hand),y_labels_acc1.data.cpu().numpy(),y_label_out[0].max(1)[1].data.cpu().numpy(),save_name="att_hand")
        else:
            att_image_hand2 = visualizing_attention_map(item_img_hand, item_att_hand, 'output/handcam/hand', i)
            att_image_hand2 = att_image_hand2.transpose(2, 0, 1)
            att_image_hand2 = torch.from_numpy(att_image_hand2)
            att_image_hand2 = torch.unsqueeze(att_image_hand2,0)
            list_att_image_hand = [att_image_hand, att_image_hand2]
            att_image_hand = torch.cat(list_att_image_hand,dim=0)
        if i == 7:
            break
        i += 1
    print('att_image_hand',att_image_hand.shape)
    #imshow(torchvision.utils.make_grid(att_image_hand),y_labels_acc1.data.cpu().numpy(),y_label_out[0].max(1)[1].data.cpu().numpy(),save_name="att_hand")
    classes = ('safe', 'out')
    fig = plt.figure()


    # 適当な訓練セットの画像を取得
    dataiter = iter(ponnet_loader)
    images, depth, meta, labels, hand, rotation = dataiter.next()
    labels = (labels[:,target_label]).long()
    # 画像の表示
    #print("images.shape",images.shape)
    #imshow(torchvision.utils.make_grid(images))

    #print(labels.shape)
    # ラベルの表示
    #print(' '.join('%5s' % classes[labels[0][j]] for j in range(4)))
