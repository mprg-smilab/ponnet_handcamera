#!/usr/bin/env python
# -*- coding: utf-8 -*-
from time import time
import datetime
from os import path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import cv2
from PIL import Image
import os
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#import tensorflow as tf
import argparse
import mlflow

#import ponnetmodel_single_parseption_branch_balancing as ponnet_model
#import ponnetmodel_handcam_end2end_NOmeta as ponnet_model
import ponnetmodel_handcam_end2end_typeD as ponnet_model

import ponnet_dataloader_handcamv2 as dataloader

parser = argparse.ArgumentParser(description='PonNetの学習を行います')
parser.add_argument('-b','--batchsize', default='150',type=int)
parser.add_argument('-t','--target_task', default='4',type=int)
parser.add_argument('-f','--path')
parser.add_argument('-m','--model', default='1',type=int)

args = parser.parse_args()

if args.model == 1:
    import ponnetmodel_handcam_end2end_NOmeta as ponnet_model
    is_hand_att = True
    model_name = "end2end_NOmeta"
    print("読み込みモデル：end2end_NOmeta")
if args.model == 2:
    import ponnetmodel_handcam_end2end_typeD as ponnet_model
    is_hand_att = True
    model_name = "end2end_typeD"
    print("読み込みモデル：end2end_typeD")

if __name__ == "__main__":
    
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    print('Use CUDA:', use_cuda)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Use device:', device)

    batch_size = args.batchsize
    epoch_num  = 10
    target_label = args.target_task
    Weight_RGB = 1
    Weight_Depth = 1
    Weight_Hand = 1
    Weight_perception = 1
    learning_rate = 0.00001
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epoch_num", epoch_num)
    mlflow.log_param("target_label", target_label)
    mlflow.log_param("Weight_RGB", Weight_RGB)
    mlflow.log_param("Weight_Depth", Weight_Depth)
    mlflow.log_param("Weight_Hand", Weight_Hand)
    mlflow.log_param("Weight_perception", Weight_perception)
    mlflow.log_param("learning_rate", learning_rate)
    print("Taget_label = ",target_label)

    model = ponnet_model.ponnet()
    if use_cuda:
        model.cuda()
        #Multi-GPU
        #model = torch.nn.DataParallel(model, device_ids = [0,1])

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    ponnet_dataset = dataloader.PonNetDataset(load_mode="train",dataset_ver='diff')
    ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=batch_size, shuffle=True)

    valid_dataset = dataloader.PonNetDataset(load_mode="valid",dataset_ver='diff')
    #valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=batch_size)
    test_dataset = dataloader.PonNetDataset(load_mode="test",dataset_ver='diff')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    if use_cuda:
        criterion.cuda()


    model.train()
    
    train_loss_list = []
    train_acc_list  = []
    valid_loss_list = []
    valid_acc_list  = []
    #---tensorboard----
    #now = datetime.datetime.now()
    # JSTとUTCの差分
    DIFF_JST_FROM_UTC = 9
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=DIFF_JST_FROM_UTC)
    logdir_name = "./logs/" + model_name + "/" + now.strftime('%Y%m%d_%H%M%S') + "/"
    #logdir_name = "./logs/debug/"
    print("logdir_name",logdir_name)
    writer = SummaryWriter(log_dir=logdir_name)
    memo = []
    memo.append("使用モデルponNetmodel_handcam_end2end_NOmeta.py,datasetv7 test,valid反転")
    memo.append("ターゲットタスク＝" + str(target_label) + "バッチサイズ" + str(batch_size) + "Epoch = " + str(epoch_num) + "learning rate = " + str(learning_rate))
    memo.append("WeightRGB = " + str(Weight_RGB) + "Weight_Perception = " + str(Weight_perception))
    np.savetxt(logdir_name + 'memo.txt', memo, fmt="%s")
    mlflow.log_artifact(logdir_name + 'memo.txt')
    start = time()
    step = 0
    best_accuracy = 0
    for epoch in tqdm(range(1,epoch_num+1)):
        train_loss = 0
        train_y_acc1 = 0
        train_y_acc_attention = 0
        valid_loss = 0
        valid_y_acc1 = 0
        #----train----
        model.train()
        for i, (rgb_images, depth_images, meta_datas, y_labels, hand_images, _targetrotation) in tqdm(enumerate(ponnet_loader)):
            step += 1
            rgb_images, depth_images = rgb_images.to(device), depth_images.to(device)
            meta_datas, y_labels = meta_datas.to(device), y_labels.to(device)
            hand_images = hand_images.to(device)

            optimizer.zero_grad()

            att_out, y_label_out1, visualization_attention_map = model(rgb_images,depth_images,meta_datas, hand_images)

            
            #5 class kara 1class totte sorewo 2class ni henkan siteru
            y_labels_acc1 = (y_labels[:,target_label]).long()
            
            #Loss
            loss_1 = criterion(y_label_out1[0], y_labels_acc1)
            writer.add_scalar("LOSS/Loss",loss_1.item(), step)
            loss_2 = criterion(att_out[0], y_labels_acc1) 
            writer.add_scalar("LOSS/RGB_ATT_Loss", loss_2.item(),step)
            loss_3 = criterion(att_out[1], y_labels_acc1) 
            writer.add_scalar("LOSS/Depth_ATT_Loss", loss_3.item(),step)
            loss_4 = criterion(att_out[2], y_labels_acc1) 
            writer.add_scalar("LOSS/Hamd_ATT_Loss", loss_4.item(),step)
            
            loss = (Weight_perception * loss_1) + ( Weight_RGB * loss_2) + ( Weight_Depth * loss_3) + ( Weight_Hand * loss_4)
            loss.backward(retain_graph=True)
            
            #tensorboad

            train_loss += loss.item()
            train_y_acc1  += (y_label_out1[0].max(1)[1] == y_labels_acc1).sum().item()
            #train_y_acc_attention  += (y_label_out1.max(1)[1] == y_labels_acc1).sum().item()
            #all_train_y_acc = (train_y_acc1 + train_y_acc_attention) / 2
            all_train_y_acc = train_y_acc1
            optimizer.step()

        avg_train_loss = train_loss / len(ponnet_loader.dataset)
        avg_train_acc  = all_train_y_acc  / len(ponnet_loader.dataset)
        writer.add_scalar("Train/loss",avg_train_loss,epoch)
        writer.add_scalar("Train/acc", avg_train_acc, epoch)
        
        #---valid--------------------------------
        model.eval
        with torch.no_grad():
            for i,(rgb_images, depth_images, meta_datas, y_labels, hand_images, _targetrotation) in tqdm(enumerate(valid_loader)):
                rgb_images, depth_images = rgb_images.to(device), depth_images.to(device)
                meta_datas, y_labels = meta_datas.to(device), y_labels.to(device) 
                hand_images = hand_images.to(device)
                
                att_out, y_label_out1, visualization_attention_map = model(rgb_images,depth_images,meta_datas, hand_images)
                
                y_labels_acc1 = (y_labels[:,target_label]).long()
                #Loss
                vall_loss_A = criterion(y_label_out1[0], y_labels_acc1)
                writer.add_scalar("LOSS/Valid_Loss",vall_loss_A.item(), step)
                vall_loss_B = criterion(att_out[0], y_labels_acc1) 
                writer.add_scalar("LOSS/Valid_RGB_ATT_Loss", vall_loss_B.item(),step)
                vall_loss_C = criterion(att_out[1], y_labels_acc1) 
                writer.add_scalar("LOSS/Depth_ATT_Loss", vall_loss_C.item(),step)
                vall_loss_D = criterion(att_out[2], y_labels_acc1) 
                writer.add_scalar("LOSS/Hamd_ATT_Loss", vall_loss_D.item(),step)
                vall_loss_Total = (Weight_perception * vall_loss_A) + ( Weight_RGB * vall_loss_B) + ( Weight_Depth * vall_loss_B) + ( Weight_Hand * vall_loss_D)

                #tensorboad
                writer.add_scalar("LOSS/Valid_TOTALLoss",vall_loss_Total,step)

                valid_loss += vall_loss_Total.item()
                valid_y_acc1  += (y_label_out1[0].max(1)[1] == y_labels_acc1).sum().item()
                

        avg_valid_loss = valid_loss / len(valid_loader.dataset)
        writer.add_scalar("Valid/loss",avg_valid_loss, epoch)
        avg_valid_acc  = valid_y_acc1  / len(valid_loader.dataset)
        writer.add_scalar("Valid/acc", avg_valid_acc, epoch)

        tqdm.write('Epoch [{}/{}], train_Loss: {loss:.4f}, train_Acc: {acc:.4f}, valid_Loss: {valid_loss:.4f}, valid_Acc: {valid_acc:.4f}'.format(epoch, epoch_num, i, loss=avg_train_loss, acc=avg_train_acc, valid_loss=avg_valid_loss, valid_acc=avg_valid_acc ))
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        valid_loss_list.append(avg_valid_loss)
        valid_acc_list.append(avg_valid_acc)
        mlflow.log_metric("avg_valid_acc", avg_valid_acc, step=epoch)
        if best_accuracy <  avg_valid_acc:
            best_accuracy = avg_valid_acc
            best_epcoh = epoch
            tqdm.write('Best Accuracy update, valid_Acc: {valid_acc:.4f}'.format(valid_acc=avg_valid_acc ))
            torch.save(model.state_dict(), logdir_name + "best_epoch_" + str(best_epcoh) + "_valid_acc" + str(best_accuracy) +'_model.ckpt')
            ##############
            test_acc = 0
            model.eval
            with torch.no_grad():
                for i,(rgb_images, depth_images, meta_datas, y_labels, hand_images, _targetrotation) in tqdm(enumerate(test_loader)):
                    rgb_images, depth_images = rgb_images.to(device), depth_images.to(device)
                    meta_datas, y_labels = meta_datas.to(device), y_labels.to(device) 
                    hand_images = hand_images.to(device)
                    
                    att_out, y_label_out1, visualization_attention_map = model(rgb_images,depth_images,meta_datas, hand_images)
                    
                    y_labels_acc1 = (y_labels[:,target_label]).long()
                    test_acc  += (y_label_out1[0].max(1)[1] == y_labels_acc1).sum().item()
            avg_test_acc  = test_acc  / len(test_loader.dataset)
            tqdm.write('test_acc: {test_acc:.4f}'.format(test_acc=avg_test_acc ))

        if epoch%10 == 0:
            torch.save(model.state_dict(), logdir_name + str(epoch) +'_model.ckpt')

        
    #tensorboard writerを閉じる
    writer.close()
    print('Best Accuracy Epoch [{}] valid_Acc: {valid_acc:.4f}'.format(best_epcoh, valid_acc=best_accuracy ))
    memo.append('Best Accuracy Epoch [{}] valid_Acc: {valid_acc:.4f}'.format(best_epcoh, valid_acc=best_accuracy ))
    np.savetxt(logdir_name + 'memo.txt', memo, fmt="%s")

    np.savetxt(logdir_name + 'train_loss.txt',train_loss_list)
    np.savetxt(logdir_name + 'train_acc.txt', train_acc_list)
    np.savetxt(logdir_name + 'valid_loss.txt',valid_loss_list)
    np.savetxt(logdir_name + 'valid_acc.txt', valid_acc_list)
    #network save
    filename3 = logdir_name + 'model.ckpt'
    torch.save(model.state_dict(), filename3)
    mlflow.end_run()