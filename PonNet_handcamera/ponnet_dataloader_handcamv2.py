#ハンドカメラに対応したデータローダです
from time import time
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
import random
import ponnetmodel_handcam_2stage as ponnet_model
import glob
from tqdm import tqdm

class PonNetDataset(torch.utils.data.Dataset):
    
    def __init__(self,root="./bf_dataset_v2/",load_mode="train", dataset_ver='hand'):
        super().__init__()
        self.root = root
        self.load_mode = load_mode
        self.dataset_ver = dataset_ver
        self.rgb_before_images_list = []
        self.depth_before_images_list = []
        self.rgb_after_images_list = []
        self.depth_after_images_list = []
        self.meta_datas_list = []
        self.y_labels_list = []
        self.hand_images_list = []
        self.target_rotation_list = []

        if self.load_mode == "train":
            dir_list = sorted(glob.glob(self.root+"train/**"))
        elif self.load_mode == "valid":
            dir_list = sorted(glob.glob(self.root+"valid/**"))
        elif self.load_mode == "test":
            dir_list = sorted(glob.glob(self.root+"test/**"))
        
        for dir_path in dir_list:
            
            #とりあえず，rgb_beforeをdiffに変更してみる
            rgb_before_list = sorted(glob.glob(dir_path+"/*_rgb_before.jpg"))
            self.rgb_before_images_list.extend(rgb_before_list)

            depth_before_list = sorted(glob.glob(dir_path+"/*_depth_before.jpg"))
            self.depth_before_images_list.extend(depth_before_list)

            hand_list = sorted(glob.glob(dir_path+"/*_hand.jpg"))
            self.hand_images_list.extend(hand_list)

            #before afterの差分画像の場合
            if self.dataset_ver == 'diff':

                rgb_diff_list = sorted(glob.glob(dir_path+"/*_rgb_diff.jpg"))
                self.rgb_before_images_list.extend(rgb_diff_list)
                depth_diff_list = sorted(glob.glob(dir_path+"/*_depth_diff.jpg"))
                self.depth_before_images_list.extend(depth_diff_list)

            if self.dataset_ver == 'before_after':

                rgb_after_list = sorted(glob.glob(dir_path+"/*_rgb_after.jpg"))
                self.rgb_after_images_list.extend(rgb_after_list)

                depth_after_list = sorted(glob.glob(dir_path+"/*_depth_after.jpg"))
                self.depth_after_images_list.extend(depth_after_list)


            meta_list = sorted(glob.glob(dir_path+"/*_rotation-meta.csv"))
            self.meta_datas_list.extend(meta_list)

            y_list = sorted(glob.glob(dir_path+"/*_y.csv"))
            self.y_labels_list.extend(y_list)

            hand_list = sorted(glob.glob(dir_path+"/*_hand.jpg"))
            self.hand_images_list.extend(hand_list)

            rotation_list = sorted(glob.glob(dir_path+"/*_rotation.csv"))
            self.target_rotation_list.extend(rotation_list)
           
    
    def __getitem__(self, index):
        rgb_before_image = self.rgb_before_images_list[index]
        depth_before_image = self.depth_before_images_list[index]

        if self.dataset_ver == 'before_after':
            rgb_after_image = self.rgb_after_images_list[index]
            depth_after_image = self.depth_after_images_list[index]
        
        hand_image = self.hand_images_list[index]
        #rgb_diff_image = self.rgb_diff_images_list[index]
        meta_data = self.meta_datas_list[index]
        y_label = self.y_labels_list[index]
        #target_rotation_label = self.target_rotation_list[index]
        
        with open(rgb_before_image,'rb') as f:
            rgb_before_image = Image.open(f).convert('RGB')
            #tensor ni henkan and normalization?
            rgb_before_image = TF.to_tensor(rgb_before_image)
       
        with open(depth_before_image,'rb') as f:
            depth_before_image = Image.open(f).convert('RGB')
            depth_before_image = TF.to_tensor(depth_before_image)


        if self.dataset_ver == 'before_after':
            with open(rgb_after_image,'rb') as f:
                rgb_after_image = Image.open(f).convert('RGB')
                #tensor ni henkan and normalization?
                rgb_after_image = TF.to_tensor(rgb_after_image)
            with open(depth_after_image,'rb') as f:
                depth_after_image = Image.open(f).convert('RGB')
                depth_after_image = TF.to_tensor(depth_after_image)

        with open(hand_image, 'rb') as f:
            hand_image = Image.open(f).convert('RGB')
            hand_image = TF.to_tensor(hand_image)

        with open(meta_data,'rb') as f:
            meta_data = (np.genfromtxt(f, delimiter=", ",dtype=np.float32))#"," or " "

        with open(y_label,'rb') as f:
            y_label = (np.genfromtxt(f, delimiter=",",dtype=np.float32))#"," or " "
            #y_label = (np.genfromtxt(f, delimiter=" ",dtype=np.int64))#"," or " "
        #with open(target_rotation_label,'rb') as f:
        #    target_rotation_label = (np.genfromtxt(f, delimiter=",", dtype=np.float32))#"," or " "

        if self.dataset_ver == 'before_after':
            return rgb_before_image, depth_before_image, meta_data, y_label, hand_image, target_rotation_label, rgb_after_image, depth_after_image
        if self.dataset_ver == 'hand':
            return rgb_before_image, depth_before_image, meta_data, y_label, hand_image, target_rotation_label
        if self.dataset_ver == 'diff':
            return rgb_before_image, depth_before_image, meta_data, y_label, hand_image, y_label
        if self.dataset_ver == 'base':
            return rgb_image, depth_image, meta_data, y_label

    def __len__(self):

        return len(self.rgb_before_images_list)


if __name__ == "__main__":

    #---kakunin
    ponnet_dataset = PonNetDataset(load_mode="test")
    ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=1)
    loss = nn.CrossEntropyLoss
    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    model = ponnet_model.ponnet()
    coll_All = 0
    coll_NO = 0
    coll_type1 = 0
    coll_type2 = 0
    target_label = 4
    pose_num1 = 0
    pose_num2 = 0
    aitem_1 = -2.8200
    aitem_2 = 0.1893
    aitem_3 = 0.0629
    aitem_4 = 0.5845
    aitem_5 = 1.7044
    aitem_6 = -0.3932
    aitem_7 = -0.4590
    aitem_8 = 0.3281
    aitem_9 = -0.6152
    coll_aitem1 = 0
    for i, (rgb_images, depth_images, meta_datas, y_labels, hand_images, target_rotation_label) in tqdm(enumerate(ponnet_loader)):
        y_labels_gt = (y_labels[:,target_label]).long()
        #print(meta_datas[0][-1].shape)
        
        if y_labels_gt == 1:
            coll_All += 1
            #if aitem_1.to(torch.float32) == meta_datas[0][-1]:
            #   coll_aitem1 += 1
        else:
            coll_NO += 1
        if target_rotation_label == 1 and y_labels_gt == 1:
            coll_type1 += 1
        if target_rotation_label == 0 and y_labels_gt == 1:
            coll_type2 += 1
        if target_rotation_label == 1:
            pose_num1 += 1
        if target_rotation_label == 0:
            pose_num2 += 1
    print("衝突有りの数",coll_All)
    print("衝突なしの数",coll_NO)
    print("姿勢１かつ衝突有りの数",coll_type1)
    print("姿勢０かつ衝突有りの数",coll_type2)
    print("姿勢1のかず",pose_num1)
    print("姿勢０のかず",pose_num2)
    print("アイテム１",coll_aitem1)