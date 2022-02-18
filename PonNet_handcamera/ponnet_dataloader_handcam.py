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
from tqdm import tqdm

class PonNetDataset(torch.utils.data.Dataset):
    
    def __init__(self,root="./handcamv5_dataset",load_mode="train", dataset_ver='hand'):
        super().__init__()
        self.root = root
        self.load_mode = load_mode
        self.dataset_ver = dataset_ver
        dataset_dir = ''#datasetがponnet_dataset/output/
        dataset_num = 12000
        start_num = 0
        end_num   = dataset_num
        zfill_number = 5 #5桁そろえ
        self.rgb_images_list = []
        self.depth_images_list = []
        self.meta_datas_list = []
        self.y_labels_list = []
        self.hand_images_list = []
        self.target_rotation_list = []

        # if self.load_mode == "train":
        #     start_num = 0
        #     end_num   = int(dataset_num * 0.8)
        #     maxindex = 10000    
        # if self.load_mode == "valid":
        #     start_num = int(dataset_num * 0.8)    
        #     end_num   = int(dataset_num * 0.9)
        #     maxindex = 1000
        # if self.load_mode == "test":
        #     start_num = int(dataset_num * 0.9)    
        #     end_num   = dataset_num 
        #     maxindex = 1000
        if self.load_mode == "test2":
            root="./handcamv3_2_test"
            dataset_dir = ''
            dataset_num = 1000
        
        count_index = 1
        int_list = [x for x in range(1, dataset_num+1)] 
        random.seed(1)
        random_list = random.sample(int_list, len(int_list))

        for index, total_index in zip(random_list, int_list):
            if self.load_mode == "test" and total_index < 1001:
                self.rgb_images_list.append(os.path.join(root, dataset_dir, str(index).zfill(zfill_number)+ "_rgb.jpg"))
                self.depth_images_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_depth.jpg"))
                self.meta_datas_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_meta.csv"))
                self.y_labels_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_y.csv"))
                self.hand_images_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_hand.jpg"))
                self.target_rotation_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_target_rotation.csv"))
            elif self.load_mode == "valid" and 1001 <= total_index and total_index < 2001:
                self.rgb_images_list.append(os.path.join(root, dataset_dir, str(index).zfill(zfill_number)+ "_rgb.jpg"))
                self.depth_images_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_depth.jpg"))
                self.meta_datas_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_meta.csv"))
                self.y_labels_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_y.csv"))
                self.hand_images_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_hand.jpg"))
                self.target_rotation_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_target_rotation.csv"))
            elif self.load_mode == "train" and 2001 <= total_index:
                self.rgb_images_list.append(os.path.join(root, dataset_dir, str(index).zfill(zfill_number)+ "_rgb.jpg"))
                self.depth_images_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_depth.jpg"))
                self.meta_datas_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_meta.csv"))
                self.y_labels_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_y.csv"))
                self.hand_images_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_hand.jpg"))
                self.target_rotation_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_target_rotation.csv"))
            # elif self.load_mode == "test2":
            #     self.rgb_images_list.append(os.path.join(root, dataset_dir, str(index).zfill(zfill_number)+ "_rgb.jpg"))
            #     self.depth_images_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_depth.jpg"))
            #     self.meta_datas_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_meta.csv"))
            #     self.y_labels_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_y.csv"))
            #     self.hand_images_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_hand.jpg"))
            #     self.target_rotation_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_target_rotation.csv"))
            # elif self.load_mode == "test3" and total_index < 500:
            #     self.rgb_images_list.append(os.path.join(root, dataset_dir, str(index).zfill(zfill_number)+ "_rgb.jpg"))
            #     self.depth_images_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_depth.jpg"))
            #     self.meta_datas_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_meta.csv"))
            #     self.y_labels_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_y.csv"))
            #     self.hand_images_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_hand.jpg"))
            #     self.target_rotation_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_target_rotation.csv"))
        
        
        # for index, total_index in zip(int_list, int_list):
        #     if self.load_mode == "train" and total_index > 500 and total_index < 1000:
        #         root="./handcamv3_2_test"
        #         dataset_dir = ''
        #         dataset_num = 1000
        #         self.rgb_images_list.append(os.path.join(root, dataset_dir, str(index).zfill(zfill_number)+ "_rgb.jpg"))
        #         self.depth_images_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_depth.jpg"))
        #         self.meta_datas_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_meta.csv"))
        #         self.y_labels_list.append(os.path.join(root,dataset_dir, str(index).zfill(zfill_number)+ "_y.csv"))
        #         self.hand_images_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_hand.jpg"))
        #         self.target_rotation_list.append(os.path.join(root,dataset_dir,str(index).zfill(zfill_number)+ "_target_rotation.csv"))
        #print('rgb_images_list',self.rgb_images_list)
    
    def __getitem__(self, index):
        rgb_image = self.rgb_images_list[index]
        depth_image = self.depth_images_list[index]
        hand_image = self.hand_images_list[index]
        meta_data = self.meta_datas_list[index]
        y_label = self.y_labels_list[index]
        target_rotation_label = self.target_rotation_list[index]
        with open(rgb_image,'rb') as f:
            rgb_image = Image.open(f).convert('RGB')
            #tensor ni henkan and normalization?
            rgb_image = TF.to_tensor(rgb_image)
        with open(depth_image,'rb') as f:
            depth_image = Image.open(f).convert('RGB')
            depth_image = TF.to_tensor(depth_image)
        with open(hand_image, 'rb') as f:
            hand_image = Image.open(f).convert('RGB')
            hand_image = TF.to_tensor(hand_image)

        with open(meta_data,'rb') as f:
            meta_data = (np.genfromtxt(f, delimiter=", ",dtype=np.float32))#"," or " "

        with open(y_label,'rb') as f:
            y_label = (np.genfromtxt(f, delimiter=",",dtype=np.float32))#"," or " "
            #y_label = (np.genfromtxt(f, delimiter=" ",dtype=np.int64))#"," or " "
        with open(target_rotation_label,'rb') as f:
            target_rotation_label = (np.genfromtxt(f, delimiter=",", dtype=np.float32))#"," or " "
            
        if self.dataset_ver == 'hand':
            return rgb_image, depth_image, meta_data, y_label, hand_image, target_rotation_label
        else:
            return rgb_image, depth_image, meta_data, y_label

    def __len__(self):

        return len(self.rgb_images_list)


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
    for i, (rgb_images, depth_images, meta_datas, y_labels, hand_images, target_rotation_label) in tqdm(enumerate(ponnet_loader)):
        y_labels_gt = (y_labels[:,target_label]).long()
        print(y_labels_gt)
        if y_labels_gt == 1:
            coll_All += 1
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
    
        #print("for_i",i)
        #print(target_rotation_label)
        #print(target_rotation_label.shape)
        #print(y_labels)
        #print(y_labels.shape)
        #att_out, y_label_out, _visual_att_map = model(rgb_images,depth_images,meta_datas, hand_images)
        #print('y_labels.shape',y_labels.shape)
        #y_labels_acc1 = (y_labels[:,0]).long()
        #print('y_labels_acc1.shape',y_labels_acc1.shape)
        #print('y_labels_acc1',y_labels_acc1)
        #print('y_label_out[0].shape',y_label_out[0].shape)
        #print('y_label_out[0]',y_label_out[0])
        #loss_y = loss(y_label_out[0], y_labels_acc1)
        #target_rotation_label = torch.eye(4)[target_rotation_label.long()].long()
        #target_rotation_label = torch.unsqueeze(target_rotation_label, 1).long()
        #print('target_rotation_label.shape',target_rotation_label.shape)
        #print('target_rotation_label',target_rotation_label.long())
        #print('y_label_out[1].shape',y_label_out[1].shape)
        #print('y_label_out[1]',y_label_out[1])
        #loss_rotation = loss(y_label_out[1], target_rotation_label.squeeze())
        #print('att_out[0].shape',att_out[0].shape)
        #print('att_out[2].shape',att_out[1].shape)
        #print('att_out[1].shape',att_out[2].shape)