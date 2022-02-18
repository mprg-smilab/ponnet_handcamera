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

class PonNetDataset(torch.utils.data.Dataset):
    
    def __init__(self,root="./N10k_release1",load_mode="train"):
        super().__init__()
        self.root = root
        self.load_mode = load_mode
        
        self.rgb_images_list = []
        self.depth_images_list = []
        self.meta_datas_list = []
        self.y_labels_list = []
        dataset_dir = '03preprocessed'
        if self.load_mode == "train":
            root_rgb_path = os.path.join(root,'train','RGB','N10k_train.txt')
            #print('root_rgb_path',root_rgb_path)
            root_depth_path = os.path.join(root,'train','depth','N10k_train.txt')
            root_meta_path = os.path.join(root,'train','meta_train.txt')
            root_label_path = os.path.join(root,'train','y_train.txt')
        if self.load_mode == "valid":
            root_rgb_path = os.path.join(root,'valid','RGB','N10k_valid.txt')
            root_depth_path = os.path.join(root,'valid','depth','N10k_valid.txt')
            root_meta_path = os.path.join(root,'valid','meta_valid.txt')
            root_label_path = os.path.join(root,'valid','y_valid.txt')
        if self.load_mode == "test":
            root_rgb_path = os.path.join(root,'test','RGB','N10k_test.txt')
            root_depth_path = os.path.join(root,'test','depth','N10k_test.txt')
            root_meta_path = os.path.join(root,'test','meta_test.txt')
            root_label_path = os.path.join(root,'test','y_test.txt')
        if self.load_mode == "test2":
            root="./kawai_release1"
            root_rgb_path = os.path.join(root,'test','RGB','RGB_files.txt')
            root_depth_path = os.path.join(root,'test','depth','depth_files.txt')
            root_meta_path = os.path.join(root,'test','meta_files.txt')
            root_label_path = os.path.join(root,'test','y_files.txt')
        if self.load_mode == "test3":
            root="./kawai_release2"
            dataset_dir = 'preprocessed'
            root_rgb_path = os.path.join(root,'test','RGB','RGB_files.txt')
            root_depth_path = os.path.join(root,'test','depth','depth_files.txt')
            root_meta_path = os.path.join(root,'test','meta_files.txt')
            root_label_path = os.path.join(root,'test','y_files.txt')


        with open(root_rgb_path) as f:
            rgb_images = f.readlines()
        with open(root_depth_path) as f:
            depth_images = f.readlines()
        with open(root_meta_path) as f:
            meta_datas = f.readlines()
        with open(root_label_path) as f:    
            y_labels = f.readlines()
        #print(rgb_images)
        
        for rgb_filename, depth_filename, meta_filename, y_label_filename in zip(rgb_images, depth_images, meta_datas, y_labels):
            self.rgb_images_list.append(os.path.join(root, dataset_dir, rgb_filename.strip()))
            self.depth_images_list.append(os.path.join(root,dataset_dir,depth_filename.strip()))
            self.meta_datas_list.append(os.path.join(root,dataset_dir,meta_filename.strip()))
            self.y_labels_list.append(os.path.join(root,dataset_dir,y_label_filename.strip()))
        #print('rgb_images_list',self.rgb_images_list)
    
    def __getitem__(self, index):
        rgb_image = self.rgb_images_list[index]
        depth_image = self.depth_images_list[index]
        meta_data = self.meta_datas_list[index]
        y_label = self.y_labels_list[index]
        with open(rgb_image,'rb') as f:
            rgb_image = Image.open(f).convert('RGB')
            #tensor ni henkan and normalization?
            rgb_image = TF.to_tensor(rgb_image)
        with open(depth_image,'rb') as f:
            depth_image = Image.open(f).convert('RGB')
            depth_image = TF.to_tensor(depth_image)
        with open(meta_data,'rb') as f:
            meta_data = (np.genfromtxt(f, delimiter=", ",dtype=np.float32))#"," or " "

        with open(y_label,'rb') as f:
            y_label = (np.genfromtxt(f, delimiter=",",dtype=np.float32))#"," or " "
            #y_label = (np.genfromtxt(f, delimiter=" ",dtype=np.int64))#"," or " "
        return rgb_image, depth_image, meta_data, y_label

    def __len__(self):

        return len(self.rgb_images_list)


if __name__ == "__main__":

    #---kakunin
    ponnet_dataset = PonNetDataset(train=True)
    ponnet_loader = torch.utils.data.DataLoader(dataset=ponnet_dataset,batch_size=5)

    def show(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

    for i, (rgb_images, depth_images, meta_datas, y_labels) in enumerate(ponnet_loader):
        print(rgb_images)
        print(depth_images)
        print(meta_datas)
        print(y_labels)
        break 