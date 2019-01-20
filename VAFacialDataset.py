
import cv2
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
import re
import os
import os.path as path

from PIL import Image,ImageFilter

import torch
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from skimage.util import random_noise

#import file_walker
from utils import *
#from config import *

#when sending jobs to cluster, set runserver to True so that remote dir is set
curDir = os.getcwd()

class ImageDatasets(data.Dataset):

    def __init__(self, data_list = ["afew-Train"],dir_gt = 'annot2',step = 1, transform = None, image_size =224):

        self.transform = transform

        self.imageHeight = image_size
        self.imageWidth = image_size

        list_img_paths = []
        list_labels_t = []

        counter_image = 0
        annot_name = 'annot'

        if dir_gt is not None :
            annot_name = dir_gt

        'Whole directory tree: afew-Train -> 01, 02, ... -> annot, annot 2, img -> data (.png, .aro)'
        'This for is to get images and labels directory paths'
        for data in data_list :  #iterates over all the list passed as first argument (0th level): afew-Train'
            print(("Opening "+data))
            collections_dir = path.join(path.join(curDir, 'images'),data)
            for col_f in os.listdir(collections_dir): #walks into 1st level directories'
                col_f_dir = path.join(collections_dir,col_f)
                #print(col_f_dir)
                for type_f in os.listdir(col_f_dir): #walks into 2nd level'
                    #print(col_f_dir, type_f)
                    list_dta = [] #list for annots or img directories, emptied each iteration

                    'Add annot2 .aro (labels) and img .png (images) paths, ignore the rest'
                    type_f_dir = path.join(col_f_dir, type_f)

                    if(type_f == annot_name): #If it's annot, add to labels_t
                        for data_f in os.listdir(type_f_dir): #this is the data
                            if(data_f.endswith('.aro')):
                                full_filePath = path.join(type_f_dir,data_f)
                                list_dta.append(full_filePath)
                        list_labels_t.append(sorted(list_dta)) #this will be used for ground_truth
                    elif(type_f == 'img'):
                        for data_f in os.listdir(type_f_dir): #this is the data
                            if(data_f.endswith('.png')):
                                full_filePath = path.join(type_f_dir,data_f)
                                list_dta.append(full_filePath)
                        list_img_paths.append(sorted(list_dta)) #this variable gt DOES NOT mean ground_truth
                        counter_image+=len(list_dta)

        self.length = counter_image
        print("Now opening keylabels")


        'List tree: All -> 01, 02, ... collections -> data paths'
        'This is to get V-A labels in a list of lists'
        list_labels = []
        for lbl in list_labels_t : #for the labels collections paths 01, 02, ...
            lbl_va = []
            for lbl_sub in lbl : #for the actual data paths
                if ('aro' in lbl_sub) : #in this case, this if will not change anything
                    x = []
                    with open(lbl_sub) as file:
                        'string calls strip, which if not argument given, takes out the ending \n (it takes out whitespaces, \n, ... from start and end)'
                        vaData = [l.strip() for l in file] #vaData has a list containing the one line content
                    x = [ float(j) for j in vaData[0].split(' ')] #x is list of lists of V-A [v a], [0] gives content with not list brackets
                    lbl_va.append(np.array(x)) #one list for all V-A from the same collection

            list_labels.append(lbl_va) #list_labels is a list of list of V-A collections


        list_images = []
        list_ground_truth = np.zeros([counter_image,2]) #each label has a V-A value
        #the ground_truth WILL HAVE all the V-A labels, from all collections

        'storing images paths and V-A labels in dataset attributes'
        indexer = 0
        for i in range(0,len(list_img_paths)): #For each dataset collection
            for j in range(0,len(list_img_paths[i]),step): #for number of data #step is usefull for video data
                list_images.append(list_img_paths[i][j]) #list of lists of images paths

                #print(len(list_img_paths),len(list_img_paths[i]),'-',len(list_labels),len(list_labels[i]))
                #print(counter_image,indexer)

                list_ground_truth[indexer] = np.array(list_labels[i][j]) #actual store of labels in ground_truth
                indexer += 1

        self.l_imgs = list_images #images path TOGETGER
        self.l_gt = list_ground_truth #V-A labels TOGETHER

        #uncomment to get a log (do just once)
        #log file containing images paths V-A values to check according to index in getitem (because collections are unnordered)
        '''with open(curDir+'DSlog.txt','w') as f:
            for i in range(0,len(self.l_imgs)) :
                f.write(str(i) + ' ' + self.l_imgs[i] + ' ' + str(self.l_gt[i]) + '\n')
            f.write(str(len(self.l_imgs)))'''

        #x,label  = self.l_imgs[1],self.l_gt[1].copy()

    'when iterating over dataloaders, getitem is called'
    def __getitem__(self,index):

        #Read all data, transform etc.
        #In image the output will be : [batch_size, channel, width, height]

        x,label  = self.l_imgs[index],self.l_gt[index].copy() #(shallow) copy because otherwise, modifying one list, would modify the other one

        tImage = Image.open(x).convert("RGB") #from png to RGB
        'format is matrix where [x,y] = (r,g,b) from 0 to 255'
        #print(np.asarray(tImage)[120][140])

        if self.transform is not None:
            'includes (pil) image to tensor in transform'
            tImage = self.transform(tImage)

        return tImage,torch.FloatTensor(label) #returns [3,width,height] torch for pil image and [2] tensor for V-A label


    def __len__(self):

        return len(self.l_imgs)

#a = ImageDatasets()
#a.__getitem__(24643)
