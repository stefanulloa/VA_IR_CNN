from __future__ import print_function
from __future__ import division

'''
Modified on Dec 09, 2018 from original by deckyal

@author: gary
'''
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#Fine tuning and feature extracton

import torch
import torch.nn as nn
import torch.optim as optim


import cv2

from math import sqrt
import re
import numpy as np

import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#import file_walker
from utils import *
#from config import *
from VAFacialDataset import ImageDatasets #,VideoDataset
from PIL import Image, ImageDraw


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, num_epochs = 25, is_inception = False):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_loss = 99999

    for epoch in range(num_epochs) :
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)

        running_loss = 0
        running_corrects = 0

        #iterate over data

        #for rinputs, rlabels in dataloaders :
        for x,(rinputs, rlabels) in enumerate(dataloaders,0) :

            model.train()

            inputs = rinputs.to(device)
            labels = rlabels.to(device)

            #zero the parameter gradients
            optimizer.zero_grad()

            #Forward
            #Track history if only in train
            #print(inputs.shape)
            with torch.set_grad_enabled(True) :
                if is_inception :
                    outputs, aux_outputs  = model(inputs)
                    loss1 = criterion(outputs,labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1+.4 * loss2
                else :
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                #_,preds = torch.max(outputs,1)

                loss.backward()
                optimizer.step()

            #statistics
            running_loss += loss.item() * inputs.size(0)
            print("{}/{} loss : {}".format(x,int(len(dataloader.dataset)/batch_size),loss.item()))

            #print(rinputs.shape, outputs.shape, rlabels.shape)


            #running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('Loss : {:.4f}'.format(epoch_loss))

        #Deep copy the model
        if epoch_loss < lowest_loss :
            lowest_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print(outputs[0],labels[0])
            #torch.save(model.state_dict(),'netFL.pt')

        '''
        #GaryComment
        rinputs will have all the transformations declared in getitem, so unnormalizedandlandmark will have to reverse them
        outputs indicates landmark coordinates prediction
        rlabels indicates actual landmark coordinates (ground_truth)
        these 2 will be added to each image, so that plotimages can show the new image
        '''

    torch.save(best_model_wts,'netFL.pt')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(lowest_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting :
        for param in model.parameters() :
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained = True):
    model_ft = None
    input_size = 0

    if model_name == 'resnet' :
        model_ft = models.resnet152(pretrained = use_pretrained)

        '''
        !!!
        #GaryComment
        in case, feature_extract is true, set_parameter_requires_grad will set all grad parameters to false
        and because after this operation, a new layer (the output one) is added, these parameters will have grad as true
        '''

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'alexnet' :
        model_ft = models.alexnet(pretrained = use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == 'squeezenet' :
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        model_ft.classifier[1] = nn.Conv2d(512,num_classes, kernel_size=(1,1), stride = (1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == 'densenet' :

        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

        input_size = 224

    elif model_name == 'inception' :
        #inception v3
        model_ft = models.inception_v3(pretrained = use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        #Aux net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

        #Primary Net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 299

    else :
        print('invalid model name')
        exit()

    return model_ft, input_size

def train():
    #Model to chosse from [resnet, alexnet, vgg, squeezenet, densenet,inception]
    model_name = 'inception'

    #Number of classes
    num_classes = 2

    #Batch size
    num_epochs = 1000

    feature_extract = False
    #Intialize the model for this run
    model_ft , image_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


    # Data augmentation and normalization for training
    # Just normalization for validation

    '''
    GaryComment
    toTensor transforms PIL image numpy array HxWxC [0,255] to tensor CxHxW [0,1]
    HxWxC: RGB, RGB, ..., RGB    CxHxW: RR...R,GG...G,BB...B   C is

    '''

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    batch_size = 8

    ID = ImageDatasets(data_list = ['afew-Train'],transform=data_transforms['train'])
    #VD = VideoDataset(data_list = ['toTest_M'],blurLevel=20,onlyFace = True, seq_length = 64, transform=transform)
    print(len(ID))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, shuffle = True)
    #dataloader = torch.utils.data.DataLoader(dataset = VD, batch_size = batch_size, shuffle = True)


    n_to_view = 4
    #to opencv format



    '''print("Initializing Datasets and Dataloaders...")

    #Training and validation datasets.
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train','val']}

    #Now the dataloader
    dataloaders_dict = {x:torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True, num_workers = 4) for x in ['train','val']}
    '''

    #model_ft.load_state_dict(torch.load('./netFL.pt', map_location=lambda storage, loc: storage))

    '''
    !!!
    GaryComment
    in case feature_extract is true, only output layer parameters will have grad as true, so these are the only needed in params_to_update
    '''
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    print("Params to learn ")

    if feature_extract :
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True :
                params_to_update.append(param)
                print("\t",name)
    else :
        for name,param in model_ft.named_parameters() :
            if param.requires_grad == True :
                print("\t",name)

    optimizer_ft = optim.SGD(params_to_update,lr=.01, momentum = .9)


    #loss
    criterion = nn.MSELoss()

    #Train and evaluate
    model_ft, hist = train_model(model_ft, dataloader, criterion, optimizer_ft, num_epochs, is_inception = (model_name == "inception"))

def test():
    model_name = 'inception'

    #Number of classes
    num_classes = 2

    feature_extract = False
    #Intialize the model for this run
    model_ft , image_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)


    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size,image_size)),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    model_ft.load_state_dict(torch.load('./netFL.pt', map_location=lambda storage, location:storage))
    model_ft.to(device)
    model_ft.eval()


    listImage = ['\\testImages\\fddb__image2665_0.jpg','\\testImages\\fddb__image2666_0.jpg','\\testImages\\fddb__image2667_0.jpg','\\testImages\\fddb__image2668_0.jpg']

    tl = []

    for x in listImage :

        tImageB = Image.open(x)
        tImageB = data_transforms['val'](tImageB)

        tl.append(tImageB.unsqueeze(0))

    li = torch.Tensor(len(tl),3,image_size,image_size) #forcing 3 channel, no need to convert to rgb

    torch.cat(tl, out=li)

    li = li.to(device)


    print(li,li.size())

    output  = model_ft.forward(li)

    print(output)

    output = output.detach().cpu()
    #img = output.cpu()[0]
    img = li.cpu()

train()
