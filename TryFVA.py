from __future__ import print_function
from __future__ import division

'''
Modified on Dec, 2018 from original by deckyal

@author: stefano
'''
#https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#Fine tuning and feature extracton

import torch
import torch.nn as nn
import torch.optim as optim

import cv2
import argparse

from math import sqrt
import re
import numpy as np

import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils

from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import time
import os
import copy
import math
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#import file_walker
from utils import *
#from config import *
from VAFacialDataset import ImageDatasets
from PIL import Image, ImageDraw

#when sending jobs to cluster, set runserver to True so that remote dir is set
curDir = os.getcwd()+'/'

# Detect if GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
#[a-alexnet, d-densenet, i-inception, r-resnet, s-squeezenet, v-vgg]
parser.add_argument('-m', '--model', nargs='?', const='a', type=str, default='r')
parser.add_argument('-b', '--batchsize', nargs='?', const='8', type=int, default='58')
parser.add_argument('-cg', '--clipgradnorm', nargs='?', const=0, type=int, default='0') #[0-false, 1-true]
args = parser.parse_args()

def RMSELoss(ypred, yact):
    return torch.sqrt(torch.mean((ypred-yact)**2))

def train_model(model, dataloaders, datasetSize, criterion, optimizer, model_name, batch_size, CVpositionPart, num_epochs = 25, is_inception = False):
    since = time.time()
    val_acc_history = []
    idDiff = ""
    messageDiff = ""
    only01 = True
    only001 = True

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
                '''
                print(outputs[0], '___', labels[0])
                print('-'*5)
                print(outputs, '___', labels)
                '''
                #_,preds = torch.max(outputs,1)
                loss.backward()
                #in case of possible gradient explotion (nan loss values), call:
                if args.clipgradnorm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()

            #statistics
            running_loss += loss.item() * inputs.size(0)
            print("{}/{} loss : {}".format(x,int(datasetSize/batch_size),loss.item()))
            #running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / datasetSize
        #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('Loss : {:.4f}'.format(epoch_loss))
        print('Loss : {:.4f}'.format(lowest_loss))

        "this if prevents saving when loss is nan"
        #Deep copy the model
        if epoch_loss < lowest_loss :
            loss_diff = lowest_loss-epoch_loss
            if(loss_diff < 0.01 and loss_diff >= 0.001 and only01):
                idDiff = "01"
                messageDiff = "Model reached 0.01 tolerance     "
                only01 = False
            elif(loss_diff < 0.001 and only001):
                idDiff = "001"
                messageDiff = "Model reached 0.001 tolerance    "
                only001 = False
            lowest_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            now = time.time()
            print(outputs[0],labels[0])
            if(idDiff == "01" or idDiff == "001"):
                torch.save(model.state_dict(),curDir+model_name+'CV'+str(CVpositionPart)+'_'+idDiff+'netFVA.pt') #it has to be here so that for a big nums_epochs we still can retrieve the best model (minimum loss) without waiting for all epochs to occur
            "write log to know if current best model is good enough because cluster will not give output until terminating all epochs"
            with open(curDir+model_name+'CV'+str(CVpositionPart)+'VAlog.txt','a') as file:
                file.write(messageDiff + 'epoch: ' + str(epoch) + '   epoch_loss: ' + str(epoch_loss) + '    time: ' + str(now) + '\n-----\n')
                file.write('outputs: ' + str(outputs.tolist()[0][0]) + ' ' + str(outputs.tolist()[0][1]) + '     labels: ' + str(labels.tolist()[0][0]) + ' ' + str(labels.tolist()[0][1]))
                file.write('\n-----\n')
            idDiff = ""
            messageDiff = ""
        if(not only001):
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(lowest_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, val_acc_history

def evaluate_model(model, dataloaders, datasetSize, model_name, batch_size, CVpositionPart, is_inception = False):

    since = time.time()
    val_acc_history = []
    predValenceSum = 0
    labelValenceSum = 0
    predArousalSum = 0
    labelArousalSum = 0

    model.eval()

    print('computing mean values')
    #first compute the mean of he different variables :
    for x,(rinputs, rlabels) in enumerate(dataloaders,0) :

        inputs = rinputs.to(device)
        labels = rlabels.to(device)

        with torch.set_grad_enabled(False) :
            if is_inception :
                outputs, aux_outputs  = model(inputs)
            else :
                outputs = model(inputs)

        #tensor structure: [arousal, valence]
        predArousalSum += outputs[:,0].sum().item()
        labelArousalSum += labels[:,0].sum().item()

        predValenceSum += outputs[:,1].sum().item()
        labelValenceSum += labels[:,1].sum().item()

        print("COMPUTING {}/{} MPA : {}, MLA : {}, MPV : {}, MLV : {}".format(x,int(datasetSize/batch_size),predArousalSum, labelArousalSum, predValenceSum, labelArousalSum))

    meanPredArousal = predArousalSum / datasetSize
    meanLabelArousal = labelArousalSum / datasetSize
    meanPredValence = predValenceSum / datasetSize
    meanLabelValence = labelArousalSum / datasetSize

    print("FINAL MPA : {}, MLA : {}, MPV : {}, MLV : {}".format(meanPredArousal, meanLabelArousal, meanPredValence, meanLabelValence))

    arousalSquaredDifferenceSum = 0
    valenceSquaredDifferenceSum = 0

    arousalCOVSumPart = 0
    valenceCOVSumPart = 0
    arousalCOV = 0
    valenceCOV = 0

    stdPredArousalSquaredDifferenceSumPart = 0
    stdLabelArousalSquaredDifferenceSumPart = 0
    stdPredValenceSquaredDifferenceSumPart = 0
    stdLabelValenceSquaredDifferenceSumPart = 0
    stdPredArousal = 0
    stdLabelArousal = 0
    stdPredValence = 0
    stdLabelValence = 0
    #now compute the metrics that depend on mean :
    for x,(rinputs, rlabels) in enumerate(dataloaders,0) :

        #tensor structure: [arousal, valence]
        inputs = rinputs.to(device)
        labels = rlabels.to(device)

        with torch.set_grad_enabled(False) :
            if is_inception :
                outputs, aux_outputs  = model(inputs)
            else :
                outputs = model(inputs)

        arousalSquaredDifferenceSum += ((outputs[:,0]-labels[:,0])**2).sum().item()
        valenceSquaredDifferenceSum += ((outputs[:,1]-labels[:,1])**2).sum().item()

        arousalCOVSumPart += ((outputs[:,0]-meanPredArousal) * (labels[:,0]-meanLabelArousal)).sum().item()
        valenceCOVSumPart += ((outputs[:,1]-meanPredValence) * (labels[:,1]-meanLabelValence)).sum().item()

        stdPredArousalSquaredDifferenceSumPart += ((outputs[:,0]-meanPredArousal)**2).sum().item()
        stdLabelArousalSquaredDifferenceSumPart += ((labels[:,0]-meanLabelArousal)**2).sum().item()
        stdPredValenceSquaredDifferenceSumPart += ((outputs[:,1]-meanPredValence)**2).sum().item()
        stdLabelValenceSquaredDifferenceSumPart += ((labels[:,1]-meanLabelValence)**2).sum().item()

        print("COMPUTING {}/{} ASDS : {}, VSDS : {}, ACSP : {}, VCSP : {}".format(x,int(datasetSize/batch_size),arousalSquaredDifferenceSum, valenceSquaredDifferenceSum, arousalCOVSumPart, valenceCOVSumPart))
        '''print(labels)
        print(outputs)
        print(labels[:,0])
        print(labels[:,1])
        print(labels.size())
        print(labels[:,0].sum().item())
        print(labels[0])
        print(labels[0][0])
        print(labels[0][0].item())
        print(arousalSquaredDifferenceSum)
        print(valenceSquaredDifferenceSum)'''

    print("Dataset size: : {}".format(datasetSize))

    print("FINAL MPA : {}, MLA : {}, MPV : {}, MLV : {}".format(meanPredArousal, meanLabelArousal, meanPredValence, meanLabelValence))

    RMSEArousal = math.sqrt(arousalSquaredDifferenceSum/datasetSize)
    RMSEValence = math.sqrt(valenceSquaredDifferenceSum/datasetSize)


    arousalCOV = arousalCOVSumPart / (datasetSize-1)
    valenceCOV = valenceCOVSumPart / (datasetSize-1)

    print('arousalCOV:', arousalCOV)
    print('valenceCOV:', valenceCOV)

    stdPredArousal = math.sqrt(stdPredArousalSquaredDifferenceSumPart/datasetSize)
    stdLabelArousal = math.sqrt(stdLabelArousalSquaredDifferenceSumPart/datasetSize)
    stdPredValence = math.sqrt(stdPredValenceSquaredDifferenceSumPart/datasetSize)
    stdLabelValence = math.sqrt(stdLabelValenceSquaredDifferenceSumPart/datasetSize)

    CORArousal = arousalCOV / (stdPredArousal * stdLabelArousal)
    CORValence = valenceCOV / (stdPredValence * stdLabelValence)

    ICCArousal = 2*arousalCOV/((stdPredArousal**2)+(stdLabelArousal**2))
    ICCValence = 2*valenceCOV/((stdPredValence**2)+(stdLabelValence**2))

    print("FINAL RMSEA : {}, RMSEV : {}, CORA : {}, CORV : {}, ICCA : {}, ICCV : {}".format(RMSEArousal, RMSEValence, CORArousal, CORValence, ICCArousal, ICCValence))

    '''#statistics
    running_loss += loss.item() * inputs.size(0)
    print("{}/{} loss : {}".format(x,int(datasetSize/batch_size),loss.item()))'''
    #running_corrects += torch.sum(preds == labels.data)

    #epoch_loss = running_loss / datasetSize
    #epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)


    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting :
        for param in model.parameters() :
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained = True):
    model_ft = None
    input_size = 0

    if model_name == 'resnet' :
        model_ft = models.resnet152(pretrained = use_pretrained) #50 or 152

        '''
        in case, feature_extract is true, set_parameter_requires_grad will set all grad parameters to false
        and because after this operation a new layer (the output one) is added, these new parameters will have grad as true
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

    elif model_name == 'vgg':
        model_ft = models.vgg11_bn(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.classifier[6].in_features

        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
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

    print('Starting net training mode', os.getcwd())

    #Model to chosse from [resnet, alexnet, vgg, squeezenet, densenet,inception]
    model_name = args.model
    if model_name=='a': model_name='alexnet'
    elif model_name=='d': model_name='densenet'
    elif model_name=='i': model_name='inception'
    elif model_name=='r': model_name='resnet'
    elif model_name=='s': model_name='squeezenet'
    elif model_name=='v': model_name='vgg'
    else: model_name=='invalid'

    #Number of classes
    num_classes = 2

    #Batch size
    num_epochs = 1000

    feature_extract = False
    #Intialize the model for this run

    type224Models = ['alexnet','densenet','resnet','squeezenet','vgg']

    if model_name in type224Models: image_size = 224
    elif model_name == 'inception': image_size = 299

    # Data augmentation and normalization for training
    # Just normalization for validation

    "toTensor transforms PIL image numpy array HxWxC [0,255] to tensor CxHxW [0,1] HxWxC: RGB, RGB, ..., RGB    CxHxW: RR...R,GG...G,BB...B "

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

    batch_size = args.batchsize

    print('pretrained model: ' + model_name + '\tbatch size: ' + str(batch_size) + '\n')

    print('Facial dataset')
    ID = ImageDatasets(data_list = ['afew-Train'],transform=data_transforms['train'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    'Cross validation section'
    datasetLen = len(ID)
    indices = list(range(datasetLen))
    k = 5 #for this cross validation, k = 5
    split = math.floor(datasetLen/k)+1 #+1, otherwise there would be an extra list of just one sample
    #data is divided in k (roughly) equal parts
    kPartsIndices = [indices[z:z+split] for z in range(0,len(indices),split)]

    trainIndLists=[]
    testIndLists=[]

    for i in range(0, len(kPartsIndices)):
        curTrainIndList=[]

        #list of lists of indexes from the rest of parts
        rest = kPartsIndices[:i] + kPartsIndices[i+1:]

        [curTrainIndList.extend(rest[x]) for x in range(0,len(rest))] #extend and the loop to get all rest lists elements in one list
        #list of lists of the rest parts
        trainIndLists.append(curTrainIndList)

        #list of lists of the individual parts
        testIndLists.append(kPartsIndices[i])



    for x in range(0, 5):

        model_ft , image_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

        datasetSize = len(trainIndLists[x]) #cannot use dataloader.datasets because it outputs all data, not just sampler ones
        trainsampler = SubsetRandomSampler(trainIndLists[x])

        dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, sampler=trainsampler, shuffle = False) #when supplying a sampler, shuffle has to be false, SubsetRandomSampler takes care of shuffling at each iteration


        #model_ft.load_state_dict(torch.load('./netFL.pt', map_location=lambda storage, loc: storage))

        "in case feature_extract is true, only output layer parameters will have grad as true, so these are the only needed in params_to_update"
        model_ft = model_ft.to(device)
        params_to_update = model_ft.parameters()
        #print("Params to learn ")

        if feature_extract :
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True :
                    params_to_update.append(param)
                    #print("\t",name)
        else :
            for name,param in model_ft.named_parameters() :
                if param.requires_grad == True :
                    pass
                    #print("\t",name)

        optimizer_ft = optim.SGD(params_to_update,lr=.01, momentum = .9)


        #loss
        criterion = nn.MSELoss()

        #Train and evaluate
        model_ft, hist = train_model(model_ft, dataloader, datasetSize, criterion, optimizer_ft, model_name, batch_size, x, num_epochs, is_inception = (model_name == "inception"))

def test():

    print('Starting net testing mode', os.getcwd())

    model_name = 'resnet'
    image_size = 224
    #Number of classes
    num_classes = 2
    #Intialize the model for this run
    batch_size = 1

    feature_extract = False

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


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    print('pretrained model: ' + model_name + '\tbatch size: ' + str(batch_size) + '\n')

    print('Facial dataset')
    ID = ImageDatasets(data_list = ['afew-Train'],transform=data_transforms['val'])


    'Cross validation section'
    datasetLen = len(ID)
    indices = list(range(datasetLen))
    k = 5 #for this cross validation, k = 5
    split = math.floor(datasetLen/k)+1 #+1, otherwise there would be an extra list of just one sample
    #data is divided in k (roughly) equal parts
    kPartsIndices = [indices[z:z+split] for z in range(0,len(indices),split)]

    testIndLists=[]

    for i in range(0, len(kPartsIndices)):
        #list of lists of the individual parts
        testIndLists.append(kPartsIndices[i])

    for x in range(0, 5): #len(testIndLists)):

        model_ft , image_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

        model_ft.load_state_dict(torch.load('./resnet_CV5_clusterTraining/resnetCV'+str(x)+'_001netFVA.pt'))
        #model_ft.to(device)
        #model_ft.eval()

        datasetSize = len(testIndLists[x]) #cannot use dataloader.datasets because it outputs all data, not just sampler ones
        testsampler = SubsetRandomSampler(testIndLists[x])

        dataloader = torch.utils.data.DataLoader(dataset = ID, batch_size = batch_size, sampler=testsampler, shuffle = False) #when supplying a sampler, shuffle has to be false, SubsetRandomSampler takes care of shuffling at each iteration

        model_ft = model_ft.to(device)

        #evaluate
        hist = evaluate_model(model_ft, dataloader, datasetSize, model_name, batch_size, x, is_inception = (model_name == "inception"))


def testingInstances():

    model_name = "resnet"
    num_classes = 2
    feature_extract = False

    model_ft , image_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

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

    model_ft.load_state_dict(torch.load('./resnet_CV5_clusterTraining/resnetCV1_01netFVA.pt'))
    model_ft.to(device)
    model_ft.eval()

    imagesPath = curDir+'images/afew-Train/02/' #02
    number = "02_051_00000" #"02_051_00000"
    number2 = "02_052_00000" #"02_052_00000"
    listImage = [imagesPath+ 'img/'+number+'.png',imagesPath +'img/'+number2+'.png']
    listLabels = [imagesPath+ 'annot2/'+number+'.aro',imagesPath +'annot2/'+number2+'.aro']

    tl = []

    for x in listImage :

        tImageB = Image.open(x)
        tImageB = data_transforms['val'](tImageB)

        tl.append(tImageB.unsqueeze(0))

    li = torch.Tensor(len(tl),3,image_size,image_size) #forcing 3 channel, no need to convert to rgb

    torch.cat(tl, out=li)

    li = li.to(device)


    #print(li,li.size())

    output  = model_ft.forward(li)

    print(output)

    output = output.detach().cpu()
    #img = output.cpu()[0]
    img = li.cpu()

    for x in listLabels:
        with open(x) as f:
        	line = f.readline()
        print("label: ", line)



testingInstances()
