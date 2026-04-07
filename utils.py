import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np

import sys
import time
import os

import shutil


# ------------ Voters specific utils -----------------

def GetVoterValidation(batchSize):
    valData = torch.load("./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth", weights_only=False)
    valImages = valData["data"].float()
    valLabels = valData["binary_labels"].long()
    
    valDataset = TensorDataset(valImages, valLabels)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False)
    return valLoader

def GetVoterTraining(batchSize):
    # trainData = torch.load("./data/kaleel_final_dataset_train_Combined_Grayscale.pth", weights_only=False)
    trainData = torch.load("./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth", weights_only=False)
    trainImages = trainData["data"].float()
    trainLabels = trainData["binary_labels"].long()
    
    trainDataset = TensorDataset(trainImages, trainLabels)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
    return trainLoader


def GetVoterTrainingBalanced(batchSize, totalSamples, numClasses):
    # Get all training data (shuffled) with same batchSize
    fullTrainLoader = GetVoterTraining(batchSize=batchSize)
    
    # Collect all shuffled data from batches
    allImages = []
    allLabels = []
    for images, labels in fullTrainLoader:
        allImages.append(images)
        allLabels.append(labels)
    
    trainImages = torch.cat(allImages, dim=0)
    trainLabels = torch.cat(allLabels, dim=0)
    
    # Calculate samples per class
    samplesPerClass = totalSamples // numClasses
    
    # Get shape of images
    imgShape = trainImages[0].shape
    
    # Initialize tensors for balanced data
    balancedImages = torch.zeros(totalSamples, imgShape[0], imgShape[1], imgShape[2])
    balancedLabels = torch.zeros(totalSamples)
    
    # Track how many samples we've collected per class
    classCount = torch.zeros(numClasses)
    
    # Collect balanced samples
    currentIndex = 0
    for i in range(len(trainLabels)):
        label = int(trainLabels[i])
        
        if classCount[label] < samplesPerClass:
            balancedImages[currentIndex] = trainImages[i]
            balancedLabels[currentIndex] = label
            classCount[label] += 1
            currentIndex += 1
        
        if currentIndex >= totalSamples:
            break
    
    # Verify we got enough samples
    for c in range(numClasses):
        if classCount[c] != samplesPerClass:
            raise ValueError(f"Not enough samples for class {c}. Got {int(classCount[c])}, needed {samplesPerClass}")
    
    print(f"Balanced training data: {totalSamples} samples ({samplesPerClass} per class)")
    
    # Create dataloader
    balancedDataset = TensorDataset(balancedImages, balancedLabels.long())
    balancedLoader = DataLoader(balancedDataset, batch_size=batchSize, shuffle=False)
    
    return balancedLoader

# ------------------------------

def GetCIFAR10Training(batchSize=128, shuffle=True, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    trainLoader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batchSize, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True
    )
    return trainLoader


def GetCIFAR10Validation(batchSize=128, shuffle=False, num_workers=2):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform_test
    )
    valLoader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batchSize, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True
    )
    return valLoader

# Find the actual min and max pixel values in the dataset
def GetDataBounds(dataLoader, device):
    minVal = float('inf')
    maxVal = float('-inf')
    
    for xData, _ in dataLoader:
        xData = xData.to(device)
        batchMin = xData.min().item()
        batchMax = xData.max().item()
        
        if batchMin < minVal:
            minVal = batchMin
        if batchMax > maxVal:
            maxVal = batchMax
    
    print(f"Data range: [{minVal:.4f}, {maxVal:.4f}]")
    return minVal, maxVal

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData

#Replicate TF's predict method behavior
#Returns class labels predicted by models on a dataloader
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    return yPred

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0 
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Show 20 images, 10 in first and row and 10 in second row 
def ShowImages(xFirst, xSecond, save_path='Output.png'):
    #Convert from Pytorch tensor to HxWxColorChannel and Numpy 
    xFirstNumpy = xFirst.detach().numpy().transpose((0,2,3,1))
    xSecondNumpy = xSecond.detach().numpy().transpose((0,2,3,1))
    #xFirstNumpy = torch.transpose(xFirst,0,2,3,1).numpy()
    #xSecondNumpy = torch.transpose(xSecond,0,2,3,1).numpy()
    n = 10  # how many digits we will display
    plt.figure(figsize=(5, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(xFirstNumpy[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(xSecondNumpy[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    # plt.show()
    
    plt.savefig(save_path)  # Save the image instead of displaying it
    plt.close()  # Close the plot to free resources
    print(f"Images saved to {save_path}")

######## CLI visualization functions #####################

# Get terminal width with fallback
try:
    term_width = shutil.get_terminal_size().columns
except:
    term_width = 80  # Default fallback

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f