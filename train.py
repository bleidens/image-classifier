import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import argparse
from PIL import Image
from collections import OrderedDict
import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import json

def save_checkpoint(path, model, optimizer, args, classifier):    
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier' : classifier,
                  'epochs': args.epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, path)    

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', help="directory with train and validation data", action='store')
    parser.add_argument('--arch', dest='arch', help="pre-trained model that will be used", default='densenet121',           choices=['densenet121', 'vgg13'])
    parser.add_argument('--learning_rate', dest='learning_rate', help="float type with learning rate for training",         default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', help="hidden layers for training", default='512')
    parser.add_argument('--epochs', dest='epochs', help="number of epochs for training, default is 3", default='3')
    parser.add_argument('--gpu',  help="include to train on the GPU via CUDA", action='store', default='gpu')
    parser.add_argument('--save_dir', dest="save_dir", help="directory where model will be saved after training",           action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    steps = 0
    print_every = 10  
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(dataloaders[0]): 
            steps += 1 
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to('cuda'), labels.to('cuda') 
            else:
                model.cpu() 
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0
                for ii, (inputs_second, labels_second) in enumerate(dataloaders[1]):  
                        optimizer.zero_grad()                      
                        if gpu == 'gpu':
                            inputs_second, labels_second = inputs_second.to('cuda') , labels_second.to('cuda') 
                            model.to('cuda:0') 
                        else:
                            model.cpu() 
                        with torch.no_grad():    
                            outputs = model.forward(inputs_second)
                            val_loss = criterion(outputs,labels_second)
                            ps = torch.exp(outputs).data
                            equality = (labels_second.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()
                val_loss = val_loss / len(dataloaders[1])
                accuracy = accuracy / len(dataloaders[1])
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss / print_every),
                      "Validation Loss {:.4f}".format(val_loss),
                      "Accuracy: {:.4f}".format(accuracy),
                     )
                running_loss = 0
            
def main():
    print("Training is starting...") 
    args = parse_args()
    
    data_dir = 'flowers'
    training_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'
    testing_dir = data_dir + '/test'
    
    training_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(), 
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(256), 
                                                transforms.CenterCrop(224), 
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256), 
                                             transforms.CenterCrop(224), 
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])]) 

    images_data = [ImageFolder(training_dir, transform=training_transforms),
                      ImageFolder(validation_dir, transform=validation_transforms),
                      ImageFolder(testing_dir, transform=testing_transforms)]
    
    dataloaders = [torch.utils.data.DataLoader(images_data[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(images_data[1], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(images_data[2], batch_size=64, shuffle=True)]
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False    
    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    gpu = args.gpu 
    
    class_index = images_data[0].class_to_idx
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    path = args.save_dir 
    
    save_checkpoint(path, model, optimizer, args, classifier)
    print("Done") 
    
if __name__ == "__main__":
    main()