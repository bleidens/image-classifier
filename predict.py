import numpy as np
from PIL import Image
import copy
import json
import os
import random
import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/3/image_06641.jpg') 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):    
    open_image = Image.open(image)    
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])    
    image = transformation(open_image)    
    return image

def predict(image_path, model, topk=3, gpu='gpu'):
    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()        
    image_torch = process_image(image_path)
    image_torch = image_torch.unsqueeze_(0)
    image_torch = image_torch.float()
    if gpu == 'gpu':
        with torch.no_grad():
            output = model.forward(image_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(image_torch)      
    probability = F.softmax(output.data,dim=1) 
    probs = np.array(probability.topk(topk)[0][0]) 
    index_to_class = {val: key for key, val in model.class_to_idx.items()} 
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]  
    return probs, top_classes

def main(): 
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    gpu = args.gpu    
    image_path = args.filepath
    probs, classes = predict(image_path, model, int(args.top_k), gpu)
    labels = [cat_to_name[str(index)] for index in classes]
    probability = probs  
    i = 0 
    while i < len(labels):
        print("{} with probability of {}".format(labels[i], probability[i]))
        i += 1 

if __name__ == "__main__":
    main()