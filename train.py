#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
#Adapted from hpo.py, added hooks from train_model.py, project3

from IPython.core.debugger import set_trace

import numpy as np
import torchvision
import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from PIL import ImageFile

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error

import smdebug.pytorch as smd

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device, file_out, hook):    
    
    logger.info("Testing Model on Whole Testing Dataset")
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss=0
    running_corrects=0
    
    if file_out==True:
        f1 = open('counting_result.txt','w')
        f2 = open('counting_label.txt','w')
       
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        if file_out==True:
            write_to_files(f1, f2, preds, labels.data, device)
       
    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects/ len(test_loader)
     
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    
    calculate_rmse(labels.data, preds, device)

    if file_out==True:
        f1.close()
        f2.close() 
        logger.info('Saving counting_result.txt & counting_label.txt')
      
                                                                            
def train(model, train_loader, validation_loader, criterion, optimizer, device, hook):
    
    epochs = 10   #30 
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
                
            running_loss = 0.0
            running_corrects = 0
          
            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                                   
            epoch_loss = running_loss / len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
                               
            logger.info('Accuracy: {:.2f}, Loss: {:.2f}, Best loss {:.2f}'.format(epoch_acc, epoch_loss, best_loss))
            
            calculate_rmse(labels.data, preds, device)
            
        if loss_counter==1:
            break
       
    return model


# Notes: Used resnet50 because it has the best accuracy and speed, and a small model size [https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/ , Accessed 11/12/21]
    

def net():
    
    num_classes = 5   #number of counting classes 
    
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, num_classes))
    return model
    
    

def create_data_loaders(data, batch_size):
        
    train_transform = transforms.Compose([
    #transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
    #transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset = datasets.ImageFolder(data, transform=train_transform)     #dataset
    #total = 10441
    #lengths = [6265, 2088, 2088]  #60, 20, 20  
    
    total = len(dataset)
    train_length = int(np.ceil(.6*total))
    test_length = int(np.floor(.2*total))
    lengths = [train_length, test_length, test_length]
    
    trainset, testset , validset = torch.utils.data.random_split(dataset, lengths) 
                 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
 
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
  
    return trainloader, testloader, validloader




def calculate_rmse(ground_truth, prediction, device):
    
    if device != 'cpu':
        ground_truth = ground_truth.cpu() 
        prediction = prediction.cpu()
      
    rmse = np.sqrt(mean_squared_error(ground_truth, prediction))                #sklearn < 0.22.0
    logger.info(f"RMSE: {rmse}")


def write_to_files(file1, file2, prediction, labels_data, device):
    
    if device != 'cpu':
        labels_data = labels_data.cpu().numpy() 
        prediction = prediction.cpu().numpy()
    
    for j in range(len(prediction)):
        file1.write('%d\n' % prediction[j])
        file2.write('%d\n' % labels_data[j])
    
        
    
def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
  
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data_dir}')
     
    logger.info("Initializing the model.")
    model=net()
    model=model.to(device)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)  
    
    hook.register_loss(loss_criterion)
    
    logger.info("Loading data")
    train_loader, test_loader, valid_loader = create_data_loaders(args.data_dir, args.batch_size) 
    
    logger.info("Training the model.")
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, device, hook)
   
    logger.info("Testing the model.")
    test(model, test_loader, loss_criterion, device, args.evaluate, hook)
       
    logger.info("Saving the model.")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth")) 

    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
      
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,    
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    
    parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR']) 
        
    args=parser.parse_args()   
    
    main(args)
