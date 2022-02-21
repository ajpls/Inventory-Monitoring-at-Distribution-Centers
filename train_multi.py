#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
#Adapted from hpo.py, train_multi_sage
#multi-instance training with gpu

#Used https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
#https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51

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

import torch.multiprocessing as mp
import torchvision

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, gpu_id):    
    
    logger.info("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(gpu_id)
        labels=labels.to(gpu_id)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects/ len(test_loader)
     
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    
    calculate_rmse(labels.data, preds, gpu_id)

           
def train(model, train_loader, validation_loader, criterion, optimizer, gpu_id, args):
                   
    epochs = 3   #30 local train
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
                image_dataset[phase].sampler.set_epoch(epoch)
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0            
          
            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(gpu_id)
                labels=labels.to(gpu_id)
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
            
            calculate_rmse(labels.data, preds, gpu_id)
         
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
   
    

def create_data_loaders(data, batch_size, rank, world_size):
    
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
     
    train_set, test_set , valid_set = torch.utils.data.random_split(dataset, lengths)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    num_replicas=world_size,
                                                                    rank=rank, 
                                                                    shuffle=False, 
                                                                    )

    train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                               batch_size=batch_size,
                                               shuffle=False,            
                                               num_workers=0,
                                               pin_memory=False,
                                               sampler=train_sampler, 
                                               )     
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
 
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
  
    return train_loader, test_loader, valid_loader 


def calculate_rmse(ground_truth, prediction, device):
    
    if device != 'cpu':
        ground_truth = ground_truth.cpu() 
        prediction = prediction.cpu()
    
    rmse = np.sqrt(mean_squared_error(ground_truth, prediction)) 
    logger.info(f"RMSE: {rmse}")   


def write_to_files(file1, file2, prediction, labels_data, device):
    
    if device != 'cpu':
        labels_data = labels_data.cpu().numpy() 
        prediction = prediction.cpu().numpy()
    
    for j in range(len(prediction)):
        file1.write('%d\n' % prediction[j])
        file2.write('%d\n' % labels_data[j])
      
    
def setup(rank, world_size):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
  
 
def cleanup():
    dist.destroy_process_group()    
    
    
    
def main(gpu_id, args):
        
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")        
    logger.info(f"Running on gpu_id: {gpu_id}")
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data_dir}')
    
    '''
    logger.info(f'Nodes: {args.nodes}')         
    logger.info(f'GPUs: {args.gpus}')
    logger.info(f'NR: {args.nr}')
    
    args.rank = args.nr * args.gpus + gpu_id     
    logger.info(f'Rank: {args.rank}')
       
    '''
    
    logger.info(f'World Size: {args.world_size}')
  
    #might need for Sage
    logger.info(f'GPUs: {args.num_gpus}')   #gpus
    logger.info(f'Hosts: {args.hosts}')     #nodes
    logger.info(f'Current Host: {args.current_host}')   #nr
    
    os.environ["WORLD_SIZE"] = str(args.world_size)
    #host_rank = args.hosts.index(args.current_host)
    args.rank = args.hosts.index(args.current_host) * args.num_gpus + gpu_id
    os.environ["RANK"] = str(args.rank)
    logger.info(f'Rank: {args.rank}')
      
    setup(args.rank, args.world_size)
   
    logger.info("Initializing the model.")
    torch.cuda.set_device(gpu_id)
    model=net().to(gpu_id)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)  
     
    logger.info("Loading data")
    train_loader, test_loader, valid_loader = create_data_loaders(args.data_dir, args.batch_size, args.rank, args.world_size)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], output_device=gpu_id)
    
    logger.info("Training the model.")
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, gpu_id, args)
  
    logger.info("Testing the model.")
    test(model, test_loader, loss_criterion, gpu_id)  
   
    logger.info("Saving the model.")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))   
    
    cleanup()
    
if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,   #64 local train 
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    
    parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')

    '''
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N') 
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    '''
 
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    #might need for Sage
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))    #like num of nodes
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])   #current node   nr
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
  
    '''
    #**********delete after works, local train
    parser.add_argument('--data_dir', type=str, default='dogImagesplay')
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--output_dir', type=str, default='output')
    #**********
    '''
     
    args=parser.parse_args()   
    
    #args.world_size = args.gpus * args.nodes                 
    
    
    #might need for Sage
    #world_size = len(args.hosts)
    args.world_size = args.num_gpus * len(args.hosts)                
   
    
    mp.spawn(
        main,
        args=(args,),
        nprocs=args.num_gpus
    )
    
    
    
    
