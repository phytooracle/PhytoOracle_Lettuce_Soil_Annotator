from dataset import LettucePointCloudDataset
from transformers import RandomRotation
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
from dgcnn import DGCNN
import numpy as np
import argparse
import os


# if not os.path.isdir('./data'):
#     raise Exception("./data dir does not exist.")

#-------------------- Variables --------------------#


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Plant clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i',
                        '--indir',
                        metavar='indir',
                        type = str,
                        help='input directory, there should be a folder inside this named data')

    parser.add_argument('-b',
                        '--batch_size',
                        metavar = 'batch_size',
                        type = int,
                        default= 32)


    parser.add_argument('-lr',
                        '--learning_rate',
                        metavar = 'learning_rate',
                        type = float,
                        default= 0.01)
    
    parser.add_argument('-mm',
                        '--model_momentum',
                        metavar = 'model_momentum',
                        type = float,
                        default= 0.9)
    
    parser.add_argument('-wd',
                        '--weight_decay',
                        metavar = 'weight_decay',
                        type = float,
                        default= 1e-4)

    parser.add_argument('-em',
                        '--eta_min',
                        metavar = 'eta_min',
                        type = float,
                        default= 1e-3)           


                        



    parser.add_argument('--num_epochs', type=int, default=150)

    return parser.parse_args()



args = get_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

model = DGCNN(num_classes=2).to(device)

model_name = type(model).__name__
print(f'Model: {model_name}\n{"-"*30}')

#-------------------- Dataset and DataLoader --------------------# 
train_dataset = LettucePointCloudDataset(
    root_dir=args.indir, 
    is_train=True,
    transform=transforms.Compose([
        RandomRotation()
    ])
)
train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)

#-------------------- Optimizer and Scheduler --------------------#
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.model_momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=args.eta_min)

#-------------------- Train --------------------#
model.train()
for epoch in range(args.num_epochs):
    train_loss, train_acc = .0, .0
    for input, labels in train_dataloader:
        input, labels = input.to(device).squeeze().float(), labels.to(device)
    
        optimizer.zero_grad()
        outputs = model(input)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (labels == outputs.argmax(1)).sum().item() / np.prod(labels.shape)
    scheduler.step()
    
    train_loss, train_acc = train_loss/len(train_dataloader), train_acc/len(train_dataloader)
    print(f'Epoch: {"{:3d}".format(epoch+1)}/{args.num_epochs} -> \t Train Loss: {"%.10f"%train_loss} \t Train Accuracy: {"%.4f"%train_acc}')

#-------------------- Save Model --------------------#
torch.save(model.state_dict(), f'model.pth')
