
#import sagemaker_containers
import argparse
import json
import logging
import os
import sys

import pandas as pd
import numpy as np
import s3fs
import boto3

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

""" ------------------------------------------------------------
Load data from specified directory
-- Split data into X and y tensors based on label label_index
-- Can switch from index to column name pending parquet importing
"""
def data_loader(args):

    data = pd.read_csv(args.data_dir)
    data = np.array(data)
    X = Variable(torch.Tensor(data[:,:args.label_index]).float())
    y = Variable(torch.Tensor(data[:,args.label_index]).long())

    return X, y

""" ------------------------------------------------------------
Create pytorch model
-- Currently using basic logistic regression
-- Would need to search for an efficient implementation of ElasticNet
-- Very easy to plug and play to test new things
"""
class LogisticRegression(torch.nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(X.shape[1], len(y.unique())) # Feature dim in and label value count out
        self.softmax = nn.Softmax(dim=1) # Ensures final output between 0-1

    def forward(self, X):
        X = torch.sigmoid(self.linear(X)) # Torch sigmoid implementation
        X = self.softmax(X)
        return X

# class NeuralNet(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(4, 100)
#         self.fc2 = nn.Linear(100, 250)
#         self.fc2 = nn.Linear(250, 50)
#         self.fc3 = nn.Linear(50, 2)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, X):
#         X = F.relu(self.fc1(X))
#         X = self.fc2(X)
#         X = self.fc3(X)
#         X = self.softmax(X)
#         return X

""" ------------------------------------------------------------
Save down the model after training to the specified path
-- Need to work on optionality
"""
def save_model(model, model_dir):
    logger.info("Model saved @ {0}".format(model_dir))
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)

""" ------------------------------------------------------------
Save down the model weights
-- Need verify***
"""
def save_weights(model, weight_dir):
    model_weights = model.linear.weight.detach().numpy()
    logger.info("Weights saved @ {0}".format(weight_dir))
    path = os.path.join(weight_dir, 'model_weights.csv')
    np.savetxt(path, model_weights, delimiter=',')

""" ------------------------------------------------------------
Model Training
-- Need to work on optionality for optimization method + Loss
-- Added l1 regularization? (maybe), but need to add group Lasso
"""
def train(args):

    # ALWAYS SET SEED TO 42!
    torch.manual_seed(42)

    # Specficy Model, and Loss and Optimization Methodologies
    model = LogisticRegression()
    #model = nnModel = NeuralNet()
    #model = torch.nn.DataParallel(model) ## Not sure if this does anything
    criterion = nn.CrossEntropyLoss() ## Need to vet diversion from LogLoss
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    ## Need to add batch training for crossval and better logging
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        yhat = model(X)
        l1_weights = torch.cat([x.view(-1) for x in model.linear.parameters()])
        l1_reg = (args.l1_lambda * torch.norm(l1_weights, 1))
        ce_loss = criterion(yhat, y)
        loss = (ce_loss + l1_reg)
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            logger.info('Epoch {}, Average loss: {:.4f}'.format(epoch, loss.item()))

    save_model(model, args.model_dir)
    save_weights(model, args.weight_dir)
    print("Model Successfully Trained!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.getcwd())
    parser.add_argument('--weight-dir', type=str, default=os.getcwd())
    parser.add_argument('--data-dir', type=str, default=os.getcwd())
    parser.add_argument('--label-index', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l1_lambda', type=float, default=0.0001)
    #parser.add_argument('--batch-size', type=int, default=64)
    #parser.add_argument('--test-batch-size', type=int, default=1000)
    #parser.add_argument('--save-model', action='store_true', default=False)

    X, y = data_loader(parser.parse_args())

    train(parser.parse_args())
