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
Define pytorch model
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

""" ------------------------------------------------------------
Load model from specified directory
-- Leaving in GPU element, but need to add to the model itself
"""
def model_load(model_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRegression()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    return model.to(device)

""" ------------------------------------------------------------
Save inference results to specified directory
-- Eventually should modify to save as parquet
"""
def save_results(yhat, inference_dir):

    model_results = yhat.detach().numpy()
    logger.info("Inferences generated for {0} observations".format(model_results.shape[0]))
    path = os.path.join(inference_dir, 'model_output.csv')
    np.savetxt(path, model_results, delimiter=',')
    logger.info("Results saved @ {0}".format(inference_dir))

""" ------------------------------------------------------------
Pass data through model and save results
-- Needs additional work
"""
def inference(args):

    model = model_load(args.model_dir)
    yhat = model(X)
    save_results(yhat, args.inference_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.getcwd())
    parser.add_argument('--data-dir', type=str, default=os.getcwd())
    parser.add_argument('--label-index', type=int, default=4)
    parser.add_argument('--inference-dir', type=str, default=os.getcwd())

    X, y = data_loader(parser.parse_args())

    inference(parser.parse_args())
