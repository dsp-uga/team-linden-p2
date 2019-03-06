"""
This code is taken from 
https://github.com/bfortuner/pytorch_tiramisu/blob/master/utils/training.py

We added a test image prediction function, get_test_results
"""

import os
import sys
import math
import string
import random
import shutil

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'

def save_weights(model, epoch, loss, err):

    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
            }, weights_fpath)

    shutil.copyfile(weights_fpath, str(WEIGHTS_PATH)+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs * h * w
    incorrect = preds.ne(targets).cpu().sum()
    err = float(incorrect) / n_pixels
    return round(err, 5)

def train(model, trn_loader, optimizer, criterion, epoch):
    use_cuda = True
    model.train()
    trn_loss = 0
    trn_error = 0
    device = torch.device("cuda" if use_cuda else "cpu")
    for trn_data in trn_loader:
        inputs, targets = trn_data[0].to(device), trn_data[1].to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        pred = get_predictions(output)
        trn_error += error(pred, targets.data.cpu())

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(model, test_loader, criterion, epoch=1):
    use_cuda = True
    model.eval()
    test_loss = 0
    test_error = 0
    device = torch.device("cuda" if use_cuda else "cpu")

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = get_predictions(output)
        test_error += error(pred, target.data.cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

def get_test_pred(model, img):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    data = Variable(img[0].to(device), volatile = True)
    output = model(data)
    pred = get_predictions(output)
    return pred
