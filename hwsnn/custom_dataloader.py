# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:14:11 2018

@author: alberto
"""

from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import scipy.io as sio

class AddTaskDatasetLoader(data.Dataset):

    def __init__(self, time_window=50, dataset_size = 128, randomness=False):
        super(AddTaskDatasetLoader, self).__init__()    

        self.size = dataset_size # same as batch size
        self.win = time_window
        self.randomness = randomness

    def __len__(self):
        return self.size

    @staticmethod
    def create_addtask_sample(win, idx, rnd):
        if not rnd:
            torch.manual_seed(idx)
        image = torch.zeros(win,2)
        image[:,0] = torch.rand(win)
        a, b = torch.rand(2) # random locations

        wn = int(win*0.8)

        idxa = int(a*wn/2)
        idxb = int( wn/2 + b*wn/2)
        image[[idxa, idxb],1] = 1
        #image[:,0] = torch.rand(win)
        image[:]
        label = image[idxa,0] + image[idxb,0]
        return image.clone().detach(), label.clone().detach()


    def __getitem__(self, idx):

        image, label =  self.create_addtask_sample(self.win, idx, self.randomness)
        return image, label
    