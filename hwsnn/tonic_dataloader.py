#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:28:04 2022

@author: alberto
"""

import os
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset, MemoryCachedDataset


TONIC_DATASET_PATH = os.path.join(os.environ.get('PYTHON_DRIVE_PATH'), 'tonic_datasets')


class DatasetLoader():

    def __init__(self, dataset='shd', caching='disk', num_workers=0, batch_size = 256, time_window=50):
        super(DatasetLoader, self).__init__()

        if dataset == 'shd':

            sensor_size = tonic.datasets.SHD.sensor_size
            frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time_window)
            
            trim_transform = transforms.CropTime(0, 1e6)
            
            one_hot_transform = tonic.transforms.ToOneHotEncoding(n_classes=20)

            transform = transforms.Compose([trim_transform, frame_transform])
            
            test_dataset = tonic.datasets.SHD(save_to=TONIC_DATASET_PATH,
                                            train=False,
                                            transform= transform,
                                            target_transform = one_hot_transform)

            train_dataset = tonic.datasets.SHD(save_to=TONIC_DATASET_PATH,
                                            train=True,
                                            transform= transform,
                                            target_transform = one_hot_transform)           

        if caching=='disk':
            train_cache_path = os.path.join(os.environ.get('PYTHON_DRIVE_PATH'),'tonic_cache', 'fast_dataloading_{}_train{}'.format(dataset, time_window))
            test_cache_path = os.path.join(os.environ.get('PYTHON_DRIVE_PATH'),'tonic_cache', 'fast_dataloading_{}_test{}'.format(dataset, time_window))
            test_dataset = DiskCachedDataset(test_dataset, cache_path=test_cache_path)
            train_dataset = DiskCachedDataset(train_dataset, cache_path=train_cache_path)

        elif caching =='memory':
            test_dataset = MemoryCachedDataset(test_dataset)
            train_dataset = MemoryCachedDataset(train_dataset)        

        self.test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True, drop_last=True, num_workers = num_workers)    
        self.train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last=True, num_workers = num_workers)
        
        
    def get_dataloaders(self):        
            return self.test_loader, self.train_loader