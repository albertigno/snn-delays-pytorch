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
import h5py
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

class CustomDatasetLoader(data.Dataset):
    def __init__(self, path='load_test.mat',method = 'h', win=30, device='cpu', num_samples = None):

        self.device=device
        
        if method=='image':
            data = 1.0-1.0*(np.asarray(Image.open(path)) > 128)
            self.images = torch.from_numpy(data).permute(2,1,0)
            self.labels = torch.ones(len(self.images))
            
            print(self.labels)

        elif method=='random':
            ns= int(path.split('_')[0])
            nn = int(path.split('_')[1])
            self.images = torch.randint(0,2,(ns, win, nn))
            self.labels = torch.ones((len(self.images), 10))            
            
        elif method=='h':
            data = h5py.File(path)
            image,label = data['image'],data['label']
            image = np.transpose(image)
            label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images =  self.images[:,:,:,:,:]
            self.labels = torch.from_numpy(label).float()

        elif method=='nmnist':
            data = h5py.File(path, 'r')
            image, label = data[list(data.keys())[0]], data[list(data.keys())[1]]
            
            self.images = torch.from_numpy(np.array(image)).to(device)
            self.labels = torch.from_numpy(np.array(label)).float()
            
        elif method=='emd':
            data = sio.loadmat(path)
            image, label = data['image'], data['label']
            #image = np.transpose(image)
            #label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images = self.images[:, :, :, :]
            self.labels = torch.from_numpy(label).float()
            self.images = self.images.permute(0, 2, 3, 1)
            print("final shape of images: " + str(self.images.shape))
            
        elif method=='emd_spike':
            data = sio.loadmat(path)
            image, label = data['image'], data['label']
            #image = np.transpose(image)
            #label = np.transpose(label)
            self.images = torch.from_numpy(image)
            self.images = self.images[:, :, :, :]
            self.labels = torch.from_numpy(label).float()
            self.images = self.images.permute(0, 2, 3, 4, 1)  
            print("final shape of images: " + str(self.images.shape))

        elif method=='shd':

            data = h5py.File(path, 'r')
            image, label = data['spikes'], data['labels']
            
            # we take first half of the real dataset in shd, because activity second half is almost zero
            # time = 1.4
            x, y = self.sparse_data_generator_from_hdf5_spikes(image, label, 2*win, 700, 1.4, shuffle=False)
            
            self.images = x.to_dense()
            self.images = 1*(self.images > 0).float()
            #self.images = self.images.float()
            #self.images = 1*(self.images > 1).float()
            
            integer_encoded = y.cpu()
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

            self.labels = torch.from_numpy(onehot_encoded).float()            
            
        else:
            data = sio.loadmat(path)
            self.images = torch.from_numpy(data['image'])
            self.labels = torch.from_numpy(data['label']).float()

        self.images=self.images[:,:win,:]
        
        if num_samples is not None:
            self.images=self.images[:num_samples,:,:]
        
        #self.num_sample = int((len(self.images)//100)*100)
        self.num_sample = len(self.images)
        print('num sample: {}'.format(self.num_sample))
        
        print(self.images.size(),self.labels.size())

        
    def sparse_data_generator_from_hdf5_spikes(self, X, y, nb_steps, nb_units, max_time, shuffle=True):
        """ This generator takes a spike dataset and generates spiking network input as sparse tensors. 

        Args:
            X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
            y: The labels
            
            Author: F. Zenke
        """
        
        device = self.device
        
        labels_ = np.array(y,dtype=np.int)
        sample_index = np.arange(len(labels_))

        # compute discrete firing times
        firing_times = X['times']
        units_fired = X['units']

        time_bins = np.linspace(0, max_time, num=nb_steps)

        if shuffle:
            np.random.shuffle(sample_index)

        total_batch_count = 0
        counter = 0
        
        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(sample_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([len(labels_),nb_steps,nb_units])).to(device)
        y_batch = torch.tensor(labels_,device=device)

        return X_batch.to(device=device), y_batch.to(device=device)

        
    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample


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

class MultTaskDatasetLoader(AddTaskDatasetLoader):

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
        image[:,0] = torch.rand(win)
        image[:]
        label = image[idxa,0]*image[idxb,0]
        return image.clone().detach(), label.clone().detach()

