#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:28:04 2022

@author: alberto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from my_snn.abstract_rsnn import Abstract_SNN_Delays
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RSNN_d_d(Abstract_SNN_Delays, nn.Module):

    '''
    Main delay implementation so far.
    It creates multilayer feedforward or recurrent SNN with fixed neurons per layer.
    Delays are 'internal'. No delays in the input-to-layer1 to decrease parameter count.
    '''

    def define_delays(self):

        if self.delay != None:
            self.max_d = self.delay[0]
            self.stride = self.delay[1]
            # self.delays = np.linspace(0, self.max_d, self.num_d).astype(int)
            self.delays = list(range(0, self.max_d, self.stride))
        else:
            # self.input_names = ['f0_i']
            self.max_d = 0
            self.delays = [0]
        print('delays: '+str(self.delays))

    def set_input_layer(self):

        setattr(self, 'f0_i', nn.Linear(self.num_input, self.num_hidden, bias=False))

    def set_hidden_layers(self):
        bias = False
        for name in self.h_names[:-1]:
            r_name = name[:2]+'_'+name[:2]

            if self.deepnet_type == 'f':
                setattr(self, name, nn.Linear(self.num_hidden *
                        len(self.delays), self.num_hidden, bias=bias))
            elif self.deepnet_type == 'r':
                setattr(self, r_name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))
                setattr(self, name, nn.Linear(self.num_hidden *
                        len(self.delays), self.num_hidden, bias=bias))

        name = self.h_names[-1]
        r_name = name[:2]+'_'+name[:2]
        if self.deepnet_type == 'r':
            setattr(self, r_name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))
        setattr(self, name, nn.Linear(self.num_hidden*len(self.delays), self.num_output, bias=bias))

    def init_state(self):

        mems = {}
        spikes = {}
        extended_spikes = {}
        self.mem_state = {}
        self.spike_state = {}

        for name in self.layer_names:
            extended_spikes[name] = torch.zeros(
                self.batch_size, self.win+self.max_d, self.num_hidden, device=self.device)
            mems[name] = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
            spikes[name] = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

            if self.debug:
                self.spike_state['input'] = torch.zeros(self.win, self.batch_size,
                                                        self.num_input, device=self.device)
                self.mem_state[name] = torch.zeros(
                    self.win, self.batch_size, self.num_hidden, device=self.device)
                self.spike_state[name] = torch.zeros(
                    self.win, self.batch_size, self.num_hidden, device=self.device)
                self.mem_state['output'] = torch.zeros(
                    self.win, self.batch_size, self.num_output, device=self.device)
                self.spike_state['output'] = torch.zeros(
                    self.win, self.batch_size, self.num_output, device=self.device)

        o_mem = torch.zeros(self.batch_size, self.num_output, device=self.device)
        o_spike = torch.zeros(self.batch_size, self.num_output, device=self.device)

        return extended_spikes, mems, spikes, o_mem, o_spike

    def logger(self, *args):
        if self.debug:
            x, mems, spikes, o_mem, o_spike = args

            self.spike_state['input'][self.step, :, :] = x
            for name in self.layer_names:
                self.mem_state[name][self.step, :, :] = mems[name]
                self.spike_state[name][self.step, :, :] = spikes[name]
            self.mem_state['output'][self.step, :, :] = o_mem
            self.spike_state['output'][self.step, :, :] = o_spike

    def forward(self, input):

        extended_spikes, mems, spikes, o_mem, o_spike = self.init_state()
        self.o_sumspike = output_mot = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0)  # for spike-regularization

        # extended_input = torch.zeros(self.batch_size, self.win+self.max_d,
        #                             self.num_input, device=self.device)
        #extended_input[:, self.max_d:, :] = input

        for step in range(self.win):

            self.step = step

            #delayed_x = extended_input[:, step:step+self.max_d, :]

            # input layer is propagated (without delays)
            prev_spikes = self.f0_i(input[:, step, :].view(self.batch_size, -1))

            self.w_idx = 0
            self.tau_idx = 0

            for layer in self.layer_names:

                mems[layer], spikes[layer] = self.mem_update_fn(
                    prev_spikes.reshape(self.batch_size, -1), spikes[layer], mems[layer])

                extended_spikes[layer][:, step+self.max_d,
                                       :] = spikes[layer].clone()  # possibly detach()

                # prev_spikes = extended_spikes[layer][:,step:step+self.max_d,:].transpose(1,2).clone() # no strided delays
                prev_spikes = extended_spikes[layer][:, range(
                    step, step+self.max_d, self.stride), :].transpose(1, 2).clone()

                self.h_sumspike = self.h_sumspike + spikes[layer].sum()

            o_mem, o_spike = self.mem_update_out(
                prev_spikes.reshape(self.batch_size, -1), o_spike, o_mem)

            self.logger(input[:, step, :].view(self.batch_size, -1), mems, spikes, o_mem, o_spike)

            self.o_sumspike = self.o_sumspike + o_spike

            output_mot = output_mot + F.softmax(o_mem, dim=1)

        self.h_sumspike = self.h_sumspike / self.n_layers

        output_sum = self.o_sumspike / (self.win)

        return output_sum, output_mot


    def mem_update_out(self, i_spike, o_spike, mem):
        # alpha = torch.exp(-1. / self.tau_m_h[self.tau_idx]).to(self.device)
        alpha = torch.sigmoid(self.tau_m_h[self.tau_idx]).to(self.device)
        mem = mem * alpha * (1 - o_spike) + self.h_layers[self.w_idx](i_spike)
        self.tau_idx = self.tau_idx+1
        self.w_idx = self.w_idx + 1
        o_spike = self.act_fun(mem-self.output_thresh)
        mem = mem*(mem < self.thresh)
        return mem, o_spike

    def mem_update(self, i_spike, o_spike, mem):
        # alpha = torch.exp(-1. / self.tau_m_h[self.tau_idx]).to(self.device)
        alpha = torch.sigmoid(self.tau_m_h[self.tau_idx]).to(self.device)
        mem = mem * alpha * (1 - o_spike) + self.h_layers[self.w_idx](i_spike)
        self.tau_idx = self.tau_idx+1
        self.w_idx = self.w_idx + 1
        o_spike = self.act_fun(mem-self.thresh)
        mem = mem*(mem < self.thresh)
        return mem, o_spike

    def mem_update_rnn(self, i_spike, o_spike, mem):
        # alpha = torch.exp(-1. / self.tau_m_h[self.tau_idx]).to(self.device)
        alpha = torch.sigmoid(self.tau_m_h[self.tau_idx]).to(self.device)
        a = self.h_layers[self.w_idx](i_spike)
        b = self.h_layers[self.w_idx+1](o_spike)  # process recurrent spikes
        self.tau_idx = self.tau_idx+1

        self.w_idx = self.w_idx + 2
        c = mem * alpha * (1-o_spike)
        # mem = a0 + a2 + a4 + b + c
        mem = a + b + c
        o_spike = self.act_fun(mem-self.thresh)
        mem = mem*(mem < self.thresh)
        return mem, o_spike


    def get_delayed_inputs(self, x):

        # input_activations  = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        input_activations = self.f0_i(x)

        return input_activations

    def pool_delays(self, lyr='i', k=1, freeze=True):
        '''
        k number of delays to be selected
        Create one delay per synapse in multi-delay model, by choosing the one with highest absolute value
        NOTE: it' only works with delay stride=1

        reshape to (output, input, num_d)

        '''

        trainable = not freeze

        def get_pooling_mask_old(w):
            mask = torch.zeros(w.shape, device=self.device)

            #c_argmax = torch.argmax(torch.abs(w), dim=2)

            c_argmax = torch.argmax(w, dim=2)
            c_argmin = torch.argmin(w, dim=2)

            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    mask[i, j, c_argmax[i, j]] = 1.0
                    mask[i, j, c_argmin[i, j]] = 1.0
            return mask

        def get_pooling_mask(w):

            mask = torch.zeros(w.shape, device=self.device)

            ww = torch.abs(w)

           

            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    # find indices of k highest values
                    idx_k = np.argpartition(ww[i,j,:].cpu().numpy(), -k)[-k:]
                    for d in idx_k: 
                        mask[i, j, d] = 1.0
            return mask

        num_d = len(self.delays)

        if 'i' in lyr:
            w = self.f0_i.weight.data.reshape(
                self.num_hidden, self.num_input, num_d)
            mask = get_pooling_mask(w)
            self.mask_weights(self.f0_i, mask.reshape(
                self.num_hidden, self.num_input*num_d), override=False, trainable=trainable)

        if 'h' in lyr:
            for layer in self.h_names[:-1]:
                w = getattr(self, layer).weight.data.reshape(self.num_hidden, self.num_hidden, num_d)
                mask = get_pooling_mask(w)
                self.mask_weights(getattr(self, layer), mask.reshape(
                    self.num_hidden, self.num_hidden*num_d), override=False, trainable=trainable)
        
        if 'o' in lyr:
            # w = torch.abs(getattr(self, self.h_names[-1]).weight.data).reshape(self.num_hidden,self.max_d, self.num_output)
            w = getattr(self, self.h_names[-1]).weight.data.reshape(self.num_output, self.num_hidden, num_d)
            mask = get_pooling_mask(w)
            self.mask_weights(getattr(self, self.h_names[-1]), mask.reshape(
                self.num_output, self.num_hidden*num_d), override=False, trainable=trainable)

    def plot_per_neuron_delays(self, w, n_cols=3, num_channels=1, mode='image', seed=None):
        '''
        randomly sample weight per Neuron
        if mode == d, plot all delays in order,else pick random

        w.shape = (output_neurons, input_neurons, delays)
        '''

        if seed is not None:
            np.random.seed(seed)

        w = w.weight.data.cpu().numpy()

        sample_idx = np.random.choice(np.arange(w.shape[0]), n_cols)
        n_rows = min(10, len(self.delays))

        sample = w[sample_idx, :]

        s = int(w.shape[1]/num_channels)

        a = len(self.delays)
        b = int(w.shape[1] / (num_channels * len(self.delays)))
        c, d = self.square(b)

        self.sample = [[] for x in range(n_cols)]

        if mode == 'image':
            for i, x in enumerate(sample):
                for r in range(a):
                    if r < n_rows:
                        xx = x[:s].reshape(b, a)
                        plt.subplot(n_rows, n_cols, i+r*n_cols+1)
                        plt.imshow(xx[:, r].reshape(c, d), cmap='RdBu')
                    self.sample[i].append(xx[:, r].reshape(c, d))
        elif mode == 'avg':
            for i, x in enumerate(sample):
                xx = x[:s].reshape(b, a)
                plt.subplot(n_cols, 1, i+1)
                plt.plot(np.sum(np.abs(xx), axis=1))
        elif mode == 'synapse':
            for i, x in enumerate(sample):
                xx = x[:s].reshape(b, a)
                plt.subplot(n_cols, 1, i+1)
                plt.plot(xx[np.random.randint(b), :])

    def plot_delays(self, mode='histogram'):

        wi = self.f0_i.weight.data.reshape(self.num_hidden, self.num_input, -1)

        w = [wi]

        for layer in self.h_names[:-1]:
            w.append(getattr(self, layer).weight.data.reshape(
                self.num_hidden, self.num_hidden,  len(self.delays)))

        wo = getattr(
            self, self.h_names[-1]).weight.data.reshape(self.num_output, self.num_hidden, len(self.delays))
        w.append(wo)

        fig = plt.figure(figsize=(20, 20))

        self.w_d = w

        #fig = plt.figure()

        ld = len(self.delays)

        # gs = fig.add_gridspec(len(w),ld, height_ratios=[50, snn.num_hidden / ld, snn.num_output /ld])

        n_rows = int(np.ceil(ld / 3))

        gs = fig.add_gridspec(1+self.n_layers*n_rows, 3)

        p = 1
        fig.add_subplot(gs[0, :])
        if mode == 'matrix':
            plt.imshow(w[0][:, :, 0].cpu().numpy(), vmin=w[0].min().item(),
                       vmax=w[0].max().item(), cmap='RdBu')
            for y in range(1, len(w)):
                for x in range(len(self.delays)):
                    fig.add_subplot(gs[p + int(x/3), x % 3])
                    plt.imshow(w[y][:, :, x].cpu().numpy(),
                               vmin=w[y].min().item(), vmax=w[y].max().item(), cmap='RdBu')
                    plt.title('d= {}'.format(self.delays[x]))
                p = p+n_rows
        elif mode == 'histogram':
            q = list(w[0][:, :, 0].cpu().numpy().reshape(1, -1)[0])
            # _, _, _ = plt.hist(q, bins=200)
            sns.histplot(q, bins=200)
            for y in range(1, len(w)):
                for x in range(len(self.delays)):
                    fig.add_subplot(gs[p + int(x/3), x % 3])
                    q = list(w[y][:, :, x].cpu().numpy().reshape(1, -1)[0])
                    # _, _, _ = plt.hist(q, bins=200)
                    sns.histplot(q, bins=200)
                    plt.title('d= {}'.format(self.delays[x]))
                p = p+n_rows
        plt.tight_layout()
        # return fig

    def prune_weights(self, percentage):

        w = torch.abs(self.f0_i.weight.data)


class RSNN_d_i(RSNN_d_d):

    def set_input_layer(self):

        setattr(self, 'f0_i', nn.Linear(self.num_input*len(self.delays), self.num_hidden, bias=False))

    def set_hidden_layers(self):
        bias = False
        for name in self.h_names[:-1]:
            r_name = name[:2]+'_'+name[:2]

            if self.deepnet_type == 'f':
                setattr(self, name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))
            elif self.deepnet_type == 'r':
                setattr(self, r_name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))
                setattr(self, name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))

        name = self.h_names[-1]
        r_name = name[:2]+'_'+name[:2]
        if self.deepnet_type == 'r':
            setattr(self, r_name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))
        setattr(self, name, nn.Linear(self.num_hidden, self.num_output, bias=bias))


    def forward(self, input):

        _, mems, spikes, o_mem, o_spike = self.init_state()
        self.o_sumspike = output_mot = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0)  # for spike-regularization

        extended_input = torch.zeros(self.batch_size, self.win+self.max_d,
                                     self.num_input, device=self.device)
        extended_input[:, self.max_d:, :] = input

        for step in range(self.win):

            self.step = step

            delayed_x = extended_input[:, range(step, step+self.max_d, self.stride), :]

            # prev_spikes = self.f0_i(delayed_x.view(self.batch_size, -1)) # input layer is propagated

            prev_spikes = self.f0_i(delayed_x.transpose(1, 2).reshape(
                self.batch_size, -1))  # input layer is propagated (with delays)

            # prev_spikes = self.f0_i(input[:, step, :].view(self.batch_size, -1)) # input layer is propagated (without delays)

            self.w_idx = 0
            self.tau_idx = 0

            for layer in self.layer_names:
                mems[layer], spikes[layer] = self.mem_update_fn(prev_spikes, spikes[layer], mems[layer])
                prev_spikes = spikes[layer]
                self.h_sumspike = self.h_sumspike + spikes[layer].sum()

            o_mem, o_spike = self.mem_update(prev_spikes, o_spike, o_mem)

            self.logger(input[:, step, :].view(self.batch_size, -1), mems, spikes, o_mem, o_spike)

            self.o_sumspike = self.o_sumspike + o_spike

            output_mot = output_mot + F.softmax(o_mem, dim=1)

        self.h_sumspike = self.h_sumspike / self.n_layers

        output_sum = self.o_sumspike / (self.win)

        return output_sum, output_mot

class RSNN_d_all(RSNN_d_d):

    def set_input_layer(self):

        setattr(self, 'f0_i', nn.Linear(self.num_input*len(self.delays), self.num_hidden, bias=False))

    def forward(self, input):

        extended_spikes, mems, spikes, o_mem, o_spike = self.init_state()
        self.o_sumspike = output_mot = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0)  # for spike-regularization

        extended_input = torch.zeros(self.batch_size, self.win+self.max_d,
                                     self.num_input, device=self.device)
        extended_input[:, self.max_d:, :] = input

        for step in range(self.win):

            self.step = step

            delayed_x = extended_input[:, range(step, step+self.max_d, self.stride), :]

            # prev_spikes = self.f0_i(delayed_x.view(self.batch_size, -1)) # input layer is propagated

            prev_spikes = self.f0_i(delayed_x.transpose(1, 2).reshape(
                self.batch_size, -1))  # input layer is propagated (with delays)

            # prev_spikes = self.f0_i(input[:, step, :].view(self.batch_size, -1)) # input layer is propagated (without delays)

            self.w_idx = 0
            self.tau_idx = 0

            for layer in self.layer_names:
                # print(layer)
                # print(prev_spikes.view(self.batch_size, -1).shape)

                mems[layer], spikes[layer] = self.mem_update_fn(
                    prev_spikes.reshape(self.batch_size, -1), spikes[layer], mems[layer])

                extended_spikes[layer][:, step+self.max_d,
                                       :] = spikes[layer].clone()  # possibly detach()

                prev_spikes = extended_spikes[layer][:, range(
                    step, step+self.max_d, self.stride), :].transpose(1, 2).clone()
                # prev_spikes = spikes[layer]

                self.h_sumspike = self.h_sumspike + spikes[layer].sum()

            o_mem, o_spike = self.mem_update(
                prev_spikes.reshape(self.batch_size, -1), o_spike, o_mem)

            self.logger(input[:, step, :].view(self.batch_size, -1), mems, spikes, o_mem, o_spike)

            self.o_sumspike = self.o_sumspike + o_spike

            output_mot = output_mot + F.softmax(o_mem, dim=1)

        self.h_sumspike = self.h_sumspike / self.n_layers

        output_sum = self.o_sumspike / (self.win)

        return output_sum, output_mot



class RSNN_d_d_ALIF(RSNN_d_d):

    def define_tau_m(self):

        if self.tau_m != 'adp':
            self.tau_m_h = [nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
                            for i in range(self.n_layers+1)]
        else:
            mean = 0  # 0.83
            std = 1  # 0.1
            for i in range(self.n_layers):
                name = 'tau_m_'+str(i+1)
                name_adp = 'tau_adp_'+str(i+1)
                setattr(self, name, nn.Parameter(torch.Tensor(self.num_hidden)))
                nn.init.normal_(getattr(self, name), mean, std)
                setattr(self, name_adp, nn.Parameter(torch.Tensor(self.num_hidden)))
                nn.init.normal_(getattr(self, name_adp), mean, std)

            setattr(self, 'tau_m_o', nn.Parameter(torch.Tensor(self.num_output)))
            nn.init.normal_(getattr(self, 'tau_m_o'), mean, std)
            setattr(self, 'tau_adp_o', nn.Parameter(torch.Tensor(self.num_output)))
            nn.init.normal_(getattr(self, 'tau_adp_o'), mean, std)

    def set_layer_lists(self):
        self.h_layers = [nn.Identity()]
        for name in self.h_names:
            self.h_layers.append(getattr(self, name))

        if self.tau_m == 'adp':
            self.tau_m_h = [getattr(self, name) for name in ['tau_m_'+str(i+1)
                                                             for i in range(self.n_layers)]]
            self.tau_m_h.append(self.tau_m_o)
            self.tau_adp_h = [getattr(self, name)
                              for name in ['tau_adp_'+str(i+1) for i in range(self.n_layers)]]
            self.tau_adp_h.append(self.tau_adp_o)

    def thresh_logger(self, B):

        if self.debug:
            layer_id = self.alif_counter % self.n_layers
            self.thresh_state[self.layer_names[layer_id]][self.step, :, :] = B
            self.alif_counter += 1

    def init_state(self):

        mems = {}
        spikes = {}
        d = {}
        extended_spikes = {}
        self.spike_state = {}
        self.mem_state = {}
        self.thresh_state = {}

        for name in self.layer_names:
            extended_spikes[name] = torch.zeros(
                self.batch_size, self.win+self.max_d, self.num_hidden, device=self.device)
            mems[name] = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
            spikes[name] = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
            d[name] = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
            if self.debug:
                self.spike_state['input'] = torch.zeros(self.win, self.batch_size,
                                                        self.num_input, device=self.device)
                self.mem_state[name] = torch.zeros(
                    self.win, self.batch_size, self.num_hidden, device=self.device)
                self.spike_state[name] = torch.zeros(
                    self.win, self.batch_size, self.num_hidden, device=self.device)
                self.mem_state['output'] = torch.zeros(
                    self.win, self.batch_size, self.num_output, device=self.device)
                self.spike_state['output'] = torch.zeros(
                    self.win, self.batch_size, self.num_output, device=self.device)

                self.thresh_state[name] = torch.zeros(
                    self.win, self.batch_size, self.num_hidden, device=self.device)
                self.alif_counter = 0

        o_mem = torch.zeros(self.batch_size, self.num_output, device=self.device)
        o_spike = torch.zeros(self.batch_size, self.num_output, device=self.device)

        return extended_spikes, mems, spikes, o_mem, o_spike, d

    def forward(self, input):

        extended_spikes, mems, spikes, o_mem, o_spike, d = self.init_state()
        self.o_sumspike = output_mot = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0)  # for spike-regularization

        for step in range(self.win):

            self.step = step

            # input layer is propagated (without delays)
            prev_spikes = self.f0_i(input[:, step, :].view(self.batch_size, -1))

            self.w_idx = 0
            self.tau_idx = 0

            for layer in self.layer_names:

                mems[layer], spikes[layer], d[layer] = self.mem_update_fn(
                    prev_spikes.reshape(self.batch_size, -1), spikes[layer], mems[layer], d[layer])

                extended_spikes[layer][:, step+self.max_d,
                                       :] = spikes[layer].clone()  # possibly detach()

                prev_spikes = extended_spikes[layer][:, range(
                    step, step+self.max_d, self.stride), :].transpose(1, 2).clone()

                self.h_sumspike = self.h_sumspike + spikes[layer].sum()

            o_mem, o_spike = self.mem_update_o(
                prev_spikes.reshape(self.batch_size, -1), o_spike, o_mem)

            self.logger(input[:, step, :].view(self.batch_size, -1), mems, spikes, o_mem, o_spike)

            self.o_sumspike = self.o_sumspike + o_spike

            output_mot = output_mot + F.softmax(o_mem, dim=1)

        self.h_sumspike = self.h_sumspike / self.n_layers

        output_sum = self.o_sumspike / (self.win)

        return output_sum, output_mot


    def mem_update(self, i_spike, o_spike, mem, d):

        alpha = torch.sigmoid(self.tau_m_h[self.tau_idx]).to(self.device)
        ro = torch.sigmoid(self.tau_adp_h[self.tau_idx]).to(self.device)

        d = ro*d + (1-ro)*(o_spike)
        B = self.thresh + 1.84*d

        a = self.h_layers[self.w_idx](i_spike)
        mem = mem * alpha + a*(1-alpha) - B*o_spike # bojian

        #mem = mem * alpha * (1 - o_spike) + self.h_layers[self.w_idx](i_spike)
        #mem = (mem * alpha + self.h_layers[self.w_idx](i_spike)) * (1 - o_spike)
        #mem = mem * alpha + self.h_layers[self.w_idx](i_spike) - B*o_spike

        self.tau_idx = self.tau_idx+1
        self.w_idx = self.w_idx + 1
        o_spike = self.act_fun(mem-B)
        # mem = mem*(mem < B)
        self.thresh_logger(B.detach())
        return mem, o_spike, d

    def mem_update_o(self, i_spike, o_spike, mem):
        # alpha = torch.exp(-1. / self.tau_m_h[self.tau_idx]).to(self.device)
        alpha = torch.sigmoid(self.tau_m_h[self.tau_idx]).to(self.device)
        # mem = mem * alpha * (1 - o_spike) + self.h_layers[self.w_idx](i_spike) # refract
        mem = (mem * alpha) * (1 - o_spike) + self.h_layers[self.w_idx](i_spike)  # alberto

        self.tau_idx = self.tau_idx+1
        self.w_idx = self.w_idx + 1
        o_spike = self.act_fun(mem-self.output_thresh)

        return mem, o_spike

    def mem_update_rnn(self, i_spike, o_spike, mem, d):

        alpha = torch.sigmoid(self.tau_m_h[self.tau_idx]).to(self.device)
        ro = torch.sigmoid(self.tau_adp_h[self.tau_idx]).to(self.device)

        d = ro*d + (1-ro)*(o_spike)
        B = self.thresh + 1.84*d

        a = self.h_layers[self.w_idx](i_spike) # process input spikes
        b = self.h_layers[self.w_idx+1](o_spike)  # process recurrent spikes
        self.tau_idx = self.tau_idx+1
        self.w_idx = self.w_idx + 2

        # c = mem*alpha
        # mem = a + b + c * (1-o_spike) # alberto
        # mem = (a+b+c) * (1 - o_spike) # refractory
        mem = mem * alpha + (1-alpha)*(a+b) - B*o_spike # bojian
        # mem = a+b+c - B*o_spike  # reset by subtraction
        o_spike = self.act_fun(mem-B)
        # mem = mem*(mem < B)
        self.thresh_logger(B.detach())
        return mem, o_spike, d