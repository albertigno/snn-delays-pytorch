#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:28:04 2022

@author: alberto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from my_snn.abstract_rsnn import Abstract_SNN

class RSNN(Abstract_SNN, nn.Module):

    def define_operations(self):

        if type(self.tau_m) == float:
            self.tau_m_h = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            self.tau_m_o = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            # self.tau_m_o = torch.Tensor([self.tau_m])
        else:
            self.tau_m_h = nn.Parameter(torch.Tensor(self.num_hidden))
            nn.init.normal_(self.tau_m_h, 0.83, 0.1)
            self.tau_m_o = nn.Parameter(torch.Tensor(self.num_output))
            nn.init.normal_(self.tau_m_o, 0.83, 0.1)

        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias=False)
        self.fc_hh = nn.Linear(self.num_hidden, self.num_hidden, bias=False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias=False)

    def logger(self, *args):
        if self.debug == True:
            i_spike, h_mem, h_spike, o_mem, o_spike = args
            self.snn_state['i_spike'][self.step, :, :] = i_spike
            self.snn_state['h_mem'][self.step, :, :] = h_mem
            self.snn_state['h_spike'][self.step, :, :] = h_spike
            self.snn_state['o_mem'][self.step, :, :] = o_mem
            self.snn_state['o_spike'][self.step, :, :] = o_spike

    def init_state(self):
        h_mem = h_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = torch.zeros(self.batch_size, self.num_output, device=self.device)

        if self.debug == True:
            self.snn_state = {}
            self.snn_state['i_spike'] = torch.zeros(
                self.win, self.batch_size, self.num_input, device=self.device)
            self.snn_state['h_mem'] = torch.zeros(
                self.win, self.batch_size, self.num_hidden, device=self.device)
            self.snn_state['h_spike'] = torch.zeros(
                self.win, self.batch_size, self.num_hidden, device=self.device)
            self.snn_state['o_mem'] = torch.zeros(
                self.win, self.batch_size, self.num_output, device=self.device)
            self.snn_state['o_spike'] = torch.zeros(
                self.win, self.batch_size, self.num_output, device=self.device)

        return h_mem, h_spike, o_mem, o_spike

    def forward(self, input):

        h_mem, h_spike, o_mem, o_spike = self.init_state()

        o_sumspike = output_mot = torch.zeros(self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0)  # for spike-regularization

        for step in range(self.win):

            self.step = step

            x = input[:, step, :]

            i_spike = x.view(self.batch_size, -1)

            h_mem, h_spike = self.mem_update_rnn(i_spike, h_spike, h_mem)
            o_mem, o_spike = self.mem_update_out(h_spike, o_spike, o_mem)

            self.h_sumspike = self.h_sumspike + h_spike.sum()
            o_sumspike = o_sumspike + o_spike

            self.logger(i_spike.detach(), h_mem.detach(), h_spike.detach(),
                        o_mem.detach(), o_spike.detach())

            output_mot = output_mot + F.softmax(o_mem, dim=1)

        output_sum = o_sumspike / (self.win)

        return output_sum, output_mot

    def mem_update_out(self, i_spike, o_spike, mem):
        alpha = torch.exp(-1. / self.tau_m_o).to(self.device)
        mem = mem * alpha * (1 - o_spike) + self.fc_ho(i_spike)
        o_spike = self.act_fun(mem-self.output_thresh)
        # mem = mem*(mem < self.output_thresh)
        return mem, o_spike

    def mem_update_rnn(self, i_spike, o_spike, mem):
        alpha = torch.exp(-1. / self.tau_m_h).to(self.device)
        a = self.fc_ih(i_spike)  # process spikes from input
        b = self.fc_hh(o_spike)  # process recurrent spikes
        c = mem * alpha * (1-o_spike)
        mem = a + b + c
        o_spike = self.act_fun(mem-self.thresh)
        # mem = mem*(mem < self.thresh)
        return mem, o_spike

class RSNN_2l(Abstract_SNN, nn.Module):

    def define_operations(self):

        if type(self.tau_m) == float:
            self.tau_m_h_1 = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            self.tau_m_h_2 = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            self.tau_m_o = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            #self.tau_m_o = torch.Tensor([self.tau_m])
        else:
            self.tau_m_h_1 = nn.Parameter(torch.Tensor(self.num_hidden))
            self.tau_m_h_2 = nn.Parameter(torch.Tensor(self.num_hidden))
            nn.init.normal_(self.tau_m_h_1, 0, 1)
            nn.init.normal_(self.tau_m_h_2, 0, 1)
            self.tau_m_o = nn.Parameter(torch.Tensor(self.num_output))
            nn.init.normal_(self.tau_m_o, 0, 1)

        self.tau_1 = [self.tau_m_h_1]
        self.tau_2 = [self.tau_m_h_2]

        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias=False)
        self.fc_h1h1 = nn.Linear(self.num_hidden, self.num_hidden, bias=False)
        self.fc_h1h2 = nn.Linear(self.num_hidden, self.num_hidden, bias=False)
        self.fc_h2h2 = nn.Linear(self.num_hidden, self.num_hidden, bias=False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias=False)

    def logger(self, *args):
        if self.debug == True:
            i_spike, h1_mem, h1_spike, h2_mem, h2_spike, o_mem, o_spike = args
            self.snn_state['i_spike'][self.step, :, :] = i_spike
            self.snn_state['h1_mem'][self.step, :, :] = h1_mem
            self.snn_state['h1_spike'][self.step, :, :] = h1_spike
            self.snn_state['h2_mem'][self.step, :, :] = h2_mem
            self.snn_state['h2_spike'][self.step, :, :] = h2_spike
            self.snn_state['o_mem'][self.step, :, :] = o_mem
            self.snn_state['o_spike'][self.step, :, :] = o_spike

    def init_state(self):
        h1_mem = h1_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        h2_mem = h2_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = torch.zeros(self.batch_size, self.num_output, device=self.device)

        if self.debug == True:
            self.snn_state = {}
            self.snn_state['i_spike'] = torch.zeros(
                self.win, self.batch_size, self.num_input, device=self.device)
            self.snn_state['h1_mem'] = torch.zeros(
                self.win, self.batch_size, self.num_hidden, device=self.device)
            self.snn_state['h1_spike'] = torch.zeros(
                self.win, self.batch_size, self.num_hidden, device=self.device)
            self.snn_state['h2_mem'] = torch.zeros(
                self.win, self.batch_size, self.num_hidden, device=self.device)
            self.snn_state['h2_spike'] = torch.zeros(
                self.win, self.batch_size, self.num_hidden, device=self.device)
            self.snn_state['o_mem'] = torch.zeros(
                self.win, self.batch_size, self.num_output, device=self.device)
            self.snn_state['o_spike'] = torch.zeros(
                self.win, self.batch_size, self.num_output, device=self.device)

        return h1_mem, h1_spike, h2_mem, h2_spike, o_mem, o_spike

    def forward(self, input):

        h1_mem, h1_spike, h2_mem, h2_spike, o_mem, o_spike = self.init_state()
        o_sumspike = output_mot = torch.zeros(self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0)  # for spike-regularization

        for step in range(self.win):

            self.step = step

            x = input[:, self.step, :]

            i_spike = x.view(self.batch_size, -1)

            h1_mem, h1_spike = self.mem_update_rnn(
                i_spike, h1_spike, h1_mem, self.fc_ih, self.fc_h1h1, self.tau_1)
            h2_mem, h2_spike = self.mem_update_rnn(
                h1_spike, h2_spike, h2_mem, self.fc_h1h2, self.fc_h2h2, self.tau_2)
            o_mem, o_spike = self.mem_update_out(h2_spike, o_spike, o_mem)

            self.h_sumspike = self.h_sumspike + (h1_spike.sum() + h2_spike.sum())/2
            o_sumspike = o_sumspike + o_spike

            self.logger(i_spike, h1_mem, h1_spike, h2_mem, h2_spike, o_mem, o_spike)

            output_mot = output_mot + F.softmax(o_mem, dim=1)

        output_sum = o_sumspike / (self.win)

        return output_sum, output_mot

    def mem_update_out(self, i_spike, o_spike, mem):
        alpha = torch.sigmoid(self.tau_m_o).to(self.device)
        mem = mem * alpha * (1 - o_spike) + self.fc_ho(i_spike)
        o_spike = self.act_fun(mem-self.output_thresh)
        #mem = mem*(mem < self.output_thresh)
        return mem, o_spike

    def mem_update_rnn(self, i_spike, o_spike, mem, op_input, op_recurrent, tau_m):
        alpha = torch.sigmoid(tau_m[0]).to(self.device)
        a = op_input(i_spike)  # process spikes from input
        b = op_recurrent(o_spike)  # process recurrent spikes
        c = mem * alpha * (1-o_spike)
        mem = a + b + c
        o_spike = self.act_fun(mem-self.thresh)
        #mem = mem*(mem < self.thresh)
        return mem, o_spike


class RSNN_2l_ALIF(RSNN_2l):

    def define_operations(self):

        if type(self.tau_m) == float:
            self.tau_m_h_1 = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            self.tau_m_h_2 = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            self.tau_m_o = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            #self.tau_m_o = torch.Tensor([self.tau_m])
        else:
            self.tau_m_h_1 = nn.Parameter(torch.Tensor(self.num_hidden))
            self.tau_m_h_2 = nn.Parameter(torch.Tensor(self.num_hidden))
            nn.init.normal_(self.tau_m_h_1, 0, 1)
            nn.init.normal_(self.tau_m_h_2, 0, 1)
            self.tau_m_o = nn.Parameter(torch.Tensor(self.num_output))
            nn.init.normal_(self.tau_m_o, 0, 1)

        self.tau_adp_1 = nn.Parameter(torch.Tensor(self.num_hidden))
        self.tau_adp_2 = nn.Parameter(torch.Tensor(self.num_hidden))
        self.tau_adp_o = nn.Parameter(torch.Tensor(self.num_output))
        nn.init.normal_(self.tau_adp_1, 0, 1)  # check this
        nn.init.normal_(self.tau_adp_2, 0, 1)
        nn.init.normal_(self.tau_adp_o, 0, 1)

        self.tau_1 = [self.tau_m_h_1, self.tau_adp_1]
        self.tau_2 = [self.tau_m_h_2, self.tau_adp_2]

        self.fc_ih = nn.Linear(self.num_input, self.num_hidden, bias=False)
        self.fc_h1h1 = nn.Linear(self.num_hidden, self.num_hidden, bias=False)
        self.fc_h1h2 = nn.Linear(self.num_hidden, self.num_hidden, bias=False)
        self.fc_h2h2 = nn.Linear(self.num_hidden, self.num_hidden, bias=False)
        self.fc_ho = nn.Linear(self.num_hidden, self.num_output, bias=False)

        if self.debug == True:
            self.alif_state = {}
            self.alif_state['thresh_1'] = torch.zeros(
                self.win, self.batch_size, self.num_hidden, device=self.device)
            self.alif_state['thresh_2'] = torch.zeros(
                self.win, self.batch_size, self.num_hidden, device=self.device)
            self.alif_counter = 0

    def thresh_logger(self, B):
        if self.debug == True:
            if self.alif_counter % 2 == 0:
                self.alif_state['thresh_1'][self.step, :, :] = B
            else:
                self.alif_state['thresh_2'][self.step, :, :] = B
            self.alif_counter += 1

    def forward(self, input):

        h1_mem, h1_spike, h2_mem, h2_spike, o_mem, o_spike = self.init_state()

        d1 = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        d2 = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        o_sumspike = output_mot = torch.zeros(self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0)  # for spike-regularization

        for step in range(self.win):

            self.step = step

            x = input[:, self.step, :]

            i_spike = x.view(self.batch_size, -1)

            h1_mem, h1_spike, d1 = self.mem_update_rnn(
                i_spike, h1_spike, h1_mem, d1, self.fc_ih, self.fc_h1h1, self.tau_1)
            h2_mem, h2_spike, d2 = self.mem_update_rnn(
                h1_spike, h2_spike, h2_mem, d2, self.fc_h1h2, self.fc_h2h2, self.tau_2)
            o_mem, o_spike = self.mem_update_out(h2_spike, o_spike, o_mem)

            self.h_sumspike = self.h_sumspike + (h1_spike.sum() + h2_spike.sum())/2
            o_sumspike = o_sumspike + o_spike

            self.logger(i_spike.detach(), h1_mem.detach(), h1_spike.detach(),
                        h2_mem.detach(), h2_spike.detach(), o_mem.detach(), o_spike.detach())

            output_mot = output_mot + F.softmax(o_mem, dim=1)

        output_sum = o_sumspike / (self.win)

        return output_sum, output_mot

    def mem_update_out(self, i_spike, o_spike, mem):
        alpha = torch.sigmoid(self.tau_m_o).to(self.device)
        mem = mem * alpha * (1 - o_spike) + self.fc_ho(i_spike)
        o_spike = self.act_fun(mem-self.output_thresh)
        #mem = mem*(mem < self.thresh)
        return mem, o_spike

    def mem_update_rnn(self, i_spike, o_spike, mem, d, op_input, op_recurrent, tau_m):
        alpha = torch.sigmoid(tau_m[0]).to(self.device)
        ro = torch.sigmoid(tau_m[1]).to(self.device)

        d = ro*d + (1-ro)*(o_spike)
        B = self.thresh + 1.84*d

        a = op_input(i_spike)  # process spikes from input
        b = op_recurrent(o_spike)  # process recurrent spikes

        mem = mem * alpha + (1-alpha)*(a+b) - B*o_spike
        o_spike = self.act_fun(mem-B)

        self.thresh_logger(B.detach())

        #mem = mem*(mem < B)
        return mem, o_spike, d


'''
    # Bojian's ALIF implementation

    def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
        alpha = torch.exp(-1. * dt / tau_m).cuda()
        ro = torch.exp(-1. * dt / tau_adp).cuda()
        if isAdapt:
            beta = 1.84
        else:
            beta = 0.

        b = ro * b + (1 - ro) * spike
        B = b_j0 + beta * b

        mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
        inputs_ = mem - B
        spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
        return mem, spike, B, b
    '''
