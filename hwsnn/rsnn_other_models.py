import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hwsnn.abstract_rsnn import Abstract_SNN, Abstract_SNN_Delays

class RSNN_d(nn.Module, Abstract_SNN_Delays):

    '''
    my first implementation of SNN with delays
    the delays are input-only, and all input delay has its own torch.parameter
    later I decided to merge all per-layer delays in the same torch.parameter
    '''

    def init_state(self):

        mems = {}
        spikes = {}

        for name in self.layer_names:
            mems[name] = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
            spikes[name] = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        o_mem = torch.zeros(self.batch_size, self.num_output, device=self.device)
        o_spike = torch.zeros(self.batch_size, self.num_output, device=self.device)

        return mems, spikes, o_mem, o_spike

    def forward(self, input):

        mems, spikes, o_mem, o_spike = self.init_state()
        self.o_sumspike = output_mot = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0)  # for spike-regularization

        extended_input = torch.zeros(self.batch_size, self.win+self.max_d,
                                     self.num_input, device=self.device)
        extended_input[:, self.max_d:, :] = input

        for step in range(self.max_d, self.win+self.max_d):

            delayed_x = [extended_input[:, step-d,
                                        :].view(self.batch_size, -1) for d in self.delays]
            prev_spikes = self.get_delayed_inputs(*delayed_x)  # input layer is propagated

            self.w_idx = 0
            self.tau_idx = 0

            for layer in self.layer_names:

                mems[layer], spikes[layer] = self.mem_update_fn(
                    prev_spikes, spikes[layer], mems[layer])
                prev_spikes = spikes[layer]
                self.h_sumspike = self.h_sumspike + prev_spikes.sum()

            o_mem, o_spike = self.mem_update(prev_spikes, o_spike, o_mem)

            self.o_sumspike = self.o_sumspike + o_spike

            output_mot = output_mot + F.softmax(o_mem, dim=1)

        self.h_sumspike = self.h_sumspike / self.n_layers

        output_sum = self.o_sumspike / (self.win)

        return output_sum, output_mot

    def forward_residual(self, input):

        mems, spikes, o_mem, o_spike = self.init_state()
        self.o_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0)  # for spike-regularization

        extended_input = torch.zeros(self.batch_size, self.win+self.max_d,
                                     self.num_input, device=self.device)
        extended_input[:, self.max_d:, :] = input

        # residual_spikes = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        for step in range(self.max_d, self.win+self.max_d):

            delayed_x = [extended_input[:, step-d,
                                        :].view(self.batch_size, -1) for d in self.delays]
            prev_spikes = self.get_delayed_inputs(*delayed_x)  # input layer is propagated

            self.w_idx = 0
            self.tau_idx = 0

            for i, layer in enumerate(self.layer_names):

                mems[layer], spikes[layer] = self.mem_update_fn(
                    prev_spikes, spikes[layer], mems[layer])
                if ((i+2) % 3) == 0:
                    prev_spikes = spikes[layer] + spikes[self.layer_names[i-1]]
                else:
                    prev_spikes = spikes[layer]
                self.h_sumspike = self.h_sumspike + prev_spikes.sum()

            o_mem, o_spike = self.mem_update(prev_spikes, o_spike, o_mem)

            self.o_sumspike = self.o_sumspike + o_spike

        self.h_sumspike = self.h_sumspike / self.n_layers
        outputs = self.o_sumspike / (self.win)

        return outputs

    def get_delayed_inputs(self, *args):

        input_activations = torch.zeros(self.batch_size, self.num_hidden, device=self.device)

        for i, x in enumerate(args):
            input_activations = input_activations + getattr(self, self.input_names[i])(x)

        return input_activations

class RSNN_FWP(nn.Module, Abstract_SNN):

    '''
    Attempt to implement forward propagation
    '''

    def forward(self, input):
        
        h_mem = h_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_output, device=self.device)
        self.h_sumspike = torch.tensor(0.0) # for spike-regularization
        
        for step in range(self.win):

            x = input[:, step, :]
            
            i_spike = x.view(self.batch_size, -1)
            
            #print(i_spike.shape)

            h_mem, h_spike = self.mem_update_rnn(i_spike, h_spike, h_mem)            
            o_mem, o_spike = self.mem_update(h_spike, o_spike, o_mem)
            
            self.h_sumspike = self.h_sumspike + h_spike.sum()
            o_sumspike = o_sumspike + o_spike
            
        outputs = o_sumspike / (self.win)

        return outputs      
 
    def forward_mem(self, input):
        
        h_mem = h_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = o_softmem = torch.zeros(self.batch_size, self.num_output, device=self.device)
        o_softmem = torch.zeros(self.win, self.batch_size, self.num_output, device=self.device)
        
        self.h_sumspike = torch.tensor(0.0) # for spike-regularization

        for step in range(self.win):

            x = input[:, step, :]
            
            i_spike = x.view(self.batch_size, -1)
            
            #print(i_spike.shape)

            h_mem, h_spike = self.mem_update_rnn(i_spike, h_spike, h_mem)            
            o_mem = self.mem_update_no_thresh(h_spike, o_mem)
            
            self.h_sumspike = self.h_sumspike + h_spike.sum()
            
            #o_softmem[step] = nn.functional.log_softmax(o_mem, dim=-1)
            o_softmem[step] = o_mem
        
        return o_softmem

    def mem_update_no_thresh(self, i_spike, mem):
        alpha = torch.exp(-1. / self.tau_m_o).to(self.device)
        mem = mem * alpha  + self.fc_ho(i_spike)
        return mem    
    
    
    def forward_fwp(self, input):
        
        h_mem = h_spike = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        o_mem = o_spike = torch.zeros(self.batch_size, self.num_output, device=self.device)
        
        i_spike = input.view(self.batch_size, -1)

        h_mem, h_spike = self.mem_update_rnn(i_spike, h_spike, h_mem)            
        o_mem, o_spike = self.mem_update(h_spike, o_spike, o_mem)

        return o_spike      

    def train_step(self, train_loader=None, optimizer=None, criterion=None, num_samples=0, spkreg=0.0):
        
        total_loss_train = 0
        running_loss = 0
        total = 0
        
        num_iter = num_samples // self.batch_size 
        sr = spkreg/self.win
        
        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            images = images.float().to(self.device)
            labels = labels.float().to(self.device)
            outputs = self(images)
            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)
            loss = criterion(outputs, labels) + sr*spk_count
            running_loss += loss.item()
            total_loss_train += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()

            if (i + 1) % int(num_iter/3.0) == 0:
                print('Step [%d/%d], Loss: %.5f'
                      % (i + 1, num_samples // self.batch_size, running_loss))
                running_loss = 0
                
        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / total]) 
 
    def train_step_mem(self, train_loader=None, optimizer=None, criterion=None, num_samples=0, spkreg=0.0):
        
        total_loss_train = 0
        running_loss = 0
        total = 0
        
        num_iter = num_samples // self.batch_size 
        sr = spkreg/self.win
        
        log_softmax_fn = nn.LogSoftmax(dim=1)
        
        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            images = images.float().to(self.device)
            labels = labels.float().to(self.device)
            outputs = self.forward_mem(images)
            
            m, _ = torch.max(outputs.data, 0)
            
            _, predicted = torch.max(m, 1)
            
            _, reference = torch.max(labels.data, 1)
            
            #print('-------')
            print(predicted[:10])
            #print(reference[:10])

            _, predicted2 = torch.max(log_softmax_fn(m), 1)
                
            print(predicted2[:10])
                
            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)
            loss = criterion(log_softmax_fn(m), reference) + sr*spk_count
            running_loss += loss.item()
            total_loss_train += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()

            if (i + 1) % int(num_iter/3.0) == 0:
                print('Step [%d/%d], Loss: %.5f'
                      % (i + 1, num_samples // self.batch_size, running_loss))
                running_loss = 0
                
        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / total]) 
        
    def test_mem(self, test_loader = None, criterion=None):
        
        self.save_model()
        
        correct = 0
        total = 0
        total_loss_test = 0
        total_spk_count = 0
        
        #snn_cpu = RSNN() # copy of self, doing this to always evaluate on cpu
        snn_cpu = type(self)()
        snn_cpu.load_model('rsnn', batch_size= self.batch_size)
        
        log_softmax_fn = nn.LogSoftmax(dim=1)
        
        for images, labels in test_loader:
            images = images.float()
            labels = labels.float()
            outputs = snn_cpu.forward_mem(images)

            m, _ = torch.max(outputs.data, 0)
            _, predicted = torch.max(m, 1)
            _, reference = torch.max(labels.data, 1)         
            
            spk_count = snn_cpu.h_sumspike / (self.batch_size * self.num_hidden)
            loss = criterion(log_softmax_fn(m), reference)
            
            total += labels.size(0)
            correct += (predicted == reference).sum()
            total_loss_test += loss.item() 
            total_spk_count += spk_count
            
        acc = 100. * float(correct) / float(total)
        
        # try to improve this
        if self.acc == []:
            self.acc.append([self.epoch, acc]) 
            self.test_loss.append([self.epoch, total_loss_test / total])
        else:
            if self.acc[-1][0] < self.epoch:
                self.acc.append([self.epoch, acc]) 
                self.test_loss.append([self.epoch, total_loss_test / total])               

        if self.test_spk_count == []:
            self.test_spk_count.append([self.epoch, total_spk_count * (self.batch_size / total)]) 
        else:
            if self.test_spk_count[-1][0] < self.epoch:
                self.test_spk_count.append([self.epoch, total_spk_count * (self.batch_size / total)])                 
                
        print('avg spk_count per neuron for all {} timesteps {}'.format(self.win, total_spk_count * (self.batch_size / total)))   
        print('Test Accuracy of the model on the test samples: %.3f' % (acc))        

    def train_step_fwp(self, train_loader=None, optimizer=None, criterion=None, num_samples=0, spkreg=0.0):
        
        total_loss_train = 0
        running_loss = 0
        total = 0
        
        num_iter = num_samples // self.batch_size 
        sr = spkreg/self.win
        
        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            
            images = images.float().to(self.device)
            labels = labels.float().to(self.device)
            
            for step in range(self.win):
                #self.zero_grad()
                #optimizer.zero_grad()                
                outputs = self.forward_fwp(images[:, step, :])
                loss = criterion(outputs, labels)
                total_loss_train += loss.item()
                loss.backward()
                optimizer.step()
            
            total += labels.size(0) 
            running_loss += loss.item() 

            if (i + 1) % int(num_iter/3.0) == 0:
                print('Step [%d/%d], Loss: %.5f'
                      % (i + 1, num_samples // self.batch_size, running_loss))
                running_loss = 0
                
        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / total])   
        
    def train_step_tr(self, train_loader=None, optimizer=None, criterion=None, num_samples=0, spkreg=0.0, depth=5, K = None, last = False):
        
        total_loss_train = 0
        running_loss = 0
        total = 0
        
        num_iter = num_samples // self.batch_size 
        sr = spkreg/self.win

        time_window = self.win
        self.win = depth
        
        if last:
            steps = [time_window-1]
        else:
            if K==None:
                steps = (np.arange(0,time_window+1,depth)-1)[1:]
            else:
                steps = (np.arange(0,time_window+1,time_window//K)-1)[1:]
                steps = [step for step in steps if (step>=depth-1)]
                
        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            images = images.float().to(self.device)
            labels = labels.float().to(self.device)

            for step in steps:

                self.zero_grad()
                optimizer.zero_grad()    

                outputs = self.forward(images[:, step-depth+1:step+1, :])
                
                truncated_loss = criterion(outputs, labels)

                truncated_loss.backward()

                optimizer.step() # parameter update

            total_loss_train += truncated_loss.item()

            total += labels.size(0) 
            running_loss += truncated_loss.item() 

            if (i + 1) % int(num_iter/3.0) == 0:
                print('Step [%d/%d], Loss: %.5f'
                      % (i + 1, num_samples // self.batch_size, running_loss))
                running_loss = 0
                
        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / total])           
        
        self.win = time_window    
        
        
class SNN_Autoencoder(nn.Module, Abstract_SNN):

    '''    
    Just worked on this for a couple of days but it looked promising
    '''

    def define_operations(self):
        
        if self.tau_m!='adp':
            self.tau_m_h = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            self.tau_m_o = nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
            #self.tau_m_o = torch.Tensor([self.tau_m])
        else:
            self.tau_m_h = nn.Parameter(torch.Tensor(self.num_hidden))
            nn.init.normal_(self.tau_m_h, 0.83, 0.1)
            self.tau_m_o = nn.Parameter(torch.Tensor(self.num_output))
            nn.init.normal_(self.tau_m_o, 0.83, 0.1)        
        
        self.h1=self.num_hidden
        self.h2=self.num_hidden//2
        self.h3=self.num_hidden//4
        
        self.fc_ih1 = nn.Linear(self.num_input, self.h1, bias= False)
        self.fc_h1h2 = nn.Linear(self.h1, self.h2, bias= False)
        self.fc_h1h3 = nn.Linear(self.h2, self.h3, bias= False)
        self.fc_h3h2 = nn.Linear(self.h3, self.h2, bias= False)
        self.fc_h2h1 = nn.Linear(self.h2, self.h1, bias= False)
        self.fc_h1o = nn.Linear(self.h1, self.num_input, bias= False)
        
        self.max_d = 5
        
    def forward_past(self, input):
        
        # encoder
        h1_mem = h1_spike = torch.zeros(self.batch_size, self.h1, device=self.device)
        h2_mem = h2_spike = torch.zeros(self.batch_size, self.h2, device=self.device)
        h3_mem = h3_spike = torch.zeros(self.batch_size, self.h3, device=self.device)
        
        # decoder
        h1d_mem = h1d_spike = torch.zeros(self.batch_size, self.h1, device=self.device)
        h2d_mem = h2d_spike = torch.zeros(self.batch_size, self.h2, device=self.device)
        #h3d_mem = h1d_spike = torch.zeros(self.batch_size, self.h3, device=self.device)
        
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_input, device=self.device)
        
        all_o_spike = torch.zeros(self.batch_size, self.win+self.max_d, self.num_input, device=self.device)
        
        self.h_sumspike = torch.tensor(0.0) # for spike-regularization
        
        extended_input = torch.zeros(self.batch_size, self.win+self.max_d, self.num_input, device=self.device) 
        extended_input[:, self.max_d:, :] = input
        
        for step in range(self.win+self.max_d):

            x = extended_input[:, step, :]
            
            i_spike = x.view(self.batch_size, -1)
            
            h1_mem, h1_spike = self.mem_update(i_spike, h1_spike, h1_mem, self.fc_ih1)
            h2_mem, h2_spike = self.mem_update(h1_spike, h2_spike, h2_mem, self.fc_h1h2)
            h3_mem, h3_spike = self.mem_update(h2_spike, h3_spike, h3_mem, self.fc_h1h3)
            h2d_mem, h2d_spike = self.mem_update(h3_spike, h2d_spike, h2d_mem, self.fc_h3h2)
            h1d_mem, h1d_spike = self.mem_update(h2d_spike, h1d_spike, h1d_mem, self.fc_h2h1)
            o_mem, o_spike = self.mem_update(h1d_spike, o_spike, o_mem, self.fc_h1o)
            
            all_o_spike[:, step, :] = o_spike
        
        return all_o_spike[:,self.max_d:,:]      

    def forward_future(self, input):
        
        # encoder
        h1_mem = h1_spike = torch.zeros(self.batch_size, self.h1, device=self.device)
        h2_mem = h2_spike = torch.zeros(self.batch_size, self.h2, device=self.device)
        h3_mem = h3_spike = torch.zeros(self.batch_size, self.h3, device=self.device)
        
        # decoder
        h1d_mem = h1d_spike = torch.zeros(self.batch_size, self.h1, device=self.device)
        h2d_mem = h2d_spike = torch.zeros(self.batch_size, self.h2, device=self.device)
        #h3d_mem = h1d_spike = torch.zeros(self.batch_size, self.h3, device=self.device)
        
        o_mem = o_spike = o_sumspike = torch.zeros(self.batch_size, self.num_input, device=self.device)
        
        all_o_spike = torch.zeros(self.batch_size, self.win-self.max_d, self.num_input, device=self.device)
        
        self.h_sumspike = torch.tensor(0.0) # for spike-regularization
                
        for step in range(self.win-self.max_d):

            x = input[:, step, :]
            
            i_spike = x.view(self.batch_size, -1)
            
            h1_mem, h1_spike = self.mem_update(i_spike, h1_spike, h1_mem, self.fc_ih1)
            h2_mem, h2_spike = self.mem_update(h1_spike, h2_spike, h2_mem, self.fc_h1h2)
            h3_mem, h3_spike = self.mem_update(h2_spike, h3_spike, h3_mem, self.fc_h1h3)
            h2d_mem, h2d_spike = self.mem_update(h3_spike, h2d_spike, h2d_mem, self.fc_h3h2)
            h1d_mem, h1d_spike = self.mem_update(h2d_spike, h1d_spike, h1d_mem, self.fc_h2h1)
            o_mem, o_spike = self.mem_update(h1d_spike, o_spike, o_mem, self.fc_h1o)
            
            all_o_spike[:, step, :] = o_spike
        
        return all_o_spike     
    
    def train_step_auto(self, train_loader=None, optimizer=None, criterion=None):
        
        total_loss_train = 0
        running_loss = 0
        total = 0
        
        num_iter = self.num_train_samples // self.batch_size 
               
        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            images = images>0
            images = images.view(self.batch_size,self.win,-1).float().squeeze().to(self.device)
            #labels = labels.float().squeeze().to(self.device)

            outputs = self.forward_past(images)
            
            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)
            loss = criterion(outputs, images)
            running_loss += loss.detach().item()
            total_loss_train += loss.detach().item()
            total += labels.size(0)
            loss.backward()
            
            # check that gradients are computed
            #grads = [getattr(self,name.split('.')[0]).weight.grad.detach() for name, _ in self.state_dict().items() if (name[0]=='f' or name[0]=='r')]
            #avg_grads = [torch.mean(q).cpu().numpy() for q in grads]
            #print(avg_grads)
            
            optimizer.step()

            if (i + 1) % int(num_iter/3.0) == 0:
                print('Step [%d/%d], Loss: %.5f'
                      % (i + 1, self.num_train_samples // self.batch_size, running_loss))
                running_loss = 0
                
        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / total])     

    def train_step_auto_future(self, train_loader=None, optimizer=None, criterion=None):
        
        total_loss_train = 0
        running_loss = 0
        total = 0
        
        num_iter = self.num_train_samples // self.batch_size 
               
        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            images = images>0
            images = images.view(self.batch_size,self.win,-1).float().squeeze().to(self.device)
            #labels = labels.float().squeeze().to(self.device)

            outputs = self.forward_future(images)
            
            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)
            loss = criterion(outputs, images[:,self.max_d:,:])
            running_loss += loss.detach().item()
            total_loss_train += loss.detach().item()
            total += labels.size(0)
            loss.backward()
            
            # check that gradients are computed
            #grads = [getattr(self,name.split('.')[0]).weight.grad.detach() for name, _ in self.state_dict().items() if (name[0]=='f' or name[0]=='r')]
            #avg_grads = [torch.mean(q).cpu().numpy() for q in grads]
            #print(avg_grads)
            
            optimizer.step()

            if (i + 1) % int(num_iter/3.0) == 0:
                print('Step [%d/%d], Loss: %.5f'
                      % (i + 1, self.num_train_samples // self.batch_size, running_loss))
                running_loss = 0
                
        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / total])             
        
    def mem_update(self, i_spike, o_spike, mem, op):
        alpha = torch.exp(-1. / self.tau_m_o).to(self.device)
        mem = mem * alpha * (1 - o_spike) + op(i_spike) - o_spike*self.vreset
        o_spike = self.act_fun(mem)
        mem = mem*(mem<self.thresh)
        return mem, o_spike    
    
    
    def plot_loss(self):
        
        train_loss = np.array(self.train_loss)        
        fig = plt.figure()
        plt.plot(train_loss[:,0], train_loss[:,1], label ='train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        
        return fig              
        
        