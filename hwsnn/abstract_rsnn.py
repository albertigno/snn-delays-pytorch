import os
import torch
import torch.nn as nn
from hwsnn.activation_functions import ActFunStep, ActFunFastSigmoid, ActFunMultiGaussian
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import json
from torch.optim.lr_scheduler import StepLR

CHECKPOINT_PATH = os.path.join(os.environ.get('PYTHON_DRIVE_PATH'), 'checkpoints')

class Abstract_SNN():

    def __init__(self, dataset='nmnist', num_hidden=256, thresh=0.3, tau_m='adp', win=50, surr='step', loss_fn='mot', batch_size=256, device='cuda', debug=False):
        '''
        Implementation of an SNN, fixed structure, without delays:
        Arguments:
            dataset used: name of the dataset: mnmist, smnist, shd, DVSGesture_32, DVSGesture, custom_{num_input}_{num_outputs}_{num_training_samples}
            num_hidden: number of neuron in the hidden layers
            thresh: neuron's threshold
            tau_m: 'adp' for trainable tau_m, or a float number to make it fixed, untrainable
            win: number of timesteps per sample. set same as the one used in your dataset
            surr: surrogate function used: can be either (1) 'step', (2) 'fs': fast sigmoid, (3): 'mg': multi-gaussian
            loss_fn: loss function can be either (1) mot: max membrane over time, (2) sum of spikes (3) prediction
            batch_size
            device: cuda or gpu
            debug: boolean, set True if you want to record internal states for all layers (membrane and spikes)
        '''

        super(Abstract_SNN, self).__init__()

        self.debug = debug

        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.win = win
        self.device = device
        self.thresh = thresh
        self.dataset = dataset
        self.tau_m = tau_m
        self.surr = surr
        self.loss_fn = loss_fn
        self.info = {} # use this to store some useful info that you want to record for the save functionality

        self.epoch = 0

        self.acc = list()  # stores accuracy every time test() is called

        self.train_loss = list()
        self.test_loss = list()
        self.test_spk_count = list()

        self.handle_info_surr_dataset()
        self.define_operations()

        self.base_params = [getattr(self, name.split('.')[0]).weight for name,
                            _ in self.state_dict().items() if (name[0] == 'f' or name[0] == 'r')]
        self.base_params_names = [name.split('.')[0] for name, _ in self.state_dict(
        ).items() if (name[0] == 'f' or name[0] == 'r')]

        self.tau_params = [getattr(self, name.split('.')[0])
                           for name, _ in self.state_dict().items() if name[0] == 't']
        self.tau_params_names = [name for name, _ in self.state_dict().items() if name[0] == 't']

        self.modelname = '{}{}_{}_{}.t7'.format(self.dataset, self.win, str(type(self)).split('.')[-1][:-2], self.num_hidden) 


    def handle_info_surr_dataset(self):

        '''
        here we set up the input/outputs depending on the dataset used, and other initialization
        '''

        if self.surr == 'step':
            self.act_fun = ActFunStep.apply
        elif self.surr == 'fs':
            self.act_fun = ActFunFastSigmoid.apply
        elif self.surr == 'mg':
            self.act_fun = ActFunMultiGaussian.apply

        if self.dataset == 'nmnist':
            self.num_train_samples = 60000
            self.num_input = 34*34*2
            self.num_output = 10
        if self.dataset == 'smnist':
            self.num_train_samples = 60000
            self.num_input = 99
            self.num_output = 10
        if self.dataset == 'shd':
            self.num_train_samples = 8156
            self.num_input = 700
            self.num_output = 20
        if self.dataset == 'ibmgestures_32':
            self.num_train_samples = 1076
            self.num_input = 2048
            self.num_output = 11
        if self.dataset == 'ibmgestures':
            self.num_input = 32768
            self.num_train_samples = 1076
            self.num_output = 11
        if self.dataset.split('_')[0] == 'custom':
            self.num_input = int(self.dataset.split('_')[1])
            self.num_output = int(self.dataset.split('_')[2])
            self.num_train_samples = int(self.dataset.split('_')[3])
            
        if self.loss_fn == 'sum':
            self.criterion = nn.MSELoss()
            self.output_thresh = self.thresh
        elif self.loss_fn == 'mot':
            self.criterion = nn.CrossEntropyLoss()
            self.output_thresh = 1e6 # if mot, output neurons never fire
        if self.loss_fn == 'prediction':
            self.criterion = nn.MSELoss()
            self.output_thresh = 1e6

    def propagate(self, images, labels):
        '''
        a propagation of a single batch is handled here
        it will depend on the loss function used        
        '''
        if self.loss_fn == 'mot':
            _, reference = torch.max(labels.data, 1)
            _, outputs_mot = self(images)
            return outputs_mot, reference
        elif self.loss_fn == 'prediction':
            self(images)
            mx, _ = torch.max(self.mem_state['output'][int(0.8*self.win):,:,0], dim=0)
            return mx, labels                
        outputs_sum, _ = self(images)
        return outputs_sum, labels

    def train_step(self, train_loader=None, optimizer=None, spkreg=0.0, l1_reg = 0.0, dropout=0.0, verbose=True):
        '''
        training of one epoch (over the whole dataset)
        arguments:
            train_loader: a pytorch dataloader
            optimizer: a pytorch optimizer can be:
                - 
                -
            spkreg: penalty for spiking activity
            l1_reg: l1 regularizer
            dropout: percentage of randomly dropped spikes (applied to the input) 

        '''
        total_loss_train = 0
        running_loss = 0
        total = 0

        dropout = torch.nn.Dropout(p=dropout, inplace=False)

        num_iter = self.num_train_samples // self.batch_size
        sr = spkreg/self.win

        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()

            # dropout
            images = dropout(images.float())

            images = images > 0
            
            images = images.view(self.batch_size, self.win, -1).float().squeeze().to(self.device)
            labels = labels.float().squeeze().to(self.device)

            #start_time = time.time()
            outputs, reference = self.propagate(images, labels)
            #print('Time elasped forward: ', time.time() - start_time)

            #_, predicted = torch.max(outputs.data, 1)
            
            d_weights = torch.cat([x.view(-1) for x in self.base_params[1:]])
            l1_loss = torch.norm(d_weights,1) / len(d_weights)

            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)

            loss = self.criterion(outputs, reference) + sr*spk_count + l1_reg*l1_loss
            running_loss += loss.detach().item()
            total_loss_train += loss.detach().item()
            total += labels.size(0)
            loss.backward()

            optimizer.step()
            
            if verbose:
                if num_iter>=3:
                    if (i + 1) % int(num_iter/3.0) == 0:
                        print('Step [%d/%d], Loss: %.5f'
                            % (i + 1, self.num_train_samples // self.batch_size, running_loss))
                else:
                    if (i + 1) % int(num_iter) == 0:
                        print('Step [%d/%d], Loss: %.5f'
                            % (i + 1, self.num_train_samples // self.batch_size, running_loss))    

            running_loss = 0           

        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / num_iter])

    def train_step_tr(self, train_loader=None, optimizer=None, criterion=None, spkreg=0.0, depth=5, K=None, last=False):

        '''
        same as train step but for truncated BPTT
        possibly outdated
        TO DO: merge with train_step
        '''

        total_loss_train = 0
        running_loss = 0
        truncated_loss = 0
        total = 0

        num_iter = self.num_train_samples // self.batch_size
        sr = spkreg/self.win

        time_window = self.win
        self.win = depth

        if last:
            steps = [time_window-1]
        else:
            if K == None:
                steps = (np.arange(0, time_window+1, depth)-1)[1:]
            else:
                steps = (np.arange(0, time_window+1, time_window//K)-1)[1:]
                steps = [step for step in steps if (step >= depth-1)]

        for i, (images, labels) in enumerate(train_loader):
            self.zero_grad()
            optimizer.zero_grad()
            images = images > 0
            images = images.float().squeeze().to(self.device)
            labels = labels.float().to(self.device)

            for step in steps:

                self.zero_grad()
                optimizer.zero_grad()

                outputs = self.forward(images[:, step-depth+1:step+1, :])

                truncated_loss = criterion(outputs, labels)

                truncated_loss.backward()

                optimizer.step()  # parameter update

            total_loss_train += truncated_loss.detach().item()

            total += labels.size(0)
            running_loss += truncated_loss.item()

            if (i + 1) % int(num_iter/3.0) == 0:
                print('Step [%d/%d], Loss: %.5f'
                      % (i + 1, self.num_train_samples // self.batch_size, running_loss))
                running_loss = 0

        self.epoch = self.epoch + 1
        self.train_loss.append([self.epoch, total_loss_train / total])

        self.win = time_window

    def test(self, test_loader=None, dropout=0.0):

        self.save_model() # every time we call test() the model is saved 

        correct = 0
        total = 0
        total_loss_test = 0
        total_spk_count = 0

        dropout = torch.nn.Dropout(p=dropout, inplace=False)

        # snn_cpu = RSNN() # copy of self, doing this to always evaluate on cpu
        #snn_cpu = type(self)()
        #snn_cpu.load_model('rsnn', batch_size= self.batch_size)

        for i, (images, labels) in enumerate(test_loader):

            images = dropout(images.float())

            images = images > 0

            images = images.view(self.batch_size, self.win, -1).float().squeeze().to(self.device)
            labels = labels.float().to(self.device)
            #outputs = snn_cpu(images)
            #spk_count = snn_cpu.h_sumspike / (self.batch_size * self.num_hidden)

            outputs, reference = self.propagate(images, labels)
            #outputs = self(images)
            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)

            loss = self.criterion(outputs, reference)

            _, predicted = torch.max(outputs.data, 1)
            _, reference = torch.max(labels.data, 1)

            total += labels.size(0)
            correct += (predicted == reference).sum()
            total_loss_test += loss.detach().item()
            total_spk_count += spk_count.detach()

        acc = 100. * float(correct) / float(total)

        # try to improve this
        if self.acc == []:
            self.acc.append([self.epoch, acc])
            self.test_loss.append([self.epoch, total_loss_test / i])
        else:
            if self.acc[-1][0] < self.epoch:
                self.acc.append([self.epoch, acc])
                self.test_loss.append([self.epoch, total_loss_test / i])

        if self.test_spk_count == []:
            self.test_spk_count.append([self.epoch, total_spk_count.detach().item() * (self.batch_size / total)])
        else:
            if self.test_spk_count[-1][0] < self.epoch:
                self.test_spk_count.append(
                    [self.epoch, total_spk_count.detach().item() * (self.batch_size / total)])

        print('Test Loss: {}'.format(total_loss_test / i))
        print('Avg spk_count per neuron for all {} timesteps {}'.format(
            self.win, total_spk_count * (self.batch_size / total)))
        print('Test Accuracy of the model on the test samples: %.3f' % (acc))    

    def conf_matrix(self, test_loader=None, labels=None, criterion=nn.MSELoss()):

        '''
        probably outdated
        '''

        self.save_model()
        snn_cpu = Abstract_SNN()  # copy of self, doing this to always evaluate on cpu
        snn_cpu.load_model('rsnn', batch_size=self.batch_size)

        all_preds = list()
        all_refs = list()

        for images, labels in test_loader:
            images = images.float()
            labels = labels.float()
            outputs = snn_cpu(images)
            spk_count = self.h_sumspike / (self.batch_size * self.num_hidden)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            _, reference = torch.max(labels.data, 1)

            all_preds = all_preds + list(predicted.numpy())
            all_refs = all_refs + list(reference.numpy())

        print(confusion_matrix(all_refs, all_preds))

    def save_model(self, modelname='rsnn', directory = ''):

        if 'delays' not in str(type(self)):
            kwargs = {'dataset': self.dataset, 'num_hidden': self.num_hidden, 'thresh': self.thresh, 'tau_m': self.tau_m,
                      'win': self.win, 'surr': self.surr, 'loss_fn': self.loss_fn, 'batch_size': None, 'device': None}
        else:
            hidden = (self.num_hidden, self.n_layers, self.deepnet_type)
            kwargs = {'dataset': self.dataset, 'hidden': hidden, 'delay': self.delay, 'thresh': self.thresh, 'tau_m': self.tau_m,
                      'win': self.win, 'surr': self.surr, 'loss_fn': self.loss_fn, 'batch_size': None, 'device': None}

        state = {
            'type': type(self),
            'net': self.state_dict(),
            'epoch': self.epoch,
            'acc_record': self.acc,
            'train_loss': self.train_loss,
            'test_loss': self.test_loss,
            'test_spk': self.test_spk_count,
            'self.info': self.info,
            'kwargs': kwargs
        }

        model_path = os.path.join(CHECKPOINT_PATH, directory)

        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        torch.save(state, os.path.join(CHECKPOINT_PATH, directory, modelname),  _use_new_zipfile_serialization=False)

    def save_to_numpy(self, directory='default'):
        layers_location = os.path.join(CHECKPOINT_PATH, directory)

        if not os.path.isdir(layers_location):
            os.mkdir(layers_location)

        weights_biases = []
        snn_state_dict = self.state_dict()

        with open(layers_location+'/model_info', 'a') as logs:
            spk = self.test_spk_count[-1][1].detach().cpu().numpy()
            logs.write("avg spk neuron/sample {}".format(spk))
            logs.write("\navg spk neuron/timestep {}".format(spk*(self.num_hidden/self.win)))

        for k in snn_state_dict:
            np.savez(layers_location+'/'+k, snn_state_dict[k].data.cpu().numpy())
            weights_biases.append(snn_state_dict[k].data.cpu().numpy())

    def lr_scheduler(self, optimizer, lr_decay_epoch=1, lr_decay=0.98):
        """Decay learning rate by a factor of 0.98 every lr_decay_epoch epochs."""

        if self.epoch % lr_decay_epoch == 0 and self.epoch > 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_decay
                #param_group['lr'] = param_group['lr'] * 0.95

        return optimizer

    def plot_loss(self):

        test_loss = np.array(self.test_loss)
        train_loss = np.array(self.train_loss)
        fig = plt.figure()
        plt.plot(train_loss[:, 0], train_loss[:, 1], label='train loss')
        plt.plot(test_loss[:, 0], test_loss[:, 1], label='test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()

        #return fig

    @staticmethod
    def plot_per_epoch(data, ax=None, label=''):
        '''
        data is either snn.train_loss, snn.test_loss, snn.acc, snn.test_spk_count
        '''
        if ax is None:
            ax = plt.gca() # get current axis
        data = np.array(data)
        ax.plot(data[:, 0], data[:, 1], label=label)
        ax.set_xlabel('epoch')
        ax.legend()
        return ax

        #return fig


    def plot(self, w, mode='histogram', title='', xlabel='', ylabel='', ax=None):

        if ax is None:
            ax = plt.gca() # get current axis

        try:
            w = w.weight.data.cpu().numpy()
        except:
            w = w.data.cpu().numpy()

        vmin = np.min(w)
        vmax = np.max(w)

        if mode == 'histogram':
            # w = list(w.reshape(1, -1)[0])
            # n, bins, fig = plt.hist(w, bins=200)
            # else:
            ax = sns.histplot(w.reshape(1, -1)[0], bins=200)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(title, fontsize=16)
            return ax
        elif mode == 'matrix':

            c = 'RdBu'
            ax.imshow(w, cmap=c, vmin=vmin, vmax=vmax)
            ax.set_xlabel('input', fontsize=14)
            ax.set_ylabel('output', fontsize=14)
            ax.set_title('weights', fontsize=16)
            return ax

    @staticmethod
    def square(num):
        '''
        get two closest factors of num so we can plot a vector of length num as an square-ish matrix
        '''
        factor1 = [x for x in range(1, num+1) if num % x == 0]
        factor2 = [int(num/x) for x in factor1]
        idx = np.argmin(np.abs(np.array(factor2) - np.array(factor1)))
        return factor1[idx], factor2[idx]

    def plot_per_neuron(self, w, n_cols=3, num_channels=1):
        '''
        randomly sample weight per Neuron
        '''

        w = w.weight.data.cpu().numpy()

        sample_idx = np.random.choice(np.arange(w.shape[0]), n_cols**2)

        sample = w[sample_idx, :]

        s = int(w.shape[1]/num_channels)

        a, b = self.square(s)

        for i, x in enumerate(sample):
            plt.subplot(n_cols, n_cols, i+1)
            plt.imshow(x[:s].reshape(a, b), cmap='RdBu')

    def plot_distributions(self, mode='weights'):

        if mode == 'weights':
            params_names = self.base_params_names
            params = self.base_params
        elif mode == 'taus':
            params_names = self.tau_params_names
            params = self.tau_params
            #params = [-1/torch.log(torch.sigmoid(x)) for x in self.tau_params]
            
        else:
            print('mode either weight or taus')

        c = len(params_names)
        fig = plt.figure(figsize=(7, 7))
        for i, name in enumerate(params_names):
            plt.subplot(c, 1, i+1)
            self.plot(params[i], title=name)
        plt.tight_layout()
        return fig

    def quantize_weights(self, bits):

        def reduce_precision(weights, bits):
            scale = (1+bits)*(weights.max()-weights.min())/(2*bits+3)
            m = scale*torch.round((weights/scale)*2**bits)/(2**bits)
            return m

        with torch.no_grad():
            self.fc_hh.weight.data = torch.nn.Parameter(reduce_precision(self.fc_hh.weight.data, bits))
            self.fc_ih.weight.data = torch.nn.Parameter(reduce_precision(self.fc_ih.weight.data, bits))
            self.fc_ho.weight.data = torch.nn.Parameter(reduce_precision(self.fc_ho.weight.data, bits))

    def prune_weights(self, percentage):
        pass

    def mask_weights(self, layer, mask, override=False, trainable=True):
        # print(mask)
        # print(self.fc_hh.weight.data)
        if layer.weight.data.shape == mask.shape:
            new_weight = mask if override else layer.weight.data * mask
            layer.weight = torch.nn.Parameter(new_weight, requires_grad=trainable)

        else:
            print('Mask weights failed: dimension mismatch')

    @staticmethod
    def animation(x, cmap='RdBu'):
        '''
        animates a [time, x, y] matrix
        '''

        fig = plt.figure(figsize=(9, 9))

        vmax = np.max(x)
        vmin = np.min(x)

        im = plt.imshow(x[0, :, :], cmap=cmap, vmax=vmax, vmin=vmin)

        def animate(frame_num):
            im.set_data(x[frame_num, :, :])
            im.axes.set_title(str(frame_num))

        interval = 2000 / len(x)  # 2 seconds

        anim = FuncAnimation(fig, animate, frames=len(x), interval=interval)

        return anim


class Abstract_SNN_Delays(Abstract_SNN):

    def __init__(self, dataset='nmnist', hidden=(256, 2, 'r'), delay=None, thresh=0.3, reset_to_zero = True, tau_m='adp', win=50, surr='step', loss_fn='mot', batch_size=256, device='cuda', debug=False):
        super(Abstract_SNN, self).__init__()

        self.debug = debug

        self.num_hidden = hidden[0]
        self.n_layers = hidden[1]
        self.deepnet_type = hidden[2]
        self.delay = delay

        self.batch_size = batch_size
        self.win = win
        self.device = device
        self.thresh = thresh
        self.dataset = dataset
        self.tau_m = tau_m
        self.surr = surr
        self.loss_fn = loss_fn
        self.info = {}
        self.epoch = 0

        if reset_to_zero:
            self.th_reset = self.thresh
        else:
            self.th_reset = 1e6

        self.acc = list()  # stores accuracy every time test() is called

        self.train_loss = list()
        self.test_loss = list()
        self.test_spk_count = list()

        self.handle_info_surr_dataset()
        self.define_operations()

        self.base_params = [getattr(self, name.split('.')[0]).weight for name,
                            _ in self.state_dict().items() if (name[0] == 'f' or name[0] == 'r')]
        self.base_params_names = [name.split('.')[0] for name, _ in self.state_dict(
        ).items() if (name[0] == 'f' or name[0] == 'r')]

        self.tau_params = [getattr(self, name.split('.')[0])
                           for name, _ in self.state_dict().items() if name[0] == 't']
        self.tau_params_names = [name for name, _ in self.state_dict().items() if name[0] == 't']

        # define which update function to use
        if self.deepnet_type == 'f':
            self.mem_update_fn = self.mem_update
        elif self.deepnet_type == 'r':
            self.mem_update_fn = self.mem_update_rnn

    def define_modelname(self):

        self.modelname = '{}{}_{}_{}_l{}_{}d{}.t7'.format(self.dataset,
         self.win, str(type(self)).split('.')[2][:-2], self.n_layers, self.num_hidden, self.delay[0], self.delay[1]) 

    def define_tau_m(self):

        if self.tau_m != 'adp':
            self.tau_m_h = [nn.Parameter(torch.Tensor([self.tau_m]), requires_grad=False)
                            for i in range(self.n_layers+1)]
        else:

            # old method: gaussian initialization
            #mean = 0  # 0.83
            #std = 1  # 0.1
            #for i in range(self.n_layers):
            #    name = 'tau_m_'+str(i+1)
            #    setattr(self, name, nn.Parameter(torch.Tensor(self.num_hidden)))
            #    nn.init.normal_(getattr(self, name), mean, std)
            #name = 'tau_m_o'
            #setattr(self, name, nn.Parameter(torch.Tensor(self.num_output)))
            #nn.init.normal_(getattr(self, name), mean, std)           
            
            rate = 10/self.win # rate 1->10
            concentration = 2.0   
            for i in range(self.n_layers):
                name = 'tau_m_'+str(i+1)
                setattr(self, name, nn.Parameter(torch.distributions.gamma.Gamma(torch.ones(self.num_hidden) * concentration ,torch.ones(self.num_hidden) * rate ).sample()))
            setattr(self, 'tau_m_o', nn.Parameter(torch.distributions.gamma.Gamma(torch.ones(self.num_output) * concentration ,torch.ones(self.num_output) * rate ).sample()))

    def define_hidden_layer_names(self):

        # Define layer names
        self.h_names = []
        self.layer_names = []

        for layer in range(self.n_layers-1):
            n1 = self.deepnet_type + str(layer+1)
            n2 = self.deepnet_type + str(layer+2)
            if self.deepnet_type == 'r':
                self.h_names.append(n1 + '_' + n1)
            self.h_names.append(n1 + '_' + n2)
            self.layer_names.append(n1)

        if self.deepnet_type == 'r':
            self.h_names.append('r'+str(self.n_layers) + '_' + 'r'+str(self.n_layers))
        self.h_names.append(self.deepnet_type + str(self.n_layers) + '_o')
        self.layer_names.append(self.deepnet_type+str(self.n_layers))

    def define_delays(self):

        if self.delay != None:
            self.max_d = self.delay[0]
            self.stride = self.delay[1]
            # self.delays = np.linspace(0, self.max_d, self.num_d).astype(int)
            self.delays = list(range(0, self.max_d, self.stride))
            self.input_names = ['f0_id'+str(k) for k in self.delays]
        else:
            self.input_names = ['f0_i']
            self.max_d = 0
            self.delays = [0]

        print('delays: '+str(self.delays))

    def set_input_layer(self):

        for name in self.input_names:
            setattr(self, name, nn.Linear(self.num_input, self.num_hidden, bias=False))

    def set_hidden_layers(self):
        bias = False
        # set linear layers dynamically
        for name in self.h_names:
            r_name = name[:2]+'_'+name[:2]
            if name[-1] == 'o':
                if self.deepnet_type == 'r':
                    setattr(self, r_name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))
                setattr(self, name, nn.Linear(self.num_hidden, self.num_output, bias=bias))
            else:
                if self.deepnet_type == 'f':
                    setattr(self, name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))
                elif self.deepnet_type == 'r':
                    setattr(self, r_name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))
                    setattr(self, name, nn.Linear(self.num_hidden, self.num_hidden, bias=bias))

    def set_layer_lists(self):
        self.h_layers = [nn.Identity()]
        for name in self.h_names:
            self.h_layers.append(getattr(self, name))

        if self.tau_m == 'adp':
            self.tau_m_h = [getattr(self, name) for name in ['tau_m_'+str(i+1)
                                                             for i in range(self.n_layers)]]
            self.tau_m_h.append(self.tau_m_o)

    def define_operations(self):
        '''
        Creates the layers
        '''
        self.define_modelname()
        self.define_tau_m()
        self.define_hidden_layer_names()
        self.define_delays()
        self.set_input_layer()
        self.set_hidden_layers()
        self.set_layer_lists()

    def save_to_json_list(self, directory='default', multidelays=True):

        '''
        probably outdated
        '''

        def project_ih_weights(weights, delays):
            inh_synapses = []
            exc_synapses = []
            print('projecting ih weights...')
            for wi, w in enumerate(weights):
                for i in range(w.shape[1]):
                    for j in range(w.shape[0]):
                        if float(w[j, i]) != 0.0:
                            if float(w[j, i]) < 0.0:
                                inh_synapses.append([i, j, float(-1.0*w[j, i]), int(delays[wi]+1)])
                            else:
                                exc_synapses.append([i, j, float(w[j, i]), int(delays[wi]+1)])
            return inh_synapses, exc_synapses

        def project_weights(weights, delay=0):
            print('projecting hidden weights')
            inh_synapses = []
            exc_synapses = []
            for i in range(weights.shape[1]):
                for j in range(weights.shape[0]):
                    if float(weights[j, i]) < 0.0:
                        inh_synapses.append([i, j, float(-1.0*weights[j, i]), delay+1])
                    else:
                        exc_synapses.append([i, j, float(weights[j, i]), delay+1])
            return inh_synapses, exc_synapses

        layers_location = 'checkpoint/' + directory
        # delays = self.delays

        if not os.path.isdir(layers_location):
            os.mkdir(layers_location)

        weight_delay_dict = {}

        if multidelays:
            weights_ih = [getattr(self, 'f0_id'+str(d)).weight.data.detach().cpu().numpy()
                          for d in self.delays]
            inh, exc = project_ih_weights(weights_ih, self.delays)
        else:
            weights_ih = self.f0_i.weight.data.detach().cpu().numpy()
            inh, exc = project_weights(weights_ih)

        weight_delay_dict['f0_i'] = {'exc': exc, 'inh': inh}

        for name in self.h_names:
            h_weights = getattr(self, name).weight.data.detach().cpu().numpy()
            inh, exc = project_weights(h_weights)
            weight_delay_dict[name] = {'exc': exc, 'inh': inh}

        with open("{}.json".format(layers_location), 'w') as outfile:
            json.dump(weight_delay_dict, outfile)
