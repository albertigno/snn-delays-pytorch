from hwsnn.models.snn import SNN
from hwsnn.utils.hw_aware_utils import modify_weights
import torch.nn.functional as F
import torch

class SNN_STDP(SNN):


    def __init__(self, dataset='nmnist', structure=(256, 2), reset_to_zero=True, tau_m='normal', win=50,
                 loss_fn='mot', batch_size=1, device='cuda', debug=False):


        super(SNN, self).__init__()
        
        # gather keyword arguments for reproducibility in loaded models
        self.kwargs = locals()

        # Set attributes for inputs
        self.dataset = dataset
        self.connection_type = 'f'
        self.delay = None
        self.thresh = 0.3
        self.reset_to_zero = reset_to_zero
        self.tau_m = tau_m
        self.win = win
        self.surr = 'fs'
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.device = device
        self.debug = debug
        self.structure = structure

        # by default, inputs are binarized according to this threshold (see training/propagate)
        # set this to None if you want to allow floating inputs
        self.input2spike_th = 0.5

        # there are two ways to specify the network structure:
        # (1) fixed: tuple -> (neurons_per_hidden_layer, number_of_layers ) 
        # (2) flexible: list -> [n1, n2, n3, ...] with each n being the number of neurons per layers 1,2,3

        if type(structure) == tuple:
            self.num_neurons_list = [structure[0] for _ in range(structure[1])]
        elif type(structure) == list:
            self.num_neurons_list = structure
        self.num_layers = len(self.num_neurons_list)

        # Set other attributes
        self.epoch = 0  # Number of epochs, initialized as 0
        self.acc = [[self.epoch, None]]  # Stores accuracy every time test() method is called
        self.train_loss = []   # Stores loss during training
        self.test_loss = [[self.epoch, None]]   # Stores loss during testing
        self.test_spk_count = [[self.epoch, None]]  # Store spiking count during testing
        self.info = {}  # Store some useful info that you want to record for
        # the save functionality (save_to_numpy method)

        # Set information about dataset, loss function and surrogate gradient
        # function (attributes initialized as None)
        self.num_train_samples = None
        self.num_input = None
        self.num_output = None
        self.act_fun = None
        self.criterion = None
        self.output_thresh = None

        # propagation attributes
        self.h_sum_spike = None
        self.h_sum_spikes_per_layer = None

        # Functions used for updating
        self.update_mem_fn = None 
        self.alpha_fn = None

        # Set model name, delays, list of layers, .... ¿?
        # (attributes initialized as None)
        self.model_name = None

        # Initialization of the layer and projections (weights) names
        self.layer_names = list()
        self.proj_names = list()

        self.max_d = None
        self.stride = None
        self.delays = None
        # self.input_names = None

        self.h_layers = None
        self.tau_m_h = None

        self.delay_type = delay_type

        self.define_metaparameters()
        self.set_input_layer()
        self.set_hidden_layers()
        self.set_tau_m()
        self.set_layer_lists()
        self.define_model_name()

        # print a summary of the model
        print(self)


    def init_state(self):
        """
        Function to set the initial state of the network. It initialized the
        membrane potential, the spikes and the spikes extended with the delays
        of all the neurons in hidden and output layer to zero. Also, the dictionary
        to log these parameters for debug is initialized with its parameters as
        zeros.

        :return: A tuple with the values of the extended spikes,
        the membrane potential of the hidden layer, the spikes of the hidden
        layer, the membrane potential of the output layer and the spikes of
        the output layer.
        """
        mems = dict()
        spikes = dict()
        traces = dict()
        # extended_spikes = dict()
        setattr(self, 'mem_state', dict())
        setattr(self, 'spike_state', dict())
        setattr(self, 'postsyn_traces', dict())
        # self.mem_state = {}
        # self.spike_state = {}

        # Initialization of membrane potential and spikes for hidden layers
        for name, num_hidden in zip(self.layer_names, self.num_neurons_list):

            # if self.delay_type != 'only_input':
            #     extended_spikes[name] = torch.zeros(
            #         self.batch_size, self.win+self.max_d,
            #         num_hidden, device=self.device)
            mems[name] = torch.zeros(
                self.batch_size, num_hidden, device=self.device)
            traces[name] = torch.zeros(
                self.batch_size, num_hidden, device=self.device)
            spikes[name] = torch.zeros(
                self.batch_size, num_hidden, device=self.device)

            # Initialization of the dictionary to log the state of the
            # network if debug is activated
            if self.debug:

                self.spike_state['input'] = torch.zeros(
                    self.win, self.batch_size,
                    self.num_input, device=self.device)
                self.spike_state[name] = torch.zeros(
                    self.win, self.batch_size,
                    num_hidden, device=self.device)
                self.spike_state['output'] = torch.zeros(
                    self.win, self.batch_size,
                    self.num_output, device=self.device)
                
                self.postsyn_traces['input'] = torch.zeros(
                    self.win, self.batch_size,
                    self.num_input, device=self.device)
                self.postsyn_traces[name] = torch.zeros(
                    self.win, self.batch_size,
                    num_hidden, device=self.device)
                self.postsyn_traces['output'] = torch.zeros(
                    self.win, self.batch_size,
                    self.num_output, device=self.device)

                self.mem_state['output'] = torch.zeros(
                    self.win, self.batch_size,
                    self.num_output, device=self.device)              
                self.mem_state[name] = torch.zeros(
                    self.win, self.batch_size,
                    num_hidden, device=self.device)

        # Initialization of traces for input
        traces['input'] = torch.zeros(
                self.batch_size, self.num_input, device=self.device)

        # Initialization of membrane potential, spikes and traces of output layer
        o_mem = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        o_traces = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        o_spikes = torch.zeros(
            self.batch_size, self.num_output, device=self.device)

        return mems, spikes, traces, o_mem, o_spikes, o_traces


    def update_logger(self, *args):
        """
        Function to log the parameters if debug is activated. It creates a
        dictionary with the state of the neural network, recording the values
        of the spikes and membrane voltage for the input, hidden and output
        layers.

        This function takes as arguments the parameters of the network to log.
        """

        # Create the dictionary for logging
        if self.debug:
            x, mems, spikes, traces, o_mem, o_spike, o_traces = args

            self.spike_state['input'][self.step, :, :] = x
            self.postsyn_traces['input'][self.step, :, :] = traces['input']
            for name in self.layer_names:
                self.mem_state[name][self.step, :, :] = mems[name]
                self.spike_state[name][self.step, :, :] = spikes[name]
                self.postsyn_traces[name][self.step, :, :] = traces[name]

            self.mem_state['output'][self.step, :, :] = o_mem
            self.spike_state['output'][self.step, :, :] = o_spike
            self.postsyn_traces['output'][self.step, :, :] = o_traces

    def update_traces(self, trace, spikes):

        trace = trace*self.stdp_alpha*(1-spikes) + spikes # if spike, trace = 1, else trace*=alpha
        return trace

    def update_weights_stdp(self, pre_trace, post_trace):

        if self.w_idx == 1:
            w = self.f0_i
        else:
            w = self.h_layers[self.w_idx-1]

        dw = torch.zeros(len(post_trace.T), len(pre_trace.T), device =self.device)

        # print(dw.shape)

        for pre_idx, pre_tr in enumerate(pre_trace.T):
            dw[:, pre_idx] = torch.squeeze((post_trace.T > pre_tr) * pre_tr * post_trace.T)

        # print(w.weight.data.shape)

        scale = 0.01
        #new_w = w.weight.data + scale*dw
        
        new_w = w.weight.data + self.stdp_scale*dw

        modify_weights(w,new_w, mode='replace')

        #return None

    def forward(self, input):

        mems, spikes, traces, o_mem, o_spikes, o_traces = self.init_state()
        self.o_sumspike = output_mot = torch.zeros(
            self.batch_size, self.num_output, device=self.device)
        self.h_sum_spike = torch.tensor(0.0)  # for spike-regularization
        self.h_sum_spikes_per_layer = torch.zeros(self.num_layers)

        for step in range(self.win):

            self.w_idx = 0
            self.tau_idx = 0

            self.step = step

            traces['input'] = self.update_traces(traces['input'], input[:,step, :].reshape(self.batch_size, -1))
            prev_trace = traces['input']

            prev_spikes = self.f0_i(input[:, step, :].view(self.batch_size, -1))

            for i, layer in enumerate(self.layer_names):

                mems[layer], spikes[layer] = self.update_mem_fn(
                    prev_spikes.reshape(self.batch_size, -1), spikes[layer], mems[layer], self.thresh)
                
                traces[layer] = self.update_traces(traces[layer], spikes[layer])

                self.update_weights_stdp(prev_trace, traces[layer])

                prev_trace = traces[layer]
                prev_spikes = spikes[layer]

                self.h_sum_spike = self.h_sum_spike + spikes[layer].sum()

                # calculate avg spikes per layer
                self.h_sum_spikes_per_layer[i] = self.h_sum_spikes_per_layer[i] + spikes[layer].sum()

            o_mem, o_spikes = self.update_mem(
                prev_spikes.reshape(self.batch_size, -1), o_spikes, o_mem, self.output_thresh)
            
            o_traces = self.update_traces(o_traces, o_spikes)
            self.update_weights_stdp(prev_trace, o_traces)

            self.update_logger(input[:, step, :].view(self.batch_size, -1), mems, spikes, traces, o_mem, o_spikes, o_traces)

            self.o_sumspike = self.o_sumspike + o_spikes

            output_mot = output_mot + F.softmax(o_mem, dim=1)

        self.h_sum_spike = self.h_sum_spike / self.num_layers

        output_sum = self.o_sumspike / (self.win)

        return output_sum, output_mot