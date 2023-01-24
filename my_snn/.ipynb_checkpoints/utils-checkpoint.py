import torch
from torchsummary import summary
from my_snn.abstract_rsnn import CHECKPOINT_PATH
import time 
import matplotlib.pyplot as plt 
import os
import numpy as np

class ModelLoader:

    def __new__(cls, *args, **kwargs):

        modelname, location, batch_size, device = args

        params = torch.load(os.path.join(CHECKPOINT_PATH,location,modelname), map_location=torch.device('cpu'))

        params['kwargs']['batch_size'] = batch_size
        params['kwargs']['device'] = device
        kwargs = params['kwargs']
        snn = params['type']
        snn = snn(**kwargs)
        snn.to(device)
        snn.load_state_dict(params['net'])
        snn.acc = params['acc_record']
        snn.train_loss = params['train_loss']
        snn.test_loss = params['test_loss']
        snn.test_spk_count = params['test_spk']
        print('instance of {} loaded sucessfully'.format(params['type']))

        return snn

def train(snn, data, learning_rate, num_epochs, spkreg = 0.0, l1_reg=0.0, dropout = 0.0, lr_scale = (2.0, 5.0), ckpt_dir = 'checkpoint', test_fn=None, scheduler=(1, 0.98)):

    '''
    lr scale: originally I worked with same (1.0, 1.0 )lr for base (weights) tau_m, tau_adp 
    then found tha for some nets its better to use different lr
    '''

    test_loader, train_loader = data

    sample_input = torch.zeros(snn.batch_size, snn.win, snn.num_input, device=snn.device)
    s = summary(snn, sample_input, verbose=0)
    print(snn)
    print(f'Total params: {s.total_params}')
    print(f'Total mult-adds (M): {s.total_mult_adds / 1e6}')
    snn.info['total_params'] = s.total_params
    snn.info['trainable_params'] = s.trainable_params
    snn.info['mult_adds'] = s.total_mult_adds
    
    tau_m_params = [getattr(snn, name.split('.')[0]) for name, _ in snn.state_dict().items() if 'tau_m' in name]
    tau_adp_params = [getattr(snn, name.split('.')[0]) for name, _ in snn.state_dict().items() if 'tau_adp' in name]
    
    tau_m_lr_scale = lr_scale[0]
    tau_adp_lr_scale = lr_scale[1]

    optimizer = torch.optim.Adam([
        {'params': snn.base_params},
        {'params': tau_m_params, 'lr': learning_rate * tau_m_lr_scale},
        {'params': tau_adp_params, 'lr': learning_rate * tau_adp_lr_scale}],
        lr=learning_rate, eps=1e-5)

    #act_fun = ActFun.apply
    print(f'training {snn.modelname} for {num_epochs} epochs...')

    # training loop
    max_acc = 0
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        print('Epoch [%d/%d]'  % (epoch + 1, num_epochs))
        start_time = time.time()
        snn.train_step(train_loader, optimizer=optimizer, spkreg=spkreg, l1_reg=l1_reg, dropout=dropout)
        t =  time.time() - start_time
        print('Time elasped:', t)

        # adjust learning rate
        #optimizer = snn.lr_scheduler(optimizer, lr_decay_epoch=1)
        
        if scheduler:
            # optimizer = snn.lr_scheduler(optimizer, lr_decay_epoch=10) bojian
            optimizer = snn.lr_scheduler(optimizer, lr_decay_epoch=scheduler[0], lr_decay= scheduler[1])

        if test_fn is not None:
            test_fn(snn, ckpt_dir, test_loader, max_acc, epoch)
        else:
            if (epoch + 1) % 5 == 0:
                snn.test(test_loader)
                snn.save_model(snn.modelname, ckpt_dir)

   
def training_plots(snn, figsize=None):
    _ , (ax1, ax2, ax3) = plt.subplots(1,3, figsize=figsize)
    snn.plot_per_epoch(snn.train_loss, ax1, 'train_loss')
    snn.plot_per_epoch(snn.test_loss, ax1, 'test_loss')
    snn.plot_per_epoch(snn.acc, ax2, 'test acc')
    snn.plot_per_epoch(snn.test_spk_count, ax3, 'test_spk_count')

    max_acc = np.max(np.array(snn.acc)[:, 1])

    print(f'{snn.modelname} max acc: {max_acc}')