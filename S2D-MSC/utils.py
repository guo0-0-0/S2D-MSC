import random
import torch
import numpy as np
from datetime import datetime
import json
from torch.nn.parallel import DistributedDataParallel
import os
import torch.nn as nn
import pickle

def setup_seed(seed=2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def concat_train_test_data(root_path, model_type, idx1, idx2, down, variant):

    if down == 4: 

        train_data1_path = root_path + 'processed_data/' + model_type + '_pseudo_data_32/' + model_type + '_pseudo' + idx1 + '_snr5_down2.pkl'
        train_data2_path = root_path + 'processed_data/' + model_type + '_pseudo_data_32/' + model_type + '_pseudo' + idx2 + '_snr5_down' + str(down) + '.pkl'

        test_data1_path = train_data1_path
        test_data2_path = root_path + 'processed_data/real_data_32/data' + idx2 + '_snr5.pkl'

    if down == 8:  

        train_data1_path = root_path + 'processed_data/' + model_type + '_pseudo_data_32/' + model_type + '_pseudo' + idx1 + '_snr5_down4.pkl'
        train_data2_path = root_path + 'processed_data/' + model_type + '_pseudo_data_32/' + model_type + '_pseudo' + idx2 + '_snr5_down' + str(down) + '.pkl'  

        test_data1_path = train_data1_path
        test_data2_path = root_path + 'processed_data/real_data_32/data' + idx2 + '_snr5.pkl'


    train_data1 = pickle.load(open(train_data1_path, 'rb'))
    train_data2 = pickle.load(open(train_data2_path, 'rb'))

    train_data1 = train_data1[np.newaxis, :, :, :, :, :]
    train_data2 = train_data2[np.newaxis, :, :, :, :, :]

    test_data1 = pickle.load(open(test_data1_path, 'rb'))
    test_data2 = pickle.load(open(test_data2_path, 'rb'))

    test_data1 = test_data1[np.newaxis, :, :, :, :, :]
    test_data2 = test_data2[np.newaxis, :, :, :, :, :]


    train_two_patch_data = np.concatenate((train_data1, train_data2), axis=0)
    test_two_patch_data = np.concatenate((test_data1, test_data2), axis=0)

    return train_two_patch_data, test_two_patch_data, train_data1_path, train_data2_path, test_data1_path, test_data2_path
    
  

def down_sampling_cross(HR, scile0, scile1, down=4):
    sampling_indices = []
    HR_size = (HR.shape[2], HR.shape[3])
    for axis in range(2):
        if axis == 0:
            indices = np.arange(HR_size[axis])
            down_sampling_indices = indices[scile0::down]
        else:
            indices = np.arange(HR_size[axis])
            down_sampling_indices = indices[scile1::down]

        sampling_indices.append(down_sampling_indices)

    LR_SM = HR[:, :, sampling_indices[0]] \
        [:, :, :, sampling_indices[1]]
    return LR_SM


class ConfigLogger:
    def __init__(self, base_dir='experiments005', args=None, log_name='logname'):
        time_now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        folder_name = f'{log_name}_{time_now}'
        self.exp_dir = os.path.join(base_dir, folder_name)
        os.makedirs( self.exp_dir, exist_ok=True)

        self.log_path = os.path.join(self.exp_dir, log_name + '.txt')

        if args is not None:
            self.save_args(args)

    def save_args(self, args):
        args_dict = vars(args) if not isinstance(args, dict) else args
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write('======== Args Configuration =========')
            f.write(json.dumps(args_dict, indent=4, ensure_ascii=False))
            f.write('\n\n')
        print(f'[ConfigLogger] Saved args to {self.log_path}')

    def add_info(self, key, value):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f'{key}: {value}\n')
        print(f'[ConfigLogger] Added extra info -> {key}: {value}')

    def log(self,message):
        time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f'[{time_now}] {message}\n')
        print(f'[ConfigLogger] {message}')

    def save_mode(self, net, net_label):
        save_file_name = f'{net_label}.pth'
        save_path = os.path.join(self.exp_dir, save_file_name)
        if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
            net = net.module
        state_dict = net.state_dict()
        for key, param in state_dict.items():
            if key.startswith('module.'):
                key = key[7:]  # remove unnecessary 'module.'
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def get_exp_dir(self):
        return self.exp_dir