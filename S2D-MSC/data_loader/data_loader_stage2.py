import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from utils import down_sampling_cross


class SM_Dataset(Dataset):

    def __init__(self, SM_path, sampling_position_path,
                     mode='train', down=4, pos=0):

        self.mode = mode
        self.down = down
        assert self.mode in ['train', 'test']

        if isinstance(SM_path, str):
            SM_data = pickle.load(open(SM_path, 'rb'))  # shape (p, f, c, d, h, w)
        else:
            SM_data = SM_path


        self.two_patch_SM = rearrange(SM_data, 'p f c d h w -> p f d c h w')
        p, f, d, c, h, w = self.two_patch_SM.shape
        print('self.two_patch_SM shape', self.two_patch_SM.shape)

        if mode == 'train':     # random sampling positions for train
            rows = f * d
            cols = int((h / self.down) * (w / self.down))
            random_sampling_index = np.random.randint(0, 1024, size=(rows, cols))
            self.sampling_position_index_expanded = np.sort(random_sampling_index, axis=1)
            print('train data self.sampling_position_index_expanded shape: ', self.sampling_position_index_expanded.shape)

        else:      # real spare sampling positions for test

            fix_sampling_index = pickle.load(
                open(sampling_position_path, 'rb'))  # poisson sampling position  # shape (27, 16)
            sampling_position_index_expanded = np.tile(fix_sampling_index, (f, 1, 1))  # shape (f, 27, 16)
            self.sampling_position_index_expanded = rearrange(sampling_position_index_expanded, 'f d n -> (f d) n')
            print('self.sampling_position_index_expanded shape', self.sampling_position_index_expanded.shape)

        self.patch1_HR_SM = self.two_patch_SM[0]  # shape (f, d, c, h, w)
        self.patch2_HR_SM = self.two_patch_SM[1]  # shape (f, d, c, h, w)

        if mode == 'train':
            self.patch1_HR_SM = rearrange(self.patch1_HR_SM, 'f d c h w -> (f d) c h w')
            self.patch2_HR_SM = rearrange(self.patch2_HR_SM, 'f d c h w -> (f d) c h w')
            print('self.patch2_HR_SM shape', self.patch2_HR_SM.shape)

            ########### data norm: patch size = 1 ############
            self.a = rearrange(self.patch2_HR_SM, 'n c h w -> n c (h w)')
            batch_idx = np.arange(self.a.shape[0])[:, None, None]  # (100,1,1)
            channel_idx = np.arange(self.a.shape[1])[None, :, None]  # (1,2,1)

            b_expanded = self.sampling_position_index_expanded[:, None, :]  # (100,1,12)

            self.selected = self.a[batch_idx, channel_idx, b_expanded]  # (100, 2, 12)

            self.patch1_mean = np.mean(self.patch1_HR_SM, (1, 2, 3)).reshape(self.patch1_HR_SM.shape[0], 1, 1, 1)
            self.patch1_std = np.std(self.patch1_HR_SM, (1, 2, 3)).reshape(self.patch1_HR_SM.shape[0], 1, 1, 1)

            self.patch2_mean = np.mean(self.selected, (1, 2)).reshape(self.patch2_HR_SM.shape[0], 1, 1, 1)
            self.patch2_std = np.std(self.selected, (1, 2)).reshape(self.patch2_HR_SM.shape[0], 1, 1, 1)

        else:

            self.patch2_LR_SM = np.zeros((f, d, c, int(h / self.down), int(w / self.down)))  # shape (f*d, c, h/2, w/2)

            for i in range(self.patch2_HR_SM.shape[1]):
                if i % 3 == 0:
                    self.patch2_LR_SM[:, i, :, :, :] = down_sampling_cross(self.patch2_HR_SM[:, i, :, :, :], pos[0], pos[1], down=self.down)
                elif i % 3 == 1:
                    self.patch2_LR_SM[:, i, :, :, :] = down_sampling_cross(self.patch2_HR_SM[:, i, :, :, :], pos[2], pos[3], down=self.down)
                else:
                    self.patch2_LR_SM[:, i, :, :, :] = down_sampling_cross(self.patch2_HR_SM[:, i, :, :, :], pos[4], pos[5], down=self.down)

            self.patch2_LR_SM = rearrange(self.patch2_LR_SM, 'f d c h w -> (f d) c h w')  # shape (f d) c h w
            print('self.patch2_LR_SM shape', self.patch2_LR_SM.shape)

            self.patch1_HR_SM = rearrange(self.patch1_HR_SM, 'f d c h w -> (f d) c h w')
            self.patch2_HR_SM = rearrange(self.patch2_HR_SM, 'f d c h w -> (f d) c h w')

            self.patch1_mean = np.mean(self.patch1_HR_SM, (1, 2, 3)).reshape(self.patch1_HR_SM.shape[0], 1, 1, 1)
            self.patch1_std = np.std(self.patch1_HR_SM, (1, 2, 3)).reshape(self.patch1_HR_SM.shape[0], 1, 1, 1)

            self.patch2_mean = np.mean(self.patch2_LR_SM, (1, 2, 3)).reshape(self.patch2_HR_SM.shape[0], 1, 1, 1)
            self.patch2_std = np.std(self.patch2_LR_SM, (1, 2, 3)).reshape(self.patch2_HR_SM.shape[0], 1, 1, 1)
            print('patch2_mean shape', self.patch2_mean.shape)


        self.norm_patch1_HR_SM = (self.patch1_HR_SM - self.patch1_mean) / self.patch1_std
        self.norm_patch2_HR_SM = (self.patch2_HR_SM - self.patch2_mean) / self.patch2_std
        print('self.norm_patch1_HR_SM shape', self.norm_patch1_HR_SM.shape)

        self.norm_patch1_HR_SM = self.norm_patch1_HR_SM[:, np.newaxis, :, :, :]
        self.norm_patch2_HR_SM = self.norm_patch2_HR_SM[:, np.newaxis, :, :, :]
        print('self.norm_patch2_HR_SM shape', self.norm_patch2_HR_SM.shape)

        self.all_HR_SM = np.concatenate((self.norm_patch1_HR_SM, self.norm_patch2_HR_SM), axis=1)  # shape (f d) p c h w
        print('self.all_HR_SM shape', self.all_HR_SM.shape)

    def __len__(self):
        self.length = self.all_HR_SM.shape[0]
        return self.length

    def __getitem__(self, idx):
        all_HR_SM = self.all_HR_SM[idx]
        all_HR_SM = torch.from_numpy(all_HR_SM).float()

        poisson_index = self.sampling_position_index_expanded[idx]
        return all_HR_SM, poisson_index

    def get_LR_mean_and_std(self):
        return self.patch1_mean, self.patch1_std, self.patch2_mean, self.patch2_std, self.all_HR_SM


def load_dataloader(root_path, sampling_path, batch_size=16, mode='train', down=4, pos=0):
    data_dataset = SM_Dataset(root_path, sampling_path, mode=mode, down=down, pos=pos)

    patch1_mean, patch1_std, patch2_mean, patch2_std, all_SM = data_dataset.get_LR_mean_and_std()

    # loaders
    if mode in ['train']:
        data_loader = DataLoader(dataset=data_dataset, batch_size=batch_size, shuffle=True)
    else:
        data_loader = DataLoader(dataset=data_dataset, batch_size=batch_size, shuffle=False)

    return data_loader, (patch1_mean, patch1_std, patch2_mean, patch2_std, all_SM), data_dataset






