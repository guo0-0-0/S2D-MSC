import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import setup_seed
from data_loader.data_loader_stage1 import load_dataloader
from model.stage1_model import SiameseAutoencoderViT
import os
import numpy as np
from functools import partial
from utils import ConfigLogger
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--total_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--feat_embed', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--decoder_embed_dim', type=int, default=384)
    parser.add_argument('--decoder_depth', type=int, default=6)
    parser.add_argument('--decoder_num_heads', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=int, default=4)

    args = parser.parse_args()
    setup_seed(args.seed)

    root_path = r'/media/S2D-MSC/'

    ref_data = 'data10'
    down = 4
    
    ref_data_id = re.findall('\d+', ref_data)[0]

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if down == 4:
        mask_ratio = 0.937
    else:
        mask_ratio = 0.984

    log_base_path = os.path.join(root_path, 'stage1_experiments', 'experiments_stage1_' + ref_data)
    log_name = ('stage1_train_' + ref_data + '_snr5' +
                '_batchsize' + str(args.batch_size) + '_epoch' + str(args.total_epochs) +
                '_192_4464_' + str(down) + 'x')

    logger = ConfigLogger(base_dir=log_base_path, args=args, log_name=log_name)

    train_data_path = root_path + 'processed_data/pseudo_data_32/pseudo10_snr5_down4.pkl'
    test_data_path = root_path + 'processed_data/pseudo_data_32/pseudo10_snr5_down4.pkl'

    if down == 4:
        sampling_path = root_path + 'processed_data/' + ref_data_id + '_all_uniform_sampling_4x.pkl'
        p = [2, 2]
    else:
        sampling_path = root_path + 'processed_data/' + ref_data_id + '_all_uniform_sampling_8x.pkl'
        p = [4, 4]

    logger.add_info('data1', ref_data)
    logger.add_info('train_data_path', train_data_path)
    logger.add_info('test_data_path', test_data_path)
    logger.add_info('sampling_path', sampling_path)
    
    train_loader = (load_dataloader(train_data_path, sampling_path,
                        batch_size=args.batch_size, mode='train',down=down, pos=p))

    test_loader = (load_dataloader(test_data_path, sampling_path,
                        batch_size=args.batch_size, mode='test', down=down, pos=p))

    model = SiameseAutoencoderViT(
        img_size=args.img_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads, mlp_ratio=args.mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    model = model.to(device)
    model = torch.nn.DataParallel(model)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    loss_fn = torch.nn.L1Loss(reduction='sum').to(device)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim,
            milestones=[35],
            gamma=0.5
        )

    optim.zero_grad()
    train_loss_list = []
    test_loss_list = []

    for epoch in range(args.total_epochs):
        model.train()
        losses = []
        train_step = len(train_loader)
        with tqdm(total=train_step, desc=f'Train Epoch {epoch + 1}/{args.total_epochs}', postfix=dict,
                  mininterval=0.3) as pbar1:
            for all_patch_SM, sampling_index in iter(train_loader):
                all_patch_SM = all_patch_SM.to(device)
                sampling_index = sampling_index.to(device).long()

                pre_imag = model(all_patch_SM, mask_ratio, sampling_index)

                spa_loss = loss_fn(pre_imag, all_patch_SM[:, 1, :, :, :])
                spa_loss = spa_loss / np.prod(pre_imag.shape[1:])
                loss = spa_loss / pre_imag.shape[0]

                loss.backward()
                optim.step()
                optim.zero_grad()

                losses.append(loss.item())
                pbar1.set_postfix(**{'Train Loss': np.mean(losses)})
                pbar1.update(1)

        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        train_loss_list.append(avg_loss)

        log_message = f'Train epoch [{epoch + 1}/{args.total_epochs}], lr: {optim.param_groups[0]["lr"]:.6f}, loss: {train_loss_list[-1]}'
        logger.log(log_message)
        print('epoch: ', epoch + 1, 'mean loss: ', train_loss_list[-1])
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_losses = []
                test_step = len(test_loader)
                with tqdm(total=test_step, desc=f'Val Epoch {epoch + 1}/{args.total_epochs}', postfix=dict,
                          mininterval=0.3) as pbar2:
                    for test_all_patch_SM, test_sampling_index in iter(test_loader):
                        test_all_patch_SM = test_all_patch_SM.to(device)
                        test_sampling_index = test_sampling_index.to(device).long()
                        test_pre_imag = model(test_all_patch_SM, mask_ratio, test_sampling_index)

                        test_spa_loss = loss_fn(test_pre_imag, test_all_patch_SM[:, 1, :, :, :])
                        test_spa_loss = test_spa_loss / np.prod(test_pre_imag.shape[1:])
                        test_loss = test_spa_loss / test_pre_imag.shape[0]

                        test_losses.append(test_loss.item())
                        pbar2.set_postfix(**{'Val Loss': np.mean(test_losses)})
                        pbar2.update(1)
                avg_test_loss = sum(test_losses) / len(test_losses)
                test_loss_list.append(avg_test_loss)

            log_message = f'Test epoch [{epoch + 1}/{args.total_epochs}], loss: {test_loss_list[-1]}'
            logger.log(log_message)
            print('epoch: ', epoch + 1, 'mean test loss: ', test_loss_list[-1])

    logger.save_mode(model, log_name)


