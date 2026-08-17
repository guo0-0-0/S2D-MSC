import torch
from utils import *
import os
from einops import rearrange
import pickle
from data_loader.data_loader_stage2 import load_dataloader
from model.stage2_model import Stage2_ViT
from utils import *
import argparse
import re

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--in_chans', type=int, default=2)
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

    ref_data = 'pseudo10'
    tgt_data = 'pseudo8'

    down = 8

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if down == 4:
        mask_ratio = 0.937
    else:
        mask_ratio = 0.984

    idx1 = re.findall('\d+', ref_data)[0]
    idx2 = re.findall('\d+', tgt_data)[0]

    # use MKD-SM network to obtained pseudo data
    if down == 4:
        sampling_path = root_path + 'processed_data/' + idx1 + '_all_slice_cross_misaligned_4x.pkl'
        p = [0, 0, 2, 2, 0, 3]
    else:
        sampling_path = root_path + 'processed_data/' + idx1 + '_all_slice_cross_misaligned_8x.pkl'
        p = [0, 0, 2, 5, 5, 3]

    model = Stage2_ViT(
        pretrained_path=None, img_size=args.img_size, patch_size=args.patch_size,
        in_chans=args.in_chans, embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio)
 
    ###########  load train data，test data ############
        
    _, test_data, _, _, _, _ = concat_train_test_data(root_path, idx1, idx2, down)
        
    test_loader, (test_mean_patch1, test_std_patch1, test_mean_patch2, test_std_patch2,
                  test_origin_SM), test_dataset = (
        load_dataloader(test_data, sampling_path, batch_size=args.batch_size, mode='test', down=down, pos=p))

    model_path = root_path + 'stage2_experiments_data' + idx1 + '/stage2_' + str(down) + 'x/' + \
                            'stage2_train_p10_p8_test_p10_r8_snr5_1e5_batchsize64_epoch10_emb192_4464_8x/' + \
                            'stage2_train_p10_p8_test_p10_r8_snr5_1e5_batchsize64_epoch10_emb192_4464_8x.pth'

    print('load resolution model success from: ', model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    pred_HR_SM, test_nrmses = [], []

    with (((torch.no_grad()))):
        model.eval()
        for step, (test_all_patch_SM, test_sampling_index) in enumerate(test_loader):
            test_all_patch_SM = test_all_patch_SM.to(device)
            test_sampling_index = test_sampling_index.to(device).long()

            test_pre_imag = model(test_all_patch_SM, mask_ratio, test_sampling_index)
            pred_HR_SM.append(test_pre_imag.cpu().numpy())

        pred_HR_SM = np.concatenate(pred_HR_SM, 0)
        pred_HR_SM = pred_HR_SM * test_std_patch2 + test_mean_patch2
        print('pred_HR_SM shape', pred_HR_SM.shape)

        origin_HR_SM_patch1 = test_origin_SM[:, 0, :, :].squeeze()
        origin_HR_SM_patch1 = origin_HR_SM_patch1 * test_std_patch1 + test_mean_patch1

        origin_HR_SM_patch2 = test_origin_SM[:, 1, :, :].squeeze()
        origin_HR_SM_patch2 = origin_HR_SM_patch2 * test_std_patch2 + test_mean_patch2
        print('origin_HR_SM_patch2 shape', origin_HR_SM_patch2.shape)

        pre_HR_SM = rearrange(pred_HR_SM, '(f z) c h w -> f c z h w', z=27)
        print('pre_HR_SM shape', pre_HR_SM.shape)

        # pre_HR_SM_path = root_path + 'SM_recovery_result/' + \
        #                   'MSC_' + idx1 + '_' + idx2 + '_down' + str(down) + '.pkl'
        # pickle.dump(pre_HR_SM, open(pre_HR_SM_path, 'wb'))

        origin_HR_SM_patch1 = rearrange(origin_HR_SM_patch1, '(f z) c h w -> f c z h w', z=27)
        origin_HR_SM_patch2 = rearrange(origin_HR_SM_patch2, '(f z) c h w -> f c z h w', z=27)
        print('origin_HR_SM_patch2 shape', origin_HR_SM_patch2.shape)

        comp_reco_HR_SM = pre_HR_SM[:, 0, :, :, :] + 1j * pre_HR_SM[:, 1, :, :, :]
        comp_origin_HR_SM_patch1 = origin_HR_SM_patch1[:, 0, :, :, :] + 1j * origin_HR_SM_patch1[:, 1, :, :, :]
        comp_origin_HR_SM_patch2 = origin_HR_SM_patch2[:, 0, :, :, :] + 1j * origin_HR_SM_patch2[:, 1, :, :, :]
        print('comp_origin_HR_SM_patch2 shape', comp_origin_HR_SM_patch2.shape)

        ############################ caclutated nRMSE ##############################
        vec_reco_HR_SM = comp_reco_HR_SM.reshape(comp_reco_HR_SM.shape[0], 1, -1)
        vec_origin_HR_SM = comp_origin_HR_SM_patch2.reshape(comp_origin_HR_SM_patch2.shape[0], 1, -1)
        print('vec_reco_HR_SM shape', vec_reco_HR_SM.shape)

        N = vec_reco_HR_SM.shape[-1]
        rmse = np.linalg.norm(vec_reco_HR_SM - vec_origin_HR_SM, 'fro', (1, 2)) / np.sqrt(N)
        val_nrmse = rmse / np.max(np.abs(vec_origin_HR_SM), axis=(1, 2))
        test_nrmses.append(val_nrmse.mean())
        print('test_nrmses: ', test_nrmses)




