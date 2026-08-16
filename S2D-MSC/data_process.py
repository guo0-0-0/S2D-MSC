import numpy as np
import h5py
import pickle
import re
from einops import rearrange
from sklearn.model_selection import train_test_split
import scipy.io as sio

####################################### first step ##############################################
########################### multi patch 获取多个数据中snr>5的索引 #################################
def snr_thre_index(mdf_file, snr_thres=5):
	# load mdf data
	f = h5py.File(mdf_file, 'r')

	# get background mask
	isBG = f['/measurement/isBackgroundFrame'][:].view(bool)

	# remove background and get SM with shape (C, K, H * W * D), e.g., (3, 3294, 33*33*27)
	SM = f['/measurement/data'][:, :, :, :].squeeze()[:, :, isBG == False]
	print('SM shape', SM.shape)

	S_size = f['/calibration/size']
	print('S_size', S_size)

	SM_img = rearrange(SM, 'n c (d h w) -> n c d h w', d=S_size[2], h=S_size[0], w=S_size[1])
	print('SM_img shape', SM_img.shape)

	# get low SNR signals mask
	snr = f['calibration']['snr'][:, :, :].squeeze()
	print('snr shape', snr.shape)

	mask = snr > snr_thres
	# print('mask', mask)
	print('mask shape', mask.shape)

	return mask

def get_snr_index(root_path, snr_thres=5):
	mdf_files = [r'/media/OpenMPI_SM/OpenData_8.mdf',
				 r'/media/OpenMPI_SM/OpenData_9.mdf',
				 r'/media/OpenMPI_SM/OpenData_10.mdf']

	for mdf_file in mdf_files:
		print(mdf_file)
		thre_index = snr_thre_index(mdf_file, snr_thres)  # snr_thres  
		print('thre_index shape', thre_index.shape)
		experiment_idx = re.findall('\d+', mdf_file)[1]  # 读取文件地址中的数字
		thre_index_file = root_path +'data' + experiment_idx + '_snr' + str(snr_thres) + '_thre_index.pkl'
		pickle.dump(thre_index, open(thre_index_file, 'wb'))


#################################################
# Extract corresponding frequency row data based on the indices.
######################################################
def get_SM_img(mdf_file, all_data_snr_index):
	# load mdf data
	f = h5py.File(mdf_file, 'r')

	# get background mask
	isBG = f['/measurement/isBackgroundFrame'][:].view(bool)

	# remove background and get SM with shape (C, K, H * W * D), e.g., (3, 3294, 33*33*27)
	SM = f['/measurement/data'][:, :, :, :].squeeze()[:, :, isBG == False]
	print('SM shape', SM.shape)

	S_size = f['/calibration/size']
	print('S_size', S_size)
	
	SM_img = rearrange(SM, 'n c (d h w) -> n c d h w', d=S_size[2], h=S_size[0], w=S_size[1])
	print('SM_img shape', SM_img.shape)

	mask = all_data_snr_index

	print('mask shape', mask.shape)

	ture_indices = np.argwhere(mask)
	print('ture_indices', ture_indices[780])
	print('ture_indices.shape', ture_indices.shape)

	# shape(N, H*W*D)
	high_snr_SM = SM[mask]
	print('high_snr_SM shape', high_snr_SM.shape)

	# two channels respectively for Real value and Imag value
	Re_SM, Im_SM = high_snr_SM.real[:, np.newaxis, :], high_snr_SM.imag[:, np.newaxis, :]
	# shape(N, 2, H*W*D)
	SM_img = np.concatenate([Re_SM, Im_SM], 1)
	print('SM_img shape', SM_img.shape)

	# shape(N, 2, D, H, W )
	SM_img = rearrange(SM_img, 'n c (d h w) -> n c d h w', d=S_size[2], h=S_size[0], w=S_size[1])

	return SM_img


##################### second step ####################
# 1.Get indices with SNR > 5 from three datasets and calculate the intersection
# 2. Select corresponding frequency row data based on the indices.

################################################################################

def get_SM_data(root_path, snr):
	data8_mask_path = root_path + 'data8_snr' + str(snr) + '_thre_index.pkl'
	data8_mask = pickle.load(open(data8_mask_path, 'rb'))

	data9_mask_path = root_path + 'data9_snr' + str(snr) + '_thre_index.pkl'
	data9_mask = pickle.load(open(data9_mask_path, 'rb'))

	data10_mask_path = root_path + 'data10_snr' + str(snr) + '_thre_index.pkl'
	data10_mask = pickle.load(open(data10_mask_path, 'rb'))


	three_data_thre_index = np.logical_and.reduce([data8_mask, data9_mask, data10_mask])  
	print('three_data_mask shape', three_data_thre_index.shape)

	########## read origin data path  
	mdf_files = [r'/media/OpenMPI_SM/OpenData_8.mdf',
					r'/media/OpenMPI_SM/OpenData_9.mdf',
					r'/media/OpenMPI_SM/OpenData_10.mdf']

	for mdf_file in mdf_files:
		print(mdf_file)
		SM_data = get_SM_img(mdf_file, three_data_thre_index)
		SM_data = SM_data[:, :, :, 1:, 1:]  #  shape: f, c, d, 32, 32
		print('SM_img shape', SM_data.shape)

		experiment_idx = re.findall('\d+', mdf_file)[1]  # 读取文件地址中的数字
		experiment_SM_file = root_path + '/real_data_40/data' + experiment_idx + '_snr' + str(snr) + '.pkl'
		pickle.dump(SM_data, open(experiment_SM_file, 'wb'))


if __name__ == '__main__':
    root_path = r'/media/S2D-MSC/processed_data/'
    get_snr_index(root_path, snr_thres=5) #
    # get_SM_data(root_path, snr=5)
   