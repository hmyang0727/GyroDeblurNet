import sys
sys.path.append('../')

from PIL import Image
import numpy as np
import os
import rawpy
import argparse
from tqdm import tqdm
import torch
from model import GyroDeblurNet
import warnings
import cv2
from glob import glob
warnings.filterwarnings("ignore")


SAMPLING_RATE = 0.005  # Sampling interval of gyro sensor (unit: second)
    
    
center_vectors = []
    
for i in range(4000):
    row = []
    for j in range(3040):
        row.append([j, i, 1])
    center_vectors.append(row)
center_vectors = torch.tensor(center_vectors, dtype=torch.float32)[0::8, 0::8, :].cuda()  # 1000 x 760


def compute_rotation_matrix(ang_vel_x: float, ang_vel_y: float, ang_vel_z: float):
    '''
    Compute rotation matrix using given angular velocities.
    ang_vel_x: Angular velocity of x-axis.
    ang_vel_y: Angular velocity of y-axis.
    ang_vel_z: Angular velocity of z-axis.
    '''
    
    R_x = torch.tensor([
        [                      1,                      0,                      0],
        [                      0,  torch.cos(-ang_vel_x), -torch.sin(-ang_vel_x)],
        [                      0,  torch.sin(-ang_vel_x),  torch.cos(-ang_vel_x)]
    ], dtype=torch.float32, device='cuda')
    R_y = torch.tensor([
        [  torch.cos(-ang_vel_y),                      0,  torch.sin(-ang_vel_y)],
        [                      0,                      1,                      0],
        [ -torch.sin(-ang_vel_y),                      0,  torch.cos(-ang_vel_y)]
    ], dtype=torch.float32, device='cuda')
    R_z = torch.tensor([
        [   torch.cos(ang_vel_z),   torch.sin(ang_vel_z),                      0],
        [  -torch.sin(ang_vel_z),   torch.cos(ang_vel_z),                      0],
        [                      0,                      0,                      1]
    ], dtype=torch.float32, device='cuda')
    
    R = R_x @ R_y @ R_z
    
    return R


def compute_rotation_matrix_vectorized_version(ang_vel_x, ang_vel_y, ang_vel_z):
    '''
    Compute rotation matrix using given angular velocities.
    ang_vel_x: Angular velocity of x-axis. (8, ) shaped torch.Tensor.
    ang_vel_y: Angular velocity of y-axis. (8, ) shaped torch.Tensor.
    ang_vel_z: Angular velocity of z-axis. (8, ) shaped torch.Tensor.
    '''
    
    sin_x = torch.cat([torch.sin(ang_vel_x)[0:4], torch.sin(ang_vel_x)[5:9]], dim=0)
    cos_x = torch.cat([torch.cos(ang_vel_x)[0:4], torch.cos(ang_vel_x)[5:9]], dim=0)
    sin_y = torch.cat([torch.sin(ang_vel_y)[0:4], torch.sin(ang_vel_y)[5:9]], dim=0)
    cos_y = torch.cat([torch.cos(ang_vel_y)[0:4], torch.cos(ang_vel_y)[5:9]], dim=0)
    sin_z = torch.cat([torch.sin(ang_vel_z)[0:4], torch.sin(ang_vel_z)[5:9]], dim=0)
    cos_z = torch.cat([torch.cos(ang_vel_z)[0:4], torch.cos(ang_vel_z)[5:9]], dim=0)
    
    R_x = torch.zeros(8, 3, 3).cuda()
    R_y = torch.zeros(8, 3, 3).cuda()
    R_z = torch.zeros(8, 3, 3).cuda()
    
    R_x[:, 0, 0] = 1
    R_x[:, 1, 1] = cos_x
    R_x[:, 1, 2] = -sin_x
    R_x[:, 2, 1] = sin_x
    R_x[:, 2, 2] = cos_x
    
    R_y[:, 0, 0] = cos_y
    R_y[:, 0, 2] = sin_y
    R_y[:, 1, 1] = 1
    R_y[:, 2, 0] = -sin_y
    R_y[:, 2, 2] = cos_y
    
    R_z[:, 0, 0] = cos_z
    R_z[:, 0, 1] = sin_z
    R_z[:, 1, 0] = -sin_z
    R_z[:, 1, 1] = cos_z
    R_z[:, 2, 2] = 1
    
    R = torch.einsum('ijk, ikl, ilm -> ijm', R_x, R_y, R_z)
    
    return R


def compute_homography(R: torch.Tensor):
    '''
    Compute homography matrix using the given rotation matrix R.
    R: Rotation matrix.
    '''
    
    # K: Camera intrinsic matrix of Samsung Galaxy S22 wide camera. (Obtained using checkerboard calibration)
    K = torch.tensor([
        [3.18259180e+03, 0.00000000e+00, 1.45868721e+03],
        [0.00000000e+00, 3.17929832e+03, 2.11425739e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=torch.float32, device='cuda')
    
    return K @ R @ torch.linalg.inv(K)


def process_raw_blurry(folder):
    '''
    Demosaic RAW image and save it as PNG image.
    Blurry image is downsampled before being saved.
    '''
    raw = rawpy.imread(os.path.join(folder, 'blurry.dng'))
    jpeg = raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.raw, no_auto_scale=False, no_auto_bright=False, gamma=(1,1))
    blurry = (((np.rot90(jpeg, k=3) / 255)) * 255).astype(np.uint8)[0:4000, 0:3040, :]
    blurry = cv2.resize(src=blurry, dsize=(760, 1000), interpolation=cv2.INTER_LANCZOS4)


def find_closest_timestamp(gyro_file: str, timestamp: int):
    # Open gyro data file
    with open(gyro_file, 'r') as f:
        gyro_ts = f.readlines()
    
    # Get timestamps only
    for i in range(len(gyro_ts)):
        gyro_ts[i] = int(gyro_ts[i].split()[0])
    
    # Find closest index
    for i in range(len(gyro_ts)):
        if timestamp < gyro_ts[i]:
            best_idx = i
            break
    
    if abs(timestamp - gyro_ts[i]) > abs(timestamp - gyro_ts[i-1]):
        best_idx -=1 
    
    return best_idx


def optimize_gyro_data(args):
    folders = sorted(glob(os.path.join(args.dir, '*')))

    img_channel = 3
    width = 32
    enc_blks = [2, 2, 2]
    middle_blk_num = 16
    dec_blks = [1, 1, 1]

    model = GyroDeblurNet(img_channel=img_channel, width=width, enc_blk_nums=enc_blks, middle_blk_num=middle_blk_num, dec_blk_nums=dec_blks)

    checkpoint = torch.load(args.ckpt, map_location='cuda')
    model.load_state_dict(checkpoint)
    model.eval()
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()

    for idx_, folder in tqdm(enumerate(folders), total=len(folders)):
        process_raw_blurry(folder)
        
        gyro_file = os.path.join(folder, 'gyro.txt')
        timestamp_file = os.path.join(folder, 'image_info.txt')
        
        with open(timestamp_file, 'r') as f:
            pic_ts_ = f.readlines()
            pic_ts = int(pic_ts_[0].split()[0])
            EXPOSURE_TIME = float(pic_ts_[0].split()[2]) / 1e9
            NUM_GYRO_SAMPLE = int(EXPOSURE_TIME // SAMPLING_RATE)
            
        with open(gyro_file, 'r') as f:
            gyro_data_ = f.readlines()
        
        gyro_ts_idx_ = find_closest_timestamp(gyro_file, pic_ts)
        
        gyro_ts_idx = gyro_ts_idx_
        gyro_data = gyro_data_[gyro_ts_idx:gyro_ts_idx+NUM_GYRO_SAMPLE+1]
        
        timestamp = []
        ang_vel_x_ = []
        ang_vel_y_ = []
        ang_vel_z_ = []
        
        for data in gyro_data:
            ts, x, y, z = data.split()
            timestamp.append(int(ts))
            # Consider gyro axis
            ang_vel_x_.append(-float(x))
            ang_vel_y_.append(float(y))
            ang_vel_z_.append(-float(z))
        
        slice_const = int(EXPOSURE_TIME // 0.05)
        timestamp = torch.tensor(timestamp, device='cuda', dtype=float)[::slice_const]
        ang_vel_x = torch.tensor(ang_vel_x_, device='cuda', dtype=torch.float32)[::slice_const]
        ang_vel_y = torch.tensor(ang_vel_y_, device='cuda', dtype=torch.float32)[::slice_const]
        ang_vel_z = torch.tensor(ang_vel_z_, device='cuda', dtype=torch.float32)[::slice_const]
        
        blur = torch.unsqueeze(torch.as_tensor(np.asarray(Image.open(os.path.join('.', 'blurry.png'))).transpose(2, 0, 1) / 255), dim=0).float().cuda()
            
        interp_factor = 8
        indices = torch.tensor([0, 2, 4, 6, 8, 0, 2, 4, 6, 8]).cuda()
            
        gap_timestamp = timestamp[1:] - timestamp[:-1]
        time_coeff = gap_timestamp * (10/8) * 1e-9
            
        gap_x_list = ((ang_vel_x[1:] - ang_vel_x[:-1]) / interp_factor) * indices
        gap_y_list = ((ang_vel_y[1:] - ang_vel_y[:-1]) / interp_factor) * indices
        gap_z_list = ((ang_vel_z[1:] - ang_vel_z[:-1]) / interp_factor) * indices
            
        input_x = ((ang_vel_x[:-1] + gap_x_list) * time_coeff)  # (10, ) shaped torch.Tensor
        input_y = ((ang_vel_y[:-1] + gap_y_list) * time_coeff)  # (10, ) shaped torch.Tensor
        input_z = ((ang_vel_z[:-1] + gap_z_list) * time_coeff)  # (10, ) shaped torch.Tensor
            
        R_list = compute_rotation_matrix_vectorized_version(input_x, input_y, input_z)
            
        # Compute H_pro
        R = torch.eye(3).cuda()
        for i in range(len(R_list)//2, len(R_list)):
            R = R_list[i] @ R
            if i == len(R_list) // 2:
                H_pro_list = torch.unsqueeze(compute_homography(R), dim=0)
            else:
                H_pro_list = torch.cat((H_pro_list, torch.unsqueeze(compute_homography(R), dim=0)), dim=0)
            
        # Compute H_pre
        R = torch.eye(3).cuda()
        for i in range((len(R_list)//2)-1, -1, -1):
            R = R @ R_list[i]
            if i == (len(R_list) // 2) - 1:
                H_pre_list = torch.unsqueeze(torch.linalg.inv(compute_homography(R)), dim=0)
            else:
                H_pre_list = torch.cat((H_pre_list, torch.unsqueeze(torch.linalg.inv(compute_homography(R)), dim=0)), dim=0)

        # Compute camera motion field
        for idx, H_pro in enumerate(H_pro_list):
            end_vectors_copied = torch.einsum('ij, klj -> kli', H_pro, center_vectors)
            end_vectors_copied = end_vectors_copied / torch.unsqueeze(end_vectors_copied[:, :, -1], dim=2)
                
            if idx == 0:
                cmf_pro = (end_vectors_copied[:, :, :2] - center_vectors[:, :, :2])
            else:
                cmf_pro_ = (end_vectors_copied[:, :, :2] - center_vectors[:, :, :2])
                cmf_pro = torch.cat((cmf_pro, cmf_pro_), dim=2)
            
        for idx, H_pre in enumerate(H_pre_list):
            initial_vectors_copied = torch.einsum('ij, klj -> kli', H_pre, center_vectors)
            initial_vectors_copied = initial_vectors_copied / torch.unsqueeze(initial_vectors_copied[:, :, -1], dim=2)
                
            if idx == 0:
                cmf_pre = (initial_vectors_copied[:, :, :2] - center_vectors[:, :, :2])
            else:
                cmf_pre_ = (initial_vectors_copied[:, :, :2] - center_vectors[:, :, :2])
                cmf_pre = torch.cat((cmf_pre_, cmf_pre), dim=2)
            
        # Concatenate cmf_pro and cmf_pre
        cmf = torch.cat((cmf_pre, cmf_pro), dim=2)

        for i in range(7, 4, -1):
            cmf[:, :, i*2:i*2+2] -= cmf[:, :, i*2-2:i*2]
        
        for j in range(0, 3):
            cmf[:, :, j*2:j*2+2] -= cmf[:, :, j*2+2:j*2+4]
    
        cmf[:, :, 0:8] = -cmf[:, :, 0:8]

        cmf /= 4  # Rescale cmf vectors according to the downscaling ratio
        cmf = torch.unsqueeze(cmf.permute(2, 0, 1), dim=0)

        deblurred = model(blur, cmf, cmf, 100)

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        if not os.path.exists(os.path.join(args.save_dir, 'deblurred')):
            os.mkdir(os.path.join(args.save_dir, 'deblurred'))
        Image.fromarray((((deblurred.clamp(0, 1))**(1/2.2)) * 255).type(torch.ByteTensor)[0].detach().cpu().numpy().transpose(1, 2, 0)).save(os.path.join(args.save_dir, 'deblurred', f'{idx_+1}'.zfill(6) + '.png'))
        if args.save_blur:
            if not os.path.exists(os.path.join(args.save_dir, 'blur')):
                os.mkdir(os.path.join(args.save_dir, 'blur'))
            Image.fromarray((((blur.clamp(0, 1))**(1/2.2)) * 255).type(torch.ByteTensor)[0].detach().cpu().numpy().transpose(1, 2, 0)).save(os.path.join(args.save_dir, 'blur', f'{idx_+1}'.zfill(6) + '.png'))
        del deblurred


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python test.py --dir /path/to/your/real/data --ckpt /path/to/your/ckpt.pt --save_dir /path/to/your/save/dir --save_blur
    # Example
    #      CUDA_VISIBLE_DEVICES=0 python test.py --dir /root/data/GyroBlur_real  \
    #                                            --ckpt /root/workspace/GyroDeblurNet/ckpt_gyroblur_synth_epoch_300.pt \
    #                                            --save_dir /root/workspace/GyroDeblurNet/real_data/result \
    #                                            --save_blur 

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='Directory of GyroBlur-Real data')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--save_blur', action='store_true')  # Whether to save blurry image along with the deblurred result
    args = parser.parse_args()
    
    optimize_gyro_data(args)