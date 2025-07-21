import os
import glob
import argparse

from tqdm import tqdm

import numpy as np
from opt_einsum import contract as einsum
import multiprocessing as mp
import time


def compute_rotation_matrix(ang_vel_x, ang_vel_y, ang_vel_z):
    '''
    Compute rotation matrix using given angular velocities.
    ang_vel_x: Angular velocity of x-axis.
    ang_vel_y: Angular velocity of y-axis.
    ang_vel_z: Angular velocity of z-axis.
    '''
    
    R_x = np.array([
        [                   1,                   0,                   0],
        [                   0,  np.cos(-ang_vel_x), -np.sin(-ang_vel_x)],
        [                   0,  np.sin(-ang_vel_x),  np.cos(-ang_vel_x)]
    ])
    R_y = np.array([
        [  np.cos(-ang_vel_y),                   0,  np.sin(-ang_vel_y)],
        [                   0,                   1,                   0],
        [ -np.sin(-ang_vel_y),                   0,  np.cos(-ang_vel_y)]
    ])
    R_z = np.array([
        [   np.cos(ang_vel_z),   np.sin(ang_vel_z),                   0],
        [  -np.sin(ang_vel_z),   np.cos(ang_vel_z),                   0],
        [                   0,                   0,                   1]
    ])
    R = R_x @ R_y @ R_z
    
    return R


'''
4KRD dataset intrinsic matrix
'''
def compute_homography(R):
    '''
    Compute homography matrix using the given rotation matrix R.
    R: Rotation matrix.
    '''
    
    # K: Camera intrinsic matrix for 4KRD dataset. For different dataset, change the intrinsic matrix accordingly.
    K = np.array([
        [   2139.03493,            0,          960],
        [            0,   2133.82853,          540],
        [            0,            0,            1]
    ])
    
    return K @ R @ np.linalg.inv(K)


def generate_camera_motion_field(save_file, mode, save_root_dir, gyro_num=10):
    starting_point = ...  # Choose the starting point based on the timestamp

    interp_factor = 8
    R_list = []
    for idx, i in enumerate(range(starting_point, starting_point + gyro_num)):
        # Gyro data interpolation
        gap_x = (ang_vel_x[i+1] - ang_vel_x[i]) / interp_factor
        gap_y = (ang_vel_y[i+1] - ang_vel_y[i]) / interp_factor
        gap_z = (ang_vel_z[i+1] - ang_vel_z[i]) / interp_factor
        
        if idx % 5 == 4:
            continue
        # Compute rotation matrix
        R = compute_rotation_matrix((ang_vel_x[i] + (gap_x*2*(idx%5))) * ((timestamp[i+1] - timestamp[i]) * (10/8) * 1e-9),
                                    (ang_vel_y[i] + (gap_y*2*(idx%5))) * ((timestamp[i+1] - timestamp[i]) * (10/8) * 1e-9),
                                    (ang_vel_z[i] + (gap_z*2*(idx%5))) * ((timestamp[i+1] - timestamp[i]) * (10/8) * 1e-9))
        R_list.append(R)

    '''
    Generate camera motion fields.
    H_pro: Homography matirx from the center frame to the end frame.
    H_pre: Homography matrix from the center frame to the initial frame.
    '''
    # Compute H_pro (t = T/2 -> (T/2)+1, (T/2)+2, ..., T)
    R = np.eye(3)
    H_pro_list = []
    for i in range(len(R_list)//2, len(R_list)):
        R = R_list[i] @ R
        H = compute_homography(R)
        H_pro_list.append(H)
    
    # Compute H_pre (t = T/2 -> (T/2)-1, (T/2)-2, ..., 0)
    R = np.eye(3)
    H_pre_list = []
    for i in range((len(R_list)//2)-1, -1, -1):
        R = R @ R_list[i]
        H = np.linalg.inv(compute_homography(R))
        H_pre_list.append(H)
    
    cmf_pro = None  # Camera motion field (Pro)
    for H_pro in H_pro_list:
        end_vectors_copied = einsum('ij, klj -> kli', H_pro, center_vectors)
        end_vectors_copied = end_vectors_copied / end_vectors_copied[:, :, -1, np.newaxis]
        
        if cmf_pro is None:
            vector = (end_vectors_copied[:, :, :2] - center_vectors[:, :, :2])
            cmf_pro = vector.copy()
        else:
            vector = (end_vectors_copied[:, :, :2] - center_vectors[:, :, :2])
            cmf_pro = np.concatenate((cmf_pro, vector), axis=2)
    
    cmf_pre = None  # Camera motion field (Pre)
    for H_pre in H_pre_list:
        initial_vectors_copied = einsum('ij, klj -> kli', H_pre, center_vectors)
        initial_vectors_copied = initial_vectors_copied / initial_vectors_copied[:, :, -1, np.newaxis]
        
        if cmf_pre is None:
            vector = (initial_vectors_copied[:, :, :2] - center_vectors[:, :, :2])
            cmf_pre = vector.copy()
        else:
            vector = (initial_vectors_copied[:, :, :2] - center_vectors[:, :, :2])
            cmf_pre = np.concatenate((vector, cmf_pre), axis=2)
    
    # Concatenate cmf_pro and cmf_pre
    cmf = np.concatenate((cmf_pre, cmf_pro), axis=2)
    
    for i in range(7, 4, -1):
        cmf[:, :, i*2:i*2+2] -= cmf[:, :, i*2-2:i*2]
    
    for j in range(0, 3):
        cmf[:, :, j*2:j*2+2] -= cmf[:, :, j*2+2:j*2+4]
    
    cmf[:, :, 0:8] = -cmf[:, :, 0:8]
    
    # Save computed camera motion field
    np.save(os.path.join(save_root_dir, mode, 'camera_motion_field', f'{save_file}'), cmf.astype('float32'))


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], required=True)
parser.add_argument('--save_root_dir', required=True)
args = parser.parse_args()
    
center_vectors = []

# Homogeneous coordinates
# Same as the resolution of original sharp image
for i in range(1080):
    row = []
    for j in range(1920):
        row.append([j, i, 1])
    center_vectors.append(row)
center_vectors = np.array(center_vectors).astype(np.float64)

np.save('center_vectors', center_vectors)

center_vectors = np.load('center_vectors.npy')[180:900, 320:1600, :][0::2, 0::2, :]  # After averaging warped images, we center-crop 1080 x 1920 to 720 x 1280.
                                                                                     # Therefore, `center_vectors` also needs to be cropped accordingly.

with open(f'gyro_{args.mode}.txt', 'r') as f:
    lines = f.readlines()

'''
IMPORTANT

When dealing with gyro data collected from real-world device, you should consider axis of the gyro sensor.
Sign of the axis varies depending on the device.
'''
ang_vel_x = []
ang_vel_y = []
ang_vel_z = []
timestamp = []
for line in lines:
    t, x, y, z = line.split()
    ang_vel_x.append(float(x))
    ang_vel_y.append(float(y))
    ang_vel_z.append(float(z))
    timestamp.append(int(t))
ang_vel_x = np.array(ang_vel_x)
ang_vel_y = np.array(ang_vel_y)
ang_vel_z = np.array(ang_vel_z)
timestamp = np.array(timestamp)

if args.mode == 'train':
    image_files = np.random.randn(14600)
else:
    image_files = np.random.randn(640)

for i in tqdm(range(len(image_files))):
    generate_camera_motion_field(f'{i+1}'.zfill(6), args.mode, args.save_root_dir)