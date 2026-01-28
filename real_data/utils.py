import numpy as np
import math
import torch


def compute_translation_matrix(trans_x, trans_y):
    T = np.array([
        [       1,       0, trans_x],
        [       0,       1, trans_y],
        [       0,       0,       1]
    ], dtype=float)
    
    return T


def compute_translation_matrix_gpu(trans_x, trans_y):
    T = torch.tensor([
        [       1,       0, trans_x],
        [       0,       1, trans_y],
        [       0,       0,       1]
    ], dtype=torch.float32).cuda()
    
    return T


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


def compute_rotation_matrix_gpu(ang_vel_x, ang_vel_y, ang_vel_z):
    '''
    Compute rotation matrix using given angular velocities.
    ang_vel_x: Angular velocity of x-axis.
    ang_vel_y: Angular velocity of y-axis.
    ang_vel_z: Angular velocity of z-axis.
    '''
    
    R_x = torch.tensor([
        [                   1,                   0,                   0],
        [                   0,  torch.cos(-ang_vel_x), -torch.sin(-ang_vel_x)],
        [                   0,  torch.sin(-ang_vel_x),  torch.cos(-ang_vel_x)]
    ], dtype=torch.float32).cuda()
    R_y = torch.tensor([
        [  torch.cos(-ang_vel_y),                   0,  torch.sin(-ang_vel_y)],
        [                   0,                   1,                   0],
        [ -torch.sin(-ang_vel_y),                   0,  torch.cos(-ang_vel_y)]
    ], dtype=torch.float32).cuda()
    R_z = torch.tensor([
        [   torch.cos(ang_vel_z),   torch.sin(ang_vel_z),                   0],
        [  -torch.sin(ang_vel_z),   torch.cos(ang_vel_z),                   0],
        [                   0,                   0,                   1]
    ], dtype=torch.float32).cuda()
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
    
    # K: Camera intrinsic matrix for 4KRD dataset
    # K = np.array([
    #     [   2139.03493,            0,         1920],
    #     [            0,   2133.82853,         1080],
    #     [            0,            0,            1]
    # ])
    K = np.array([
        [   2139.03493,            0,          960],
        [            0,   2133.82853,          540],
        [            0,            0,            1]
    ])
    
    return K @ R @ np.linalg.inv(K)


def compute_homography_gpu(R):
    '''
    Compute homography matrix using the given rotation matrix R.
    R: Rotation matrix.
    '''
    
    # K: Camera intrinsic matrix for 4KRD dataset
    # K = np.array([
    #     [   2139.03493,            0,         1920],
    #     [            0,   2133.82853,         1080],
    #     [            0,            0,            1]
    # ])
    K = torch.tensor([
        [   2139.03493,            0,          960],
        [            0,   2133.82853,          540],
        [            0,            0,            1]
    ], dtype=torch.float32).cuda()
    
    return K @ R @ torch.linalg.inv(K)


def PSNR(output, sharp):
    mse = np.mean((output - sharp) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                   # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


""" Get Perspective Projection Matrix """
def get_H(theta, phi, gamma, dx, dy, dz, height, width, focal):
    '''
    theta: theta_x
    phi: theta_y
    gamma: theta_z
    '''
    
    # Axis transformation (Galaxy S22)
    gamma = -gamma
        
    w = width
    h = height
    f = focal

    # Projection 2D -> 3D matrix
    A1 = np.array([ [1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 1],
                    [0, 0, 1]])
        
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
        
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
        
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([ [f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))