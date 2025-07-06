from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
import torch
import torchvision
import os
from glob import glob
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
from model import GyroDeblurNet


def main():
    psnr = 0.0
    ssim = 0.0
    time = 0.0
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/root/data/heeminid/GyroBlur-Synth/test/', help='Directory to your data')
    parser.add_argument('--ckpt', default='ckpt_gyroblur_synth_epoch_300.pt', help='Location of the pre-trained checkpoint')
    args = parser.parse_args()

    blur_imgs = sorted(glob(os.path.join(args.data_dir, 'avg_blur', '*')))
    sharp_imgs = sorted(glob(os.path.join(args.data_dir, 'sharp', '*')))
    cmf_files = sorted(glob(os.path.join(args.data_dir, 'camera_motion_field', '*')))
    iter = len(blur_imgs)
    
    # Load model
    checkpoint = torch.load(args.ckpt, map_location='cuda')
    
    img_channel = 3
    width = 32
    enc_blks = [2, 2, 2]
    middle_blk_num = 16
    dec_blks = [1, 1, 1]
    model = GyroDeblurNet(img_channel=img_channel, width=width, enc_blk_nums=enc_blks, middle_blk_num=middle_blk_num, dec_blk_nums=dec_blks)

    # new_state_dict = {}
    # for key, value in checkpoint['model_state_dict'].items():
    #     new_key = key.replace('temp_align_blk', 'gyro_refine_blk') \
    #                  .replace('middle_gyro_blk', 'gyro_deblurring_blk')
    #     new_state_dict[new_key] = value

    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    del checkpoint

    # GPU warm-up
    dummy_img = torch.randn(1, 3, 720, 1280, dtype=torch.float).cuda()
    dummy_mgv = torch.randn(1, 16, 360, 640, dtype=torch.float).cuda()
    for _ in range(10):
        _ = model(dummy_img, dummy_mgv, dummy_mgv, 100)
        
    del dummy_img
    del dummy_mgv
        
    # Set up loggers
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    # Test iteration
    with torch.no_grad():
        for i in tqdm(range(iter)):
            blurred_file = blur_imgs[i]
            sharp_file = sharp_imgs[i]
            cmf_file = cmf_files[i]

            blurred = torch.unsqueeze(torchvision.transforms.functional.to_tensor(Image.open(blurred_file)).float().cuda(), dim=0)
            sharp = np.asarray(Image.open(sharp_file))
            dmgv = torch.unsqueeze(torch.as_tensor(np.load(cmf_file).transpose(2, 0, 1)), dim=0).float().cuda()
        
            # Get result of the model and compute PSNR
            starter.record()
            result = model(blurred, dmgv, dmgv, 100)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            result = (result.clamp(0, 1) * 255).type(torch.ByteTensor)[0].detach().cpu().numpy().transpose(1, 2, 0)
            curr_psnr = PSNR(sharp, result)
            curr_ssim = SSIM(sharp, result, channel_axis=2)
            with open('gyrodeblurnet_psnr.txt', 'a') as f:
                f.write(f'{curr_psnr}\n')
            psnr += curr_psnr
            ssim += curr_ssim
            time += (curr_time / 1000)
            
            result = result.astype(float)
            result = result / 255.
            result = result ** (1/2.2)
            result = result * 255
            result = result.astype(np.uint8)
            
    
    print("==========================================================")
    print(f"Average PSNR: {psnr/iter:.2f} dB")
    print(f"Average SSIM: {ssim/iter:.4f}")
    print(f"Average inference time: {time/iter:.4f} s")
    print(f"The number of model parameters: {sum(p.numel() for p in model.parameters())}")
    
    
if __name__ == '__main__':
    main()