import torch
import argparse
from model import GyroDeblurNet
from glob import glob
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import os
import scipy


class GyroBlurDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.blur_imgs = sorted(glob(os.path.join(args.blur_dir, '*.png')))
        self.sharp_imgs = sorted(glob(os.path.join(args.sharp_dir, '*.png')))
        self.sat_mask_files = sorted(glob(os.path.join(args.sat_dir, '*.png')))
        self.cmf_files_inacc = sorted(glob(os.path.join(args.cmf_inacc_dir, '*.npy')))
        self.cmf_files_acc = sorted(glob(os.path.join(args.cmf_acc_dir, '*.npy')))

        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __len__(self):
        return len(self.blur_imgs)

    def __getitem__(self, idx):
        '''
        Image cropping parameter
        '''
        H = 720
        W = 1280
        x_crop_start = random.randint(0, W - 256)
        x_crop_end = x_crop_start + 256
        y_crop_start = random.randint(0, H - 256)
        y_crop_end = y_crop_start + 256

        blur = torch.from_numpy((np.asarray(Image.open(self.blur_imgs[idx])).astype(float) / 255))
        blur = blur[y_crop_start:y_crop_end, x_crop_start:x_crop_end, :]  # Blur cropping
        sharp = torch.as_tensor((np.asarray(Image.open(self.sharp_imgs[idx])).transpose(2, 0, 1).astype(float) / 255))
        sharp = sharp[:, y_crop_start:y_crop_end, x_crop_start:x_crop_end]
        sat_mask = torch.as_tensor((np.asarray(Image.open(self.sat_mask_files[idx])).astype(float) / 255))
        sat_mask = sat_mask[y_crop_start:y_crop_end, x_crop_start:x_crop_end, :]

        '''
        Load Camera Motion Field
        '''
        cmf_inacc = torch.as_tensor(np.load(self.cmf_files_inacc[idx]).transpose(2, 0, 1)[:, y_crop_start//2:y_crop_end//2, x_crop_start//2:x_crop_end//2])
        cmf_acc = torch.as_tensor(np.load(self.cmf_files_acc[idx]).transpose(2, 0, 1)[:, y_crop_start//2:y_crop_end//2, x_crop_start//2:x_crop_end//2])
        return blur, sharp, sat_mask, cmf_inacc, cmf_acc


class PSNRLoss(torch.nn.Module):

    def __init__(self):
        super(PSNRLoss, self).__init__()
        self.scale = 10 / np.log(10)

    def forward(self, pred, target):
        assert len(pred.size()) == 4

        return self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blur_dir', required=True)
    parser.add_argument('--sharp_dir', required=True)
    parser.add_argument('--sat_dir', required=True)
    parser.add_argument('--cmf_acc_dir', required=True, help='Directory to accurate camera motion fields (no error)')
    parser.add_argument('--cmf_inacc_dir', required=True, help='Directory to inaccurate camera motion field (erroneous)')
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--loss_file', required=True)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_ckpt')
    parser.add_argument('--port', required=True, type=str)
    parser.add_argument('--save_epoch', default=10, type=int)
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    args.workers = ngpus_per_node * 8

    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = gpu

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    torch.distributed.init_process_group(backend='nccl', rank=args.rank, world_size=args.world_size)

    img_channel = 3
    width = 32
    enc_blks = [2, 2, 2]
    middle_blk_num = 16
    dec_blks = [1, 1, 1]
    model = GyroDeblurNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num, enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model.train()
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    device = torch.device(f'cuda:{args.gpu}')

    criterion = PSNRLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=1e-7)

    if args.resume:
        ckpt = torch.load(args.resume_ckpt, map_location='cuda')
        start_epoch = ckpt['epoch'] + 1
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    else:
        if os.path.exists(args.loss_file):
            if args.rank == (ngpus_per_node - 1):
                os.remove(args.loss_file)
        start_epoch = 0

    train_dataset = GyroBlurDataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  # torch.utils.data.distributed.DistributedSampler automatically shuffles the indices.
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for epoch in tqdm(range(start_epoch, args.epoch)):
        train(model, train_dataloader, criterion, optimizer, epoch, device, scaler, args, ngpus_per_node)

        scheduler.step()

        if args.rank == (ngpus_per_node - 1) and (epoch + 1) % args.save_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, os.path.join(args.checkpoint_dir, f'{epoch + 1}'.zfill(6) + '.pt'))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, os.path.join(args.checkpoint_dir, 'model.pt'))


def train(model, dataloader, criterion, optimizer, epoch, device, scaler, args, ngpus_per_node):
    bar = tqdm(dataloader) if args.rank == (ngpus_per_node - 1) else dataloader
    losses = []

    ccm = torch.from_numpy(scipy.io.loadmat('RSBlur/CCM_matrix.mat')['colorCorrectionMatrix']).float().to(device)
    lin2xyz = torch.from_numpy(scipy.io.loadmat('RSBlur/M_lin2xyz.mat')['M']).float().to(device)
    xyz2lin = torch.from_numpy(scipy.io.loadmat('RSBlur/M_xyz2lin.mat')['M']).float().to(device)

    shot_noise_log_min = -10.000993824378506
    shot_noise_log_max = -9.334882426674499
    noise_slope = 3.15578751
    noise_intercept = 10.00035141523999

    for blur, sharp, sat_mask, cmf_inacc, cmf_acc in bar:
        if args.rank == (ngpus_per_node - 1):
            bar.set_description(f'Epoch {epoch+1}')
            
        blur = blur.to(device, non_blocking=True).float()
        sharp = sharp.to(device, non_blocking=True).float()
        sat_mask = sat_mask.to(device, non_blocking=True).float()
        cmf_inacc = cmf_inacc.to(device, non_blocking=True).float()
        cmf_acc = cmf_acc.to(device, non_blocking=True).float()

        alpha = np.random.uniform(low=3.0, high=5.0)                                            # Saturation synthesis parameter
        shot_noise_log = np.random.uniform(shot_noise_log_min, shot_noise_log_max)              # Randomly sample shot noise in the range between ISO 100 and ISO 1600
        read_noise_log = noise_slope * shot_noise_log + noise_intercept                         # Estimate read noise from the regressed line
        read_noise_log += np.random.normal(0, 0.5 ** 2)                                         # Give randomness to the read noise
        shot_noise = 2 ** shot_noise_log                                                        # Inverse log2
        read_noise = 2 ** read_noise_log                                                        # Inverse log2

        beta_1 = shot_noise                                                                     # Noise synthesis parameter (Shot noise variance)
        beta_2 = read_noise                                                                     # Noise synthesis parameter (Read noise variance)

        '''
        Step 1: Saturation synthesis
        '''
        blur += (alpha * sat_mask)
        blur = torch.clip(blur, min=0, max=1)

        '''
        Step 2: Lin2XYZ
        '''
        blur = torch.matmul(blur, lin2xyz)

        '''
        Step 3: Inverse color correction
        '''
        blur = torch.matmul(blur, torch.linalg.inv(ccm))

        '''
        Step 4: Inverse white balance
        '''
        gain_r  = np.random.uniform(1.9, 2.4)  # Randomly sample gains for the red channels (Refer to the RSBlur paper)
        gain_b = np.random.uniform(1.5, 1.9)   # Randomly sample gains for the blue channels (Refer to the RSBlur paper)
        blur[:, :, 0] *= (1/gain_r)
        blur[:, :, 2] *= (1/gain_b)

        '''
        Step 5: Noise synthesis
        '''
        blur = (torch.poisson(blur / beta_1) * beta_1)  # Shot noise synthesis
        blur += torch.zeros(blur.shape).normal_(mean=0, std=np.sqrt(beta_2)).to(device)  # Read noise synthesis
        
        '''
        Step 6: White balance
        '''
        blur[:, :, 0] *= gain_r
        blur[:, :, 2] *= gain_b
        blur = torch.clip(blur, min=0, max=1)

        '''
        Step 7: Color correction
        '''
        blur = torch.matmul(blur, ccm)

        '''
        Step 8: XYZ2Lin
        '''
        blur = torch.matmul(blur, xyz2lin)
        blur = torch.clip(blur, min=0, max=1)

        blur = blur.permute(0, 3, 1, 2)

        outputs = model(blur, cmf_acc, cmf_inacc, epoch)
        loss = criterion(outputs, sharp)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

        if args.rank == (ngpus_per_node - 1):
            bar.set_postfix({'Train Loss': sum(losses)/len(losses)})
            with open(args.loss_file.split('.')[0] + '_all_loss.txt', 'a') as f:
                f.write(f'{sum(losses)/len(losses)}\n')

    if args.rank == (ngpus_per_node - 1):
        with open(args.loss_file, 'a') as f:
            f.write(f'{sum(losses)/len(losses)}\n')


if __name__ == '__main__':
    main()
