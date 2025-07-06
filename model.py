import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from arch_util import LayerNorm2d, DeformableConv2d, IdentityGyro, IdentityTwo


class ChannelPool(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.cat((torch.max(x, dim=1)[0].unsqueeze(1), torch.mean(x, dim=1).unsqueeze(1)), dim=1)
    

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding='same', dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_pool = self.pool(x)
        x_out = self.conv(x_pool)
        scale = self.sigmoid(x_out)

        return x * scale


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand

        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # 3x3 depthwise conv
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        # 1x1 conv
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c

        # 1x1 conv
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # 1x1 conv
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # LayerNorm
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
            
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class GyroRefinementBlock(nn.Module):  # Previously GyroTemporalAlignBlock
    def __init__(self, c_gyro, c_blur):
        super().__init__()

        self.conv_ca_weight = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=c_gyro+c_blur, out_channels=c_gyro, kernel_size=3, stride=1, padding=1)
                # nn.Conv2d(in_channels=c_gyro+c_blur, out_channels=c_gyro, kernel_size=1, stride=1)
            )
        self.conv_gyro = nn.Conv2d(in_channels=c_gyro, out_channels=c_gyro, kernel_size=3, stride=1, padding=1)
        self.conv_down = nn.Conv2d(in_channels=c_gyro, out_channels=2*c_gyro, kernel_size=3, stride=2, padding=1)

    def forward(self, blur, gyro):
        '''
        Input
            gyro: (B, c_gyro, H, W) shaped torch.tensor
            blur: (B, c_blur, H, W) shaped torch.tensor
        
        Output
            gyro: (B, 2*c_gyro, H/2, W/2) shaped torch.tensor
        '''

        ca_weight = self.conv_ca_weight(torch.cat((blur, gyro), dim=1))  # (B, c_gyro, 1, 1)
        gyro = gyro * ca_weight  # (B, c_gyro, H, W)
        gyro = F.relu(self.conv_gyro(gyro))  # (B, c_gyro, H, W)
        gyro = self.conv_down(gyro)  # (B, 2*c_gyro, H/2, W/2)

        return gyro


class GyroBlock(nn.Module):
    def __init__(self, channel_blur, channel_gyro):
        super().__init__()
        
        self.conv_deform = DeformableConv2d(in_channels_blur=channel_blur, in_channels_gyro=channel_gyro, out_channels=channel_blur)

    def forward(self, feat_blur, feat_gyro):
        '''
        Input
            feat_blur: (B, 256, H/8, W/8)
            feat_gyro: (B, 256, H/8, W/8)
        
        Output
            feat_blur: (B, 256, H/8, W/8)
        '''

        feat = self.conv_deform(feat_blur, feat_gyro)
        
        return feat, feat_gyro


class GyroDeblurringBlock(nn.Module):  # Previously, ConcatBlock
    def __init__(self, channel):
        super().__init__()
        
        self.gyro_block = GyroBlock(channel, channel)
        self.naf_block = NAFBlock(channel)
        self.spatial_attn = SpatialAttention()
        self.conv = nn.Conv2d(in_channels=channel*2, out_channels=channel, kernel_size=3, stride=1, padding=1)

    def forward(self, feat_blur, feat_gyro):
        '''
        Input
            feat_wo_gyro: (B, feat_blur, H, W) shaped torch.tensor
            feat_w_gyro: (B, feat_gyro, H, W) shaped torch.tensor
        
        Output
            feat: (B, feat_gyro, H, W) shaped torch.tensor
        '''

        feat_, feat_gyro = self.gyro_block(feat_blur, feat_gyro)
        feat = self.spatial_attn(feat_)
        feat_refined = self.naf_block(feat)
        feat = torch.cat((feat_, feat_refined), dim=1)
        feat = self.conv(feat)

        return feat, feat_gyro
    

class GyroDeblurNet(nn.Module):  # Previously, NAFNet
    def __init__(self, img_channel=3, gyro_channel=16, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.intro_gyro = nn.Conv2d(in_channels=gyro_channel, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.gyro_refine_blks = nn.ModuleList()  # Previously, temp_align_blks
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.gyro_deblurring_blks = nn.ModuleList()  # Previously, middle_gyro_blks
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        
        self.gyro_refine_blks.append(IdentityGyro())
        self.gyro_refine_blks.append(GyroRefinementBlock(c_gyro=64, c_blur=64))
        self.gyro_refine_blks.append(GyroRefinementBlock(c_gyro=128, c_blur=128))

        for _ in range(4):
            self.middle_blks.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num//4)])
            )
        
        for _ in range(3):
            self.gyro_deblurring_blks.append(
                GyroDeblurringBlock(256)
            )
        self.gyro_deblurring_blks.append(IdentityTwo())
            
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, gyro_acc, gyro_inacc, epoch):
        _, _, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)     # B x 32 x H x W

        encs = []

        # x_gyro = self.intro_gyro(gyro)  # B x 64 x H/2 x W/2
        x_gyro_acc = self.intro_gyro(gyro_acc)  # B x 64 x H/2 x W/2
        x_gyro_inacc = self.intro_gyro(gyro_inacc)
        alpha = (epoch//10)*0.1 if epoch < 100 else 1
        x_gyro = x_gyro_inacc*alpha + x_gyro_acc*(1-alpha)

        for idx, (encoder, down, gyro_refine_blk) in enumerate(zip(self.encoders, self.downs, self.gyro_refine_blks)):
            x = encoder(x)      
            encs.append(x)

            if idx != 0:
                x_gyro = gyro_refine_blk(x, x_gyro)

            x = down(x)         

        for idx, (middle_blk, gyro_deblurring_blk) in enumerate(zip(self.middle_blks, self.gyro_deblurring_blks)):
            x = middle_blk(x)
            x, x_gyro = gyro_deblurring_blk(x, x_gyro)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [2, 2, 2]
    middle_blk_num = 16
    dec_blks = [1, 1, 1]
    
    net = GyroDeblurNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    
    print(sum(p.numel() for p in net.parameters()))
    
    img = torch.randn(1,3,256,256).cuda()
    gyro = torch.randn(1,16,128,128).cuda()
    out = net(img, gyro, gyro, 100)
    print(out.shape)