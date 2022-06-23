#!/usr/bin/env python
# encoding: utf-8

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

import torch
import torch.nn.functional as F

from neural_renderer import ARNet, IUNet

from classical_renderer.scatter import ModuleRenderScatter  # circular aperture
from classical_renderer.scatter_ex import ModuleRenderScatterEX  # adjustable aperture shape


def gaussian_blur(x, r, sigma=None):
    r = int(round(r))
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8
    x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r) + 1), torch.arange(-int(r), int(r) + 1))
    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
    kernel = kernel.float() / kernel.sum()
    kernel = kernel.expand(1, 1, 2*r+1, 2*r+1).to(x.device)
    x = F.pad(x, pad=(r, r, r, r), mode='replicate')
    x = F.conv2d(x, weight=kernel, padding=0)
    return x


def pipeline(classical_renderer, arnet, iunet, image, defocus, gamma, args):
    bokeh_classical, defocus_dilate = classical_renderer(image**gamma, defocus*args.defocus_scale)
    # bokeh_classical, defocus_dilate = classical_renderer_ex(image**gamma, defocus*args.defocus_scale, poly_sides=6)

    bokeh_classical = bokeh_classical ** (1/gamma)
    defocus_dilate = defocus_dilate / args.defocus_scale
    gamma = (gamma - args.gamma_min) / (args.gamma_max - args.gamma_min)
    adapt_scale = max(defocus.abs().max().item(), 1)

    image_re = F.interpolate(image, scale_factor=1/adapt_scale, mode='bilinear', align_corners=True)
    defocus_re = 1 / adapt_scale * F.interpolate(defocus, scale_factor=1/adapt_scale, mode='bilinear', align_corners=True)
    bokeh_neural, error_map = arnet(image_re, defocus_re, gamma)
    error_map = F.interpolate(error_map, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
    bokeh_neural.clamp_(0, 1e5)

    if args.save_intermediate:
        cv2.imwrite(os.path.join(save_root, 'bokeh_neural_s0.jpg'), bokeh_neural[0].cpu().permute(1, 2, 0).numpy()[..., ::-1] * 255)

    scale = -1
    for scale in range(int(np.log2(adapt_scale))):
        ratio = 2**(scale+1) / adapt_scale
        h_re, w_re = int(ratio * image.shape[2]), int(ratio * image.shape[3])
        image_re = F.interpolate(image, size=(h_re, w_re), mode='bilinear', align_corners=True)
        defocus_re = ratio * F.interpolate(defocus, size=(h_re, w_re), mode='bilinear', align_corners=True)
        defocus_dilate_re = ratio * F.interpolate(defocus_dilate, size=(h_re, w_re), mode='bilinear', align_corners=True)
        bokeh_neural_refine = iunet(image_re, defocus_re.clamp(-1, 1), bokeh_neural, gamma).clamp(0, 1e5)
        mask = gaussian_blur(((defocus_dilate_re < 1) * (defocus_dilate_re > -1)).float(), 0.005 * (defocus_dilate_re.shape[2] + defocus_dilate_re.shape[3]))
        bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(h_re, w_re), mode='bilinear', align_corners=True)
        if args.save_intermediate:
            cv2.imwrite(os.path.join(save_root, f'bokeh_neural_s{scale+1}.jpg'), bokeh_neural[0].cpu().permute(1, 2, 0).numpy()[..., ::-1] * 255)
            cv2.imwrite(os.path.join(save_root, f'fmask_neural_s{scale+1}.jpg'), mask[0][0].cpu().numpy() * 255)

    bokeh_neural_refine = iunet(image, defocus.clamp(-1, 1), bokeh_neural, gamma).clamp(0, 1e5)
    mask = gaussian_blur(((defocus_dilate < 1) * (defocus_dilate > -1)).float(), 0.005 * (defocus_dilate.shape[2] + defocus_dilate.shape[3]))
    bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
    if args.save_intermediate:
        cv2.imwrite(os.path.join(save_root, f'bokeh_neural_s{scale+2}.jpg'), bokeh_neural[0].cpu().permute(1, 2, 0).numpy()[..., ::-1] * 255)
        cv2.imwrite(os.path.join(save_root, f'fmask_neural_s{scale+2}.jpg'), mask[0][0].cpu().numpy() * 255)

    bokeh_pred = bokeh_classical * (1 - error_map) + bokeh_neural * error_map

    return bokeh_pred.clamp(0, 1), bokeh_classical.clamp(0, 1), bokeh_neural.clamp(0, 1), error_map




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Bokeh Rendering', fromfile_prefix_chars='@')


parser.add_argument('--defocus_scale',               type=float, default=10.)
parser.add_argument('--gamma_min',                   type=float, default=1.)
parser.add_argument('--gamma_max',                   type=float, default=5.)

# Model 1
parser.add_argument('--arnet_shuffle_rate',          type=int,   default=2)
parser.add_argument('--arnet_in_channels',           type=int,   default=5)
parser.add_argument('--arnet_out_channels',          type=int,   default=4)
parser.add_argument('--arnet_middle_channels',       type=int,   default=128)
parser.add_argument('--arnet_num_block',             type=int,   default=3)
parser.add_argument('--arnet_share_weight',                      action='store_true')
parser.add_argument('--arnet_connect_mode',          type=str,   default='distinct_source')
parser.add_argument('--arnet_use_bn',                            action='store_true')
parser.add_argument('--arnet_activation',            type=str,   default='elu')

# Model 2
parser.add_argument('--iunet_shuffle_rate',          type=int,   default=2)
parser.add_argument('--iunet_in_channels',           type=int,   default=8)
parser.add_argument('--iunet_out_channels',          type=int,   default=3)
parser.add_argument('--iunet_middle_channels',       type=int,   default=64)
parser.add_argument('--iunet_num_block',             type=int,   default=3)
parser.add_argument('--iunet_share_weight',                      action='store_true')
parser.add_argument('--iunet_connect_mode',          type=str,   default='distinct_source')
parser.add_argument('--iunet_use_bn',                            action='store_true')
parser.add_argument('--iunet_activation',            type=str,   default='elu')

# Checkpoint
parser.add_argument('--arnet_checkpoint_path',       type=str,   default='./checkpoints/arnet.pth')
parser.add_argument('--iunet_checkpoint_path',       type=str,   default='./checkpoints/iunet.pth')

# Input
parser.add_argument('--image_path',                  type=str,   default='./inputs/21.jpg')
parser.add_argument('--disp_path',                   type=str,   default='./inputs/21.png')
parser.add_argument('--save_dir',                    type=str,   default='./outputs')
parser.add_argument('--K',                           type=float, default=60,          help='blur parameter')
parser.add_argument('--disp_focus',                  type=float, default=90/255,      help='refocused disparity (0~1)')
parser.add_argument('--gamma',                       type=float, default=4,           help='gamma value (1~5)')

parser.add_argument('--highlight',                               action='store_true', help='forcibly enchance RGB values of highlights')
parser.add_argument('--highlight_RGB_threshold',     type=float, default=220/255)
parser.add_argument('--highlight_enhance_ratio',     type=float, default=0.4)

parser.add_argument('--save_intermediate',                       action='store_true', help='save intermediate results')

args = parser.parse_args()

arnet_checkpoint_path = args.arnet_checkpoint_path
iunet_checkpoint_path = args.iunet_checkpoint_path

classical_renderer = ModuleRenderScatter().to(device)
# classical_renderer_ex = ModuleRenderScatterEX().to(device)

arnet = ARNet(args.arnet_shuffle_rate, args.arnet_in_channels, args.arnet_out_channels, args.arnet_middle_channels,
              args.arnet_num_block, args.arnet_share_weight, args.arnet_connect_mode, args.arnet_use_bn, args.arnet_activation)
iunet = IUNet(args.iunet_shuffle_rate, args.iunet_in_channels, args.iunet_out_channels, args.iunet_middle_channels,
              args.iunet_num_block, args.iunet_share_weight, args.iunet_connect_mode, args.iunet_use_bn, args.iunet_activation)

arnet.cuda()
iunet.cuda()

checkpoint = torch.load(arnet_checkpoint_path)
arnet.load_state_dict(checkpoint['model'])
checkpoint = torch.load(iunet_checkpoint_path)
iunet.load_state_dict(checkpoint['model'])

arnet.eval()
iunet.eval()

save_root = os.path.join(args.save_dir, os.path.splitext(os.path.basename(args.image_path))[0])
os.makedirs(save_root, exist_ok=True)

K = args.K                     # blur parameter
disp_focus = args.disp_focus   # 0~1
gamma = args.gamma             # 1~5

image = cv2.imread(args.image_path).astype(np.float32) / 255.0
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_ori = image.copy()

disp = np.float32(cv2.imread(args.disp_path, cv2.IMREAD_GRAYSCALE))
disp = (disp - disp.min()) / (disp.max() - disp.min())

########## Highlights ##########
if args.highlight:
    mask1 = np.clip(np.tanh(200 * (np.abs(disp - disp_focus)**2 - 0.01)), 0, 1)[..., np.newaxis]  # out-of-focus areas
    # mask2 = (np.max(image, axis=2, keepdims=True) > args.highlight_RGB_threshold)  # highlight areas
    mask2 = np.clip(np.tanh(10*(image - args.highlight_RGB_threshold)), 0, 1)    # highlight areas
    mask = mask1 * mask2
    image = image * (1 + mask * args.highlight_enhance_ratio)
################################


defocus = K * (disp - disp_focus) / args.defocus_scale

with torch.no_grad():
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    defocus = torch.from_numpy(defocus).unsqueeze(0).unsqueeze(0)
    image = image.cuda()
    defocus = defocus.cuda()

    bokeh_pred, bokeh_classical, bokeh_neural, error_map = pipeline(
        classical_renderer, arnet, iunet, image, defocus, gamma, args
    )


defocus = defocus[0][0].cpu().numpy()
error_map = error_map[0][0].cpu().numpy()
bokeh_classical = bokeh_classical[0].cpu().permute(1, 2, 0).numpy()
bokeh_neural = bokeh_neural[0].cpu().permute(1, 2, 0).detach().numpy()
bokeh_pred = bokeh_pred[0].cpu().permute(1, 2, 0).detach().numpy()

cv2.imwrite(os.path.join(save_root, 'image.jpg'), image_ori[..., ::-1] * 255)
plt.imsave(os.path.join(save_root, 'defocus.jpg'), defocus, cmap='coolwarm', vmin=-max(defocus.max(), -defocus.min()), vmax=max(defocus.max(), -defocus.min()))
cv2.imwrite(os.path.join(save_root, 'disparity.jpg'), disp * 255)
cv2.imwrite(os.path.join(save_root, 'error_map.jpg'), error_map * 255)
cv2.imwrite(os.path.join(save_root, 'bokeh_classical.jpg'), bokeh_classical[..., ::-1] * 255)
cv2.imwrite(os.path.join(save_root, 'bokeh_neural.jpg'), bokeh_neural[..., ::-1] * 255)
cv2.imwrite(os.path.join(save_root, 'bokeh_pred.jpg'), bokeh_pred[..., ::-1] * 255)
