import math
import os
import sys
import numpy as np
import torch
import argparse
import torch.nn as nn
from torch.utils import data
from datasets import ViperLoader
import torch.nn.functional as F
from tqdm import tqdm
import datasets
sys.path.append('core')
from core.network import RAFTGMA


def warp(x, flo, padding_mode='border'):
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid - flo
    
    # Scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='bilinear')
    return output

@torch.no_grad()
def validate_viper(model, val_loader, iters=6, wauc_bins=100):
    """ Peform validation using the (official) Viper validation split"""

    model.eval()

    out_list, epe_list, wauc_list = [], [], []

    for data_blob in tqdm(val_loader):
        image1, image2, flow_gt, valid_gt = [x.cuda() for x in data_blob]

        # flow_gt = flow_gt[0]
        # valid_gt = valid_gt[0]

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = flow_pr[0]

        # epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        # mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
        mag = torch.sum(flow_gt**2, dim=1).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt >= 0.5
        # val_e = val & ((flow_gt[0].abs() < 1000) & (flow_gt[1].abs() < 1000))
        val_e = val & ((flow_gt[:,0,...].abs() < 1000) & (flow_gt[:,1,...].abs() < 1000))
        val_e = val_e.reshape(-1)
        val = val.reshape(-1)

        # weighted area under curve
        wauc = 0.0
        w_total = 0.0
        for i in range(1, wauc_bins + 1):
            w = 1.0 - (i - 1.0) / wauc_bins
            d = 5 * (i / wauc_bins)

            wauc += w * (epe[val] <= d).float().mean()
            w_total += w

        wauc = (100.0 / w_total) * wauc

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val_e].mean().cpu().item())
        out_list.append(out[val_e].cpu().numpy())
        wauc_list.append(wauc.cpu().item())

    wauc_list = np.array(wauc_list)
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    wauc = np.mean(wauc_list)
    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print(f"Viper: iters: {sum(iters)}, wauc: {wauc} ({wauc_bins} bins), epe: {epe}, f1: {f1}")
    return {"viper-epe": epe, "viper-f1": f1, f"viper-wauc-{wauc_bins}": wauc}

def cal_warping_error(source_frames, translated_frames, flows, alpha=0.075):
    error = 0
    for frame_index in range(1, len(source_frames)):
        prev_source_frame = source_frames[frame_index - 1]
        current_source_frame = source_frames[frame_index].cpu().float().numpy()

        prev_translated_frame = translated_frames[frame_index - 1]
        current_translated_frame = translated_frames[frame_index].cpu().float().numpy()

        warped_source_frame = warp(prev_source_frame, flows[frame_index - 1]).cpu().float().numpy()
        warped_translated_frame = warp(prev_translated_frame, flows[frame_index - 1]).cpu().float().numpy()

        source_dis = math.exp(-1 * alpha * np.linalg.norm(warped_source_frame - current_source_frame))
        translated_dis = np.sum(np.abs(current_translated_frame - warped_translated_frame))

        del prev_source_frame, current_source_frame, prev_translated_frame, current_translated_frame
        del warped_source_frame, warped_translated_frame
        
        warping_error = source_dis * translated_dis
        error += warping_error

    return error


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--batch_size', type=int, default=6)
    # parser.add_argument('--image_size', type=int, nargs='+', default=[256, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    # parser.add_argument('--output', type=str, default='checkpoints', help='output directory to save checkpoints and plots')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    # if not os.path.isdir(args.output):
    #     os.makedirs(args.output)

    data_path = 'data/viper_day_rain'
    path_n = 4
    val_dataset = ViperLoader(data_path, split='val', path_num=path_n)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    model = nn.DataParallel(RAFTGMA(args), device_ids=args.gpus)
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
    model.cuda()

    validate_viper(model, val_loader)



    
