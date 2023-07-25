import os
import sys
import torch
import argparse
import timeit
import numpy as np
import oyaml as yaml
from torch.utils import data
from PIL import Image
from ptsemseg.loss.loss import OhemCELoss2D
from ptsemseg.models import get_model
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
from ptsemseg.loss import get_loss_function
import cv2
import pdb
from tqdm import tqdm
from datasets import ViperLoader
sys.path.append('core')
from core.network import RAFTGMA
from core.utils import flow_viz
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

MAX_FLOW = 400

def convert_flow_to_image(image1, flow):
    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[3], image1.shape[2]))
    return flow_image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def sequence_loss(flow_preds, flow_gt, valid, gamma):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements
    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    return flow_loss

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

def shift_matrix(matrix, shift="r", add=0):

    if shift == "r":
        tmp = np.roll(matrix, 1, axis=2)
        tmp[:,:,0,...] = add
    elif shift == "l":
        tmp = np.roll(matrix, -1, axis=2)
        tmp[:,:,-1,...] = add
    elif shift == "u":
        tmp = np.roll(matrix, -1, axis=1)
        tmp[:,-1,...] = add
    elif shift == "d":
        tmp = np.roll(matrix, 1, axis=1)
        tmp[:,0,...] = add
    if shift == "r+d":
        tmp = np.roll(matrix, (1,1), axis=(1,2))
        tmp[:,:,0,...] = add
        tmp[:,0,...] = add
    if shift == "r+u":
        tmp = np.roll(matrix, (-1,1), axis=(1,2))
        tmp[:,:,0,...] = add
        tmp[:,-1,...] = add
    if shift == "l+d":
        tmp = np.roll(matrix, (1,-1), axis=(1,2))
        tmp[:,:,-1,...] = add
        tmp[:,0,...] = add
    if shift == "l+u":
        tmp = np.roll(matrix, (-1,-1), axis=(1,2))
        tmp[:,:,-1,...] = add
        tmp[:,-1,...] = add
    
    return tmp

def compute_adjacency_loss(pred_flow, pred_vss):

    B, h, w, c = pred_flow.shape
    num_a = 8   # Adjacency
    a_list = ["l+u", "u", "r+u", "l", "r", "l+d", "d", "r+d"]

    pred_vss = pred_vss.detach().cpu().numpy()
    pred_flow_detached = pred_flow.detach().cpu().numpy()
    pred_flow = pred_flow.cpu()

    masks = np.zeros((num_a, B, h, w), dtype=np.uint8)
    loss = np.zeros((num_a, B, h, w), dtype=np.uint8)

    for i in range(num_a):
        shifted_pred = shift_matrix(pred_vss, a_list[i], add=-1)
        masks[i][pred_vss==shifted_pred] = 1

    masks = torch.tensor(masks)
    loss = torch.tensor(loss)
    for i,a in enumerate(a_list):

        shifted_pred = torch.tensor(shift_matrix(pred_flow_detached, a, np.array([0,0])))
        loss[i] = masks[i] * torch.sum((pred_flow - shifted_pred)**2, dim=3).sqrt()
    
    return loss.float().mean().cuda()

def compute_seg_loss(pred, flow, lbl):

    lbl  = torch.tensor(lbl)

    pred_3c = np.tile(pred[...,None], (1, 1, 3))
    pred_to_warp = torch.tensor(pred_3c).permute(0, 3, 1, 2).float()

    flow_to_warp = flow.detach().cpu().numpy()

    predict= warp(pred_to_warp , flow_to_warp)
    predict = predict.permute(0, 2, 3, 1).to(torch.uint8)[...,0]

    loss = torch.ones_like(predict)
    loss[predict==lbl] = 0

    return loss.float().mean()

def compute_unsup_loss(img1, img2, flow):

    mse_criterion = nn.MSELoss()

    predict= warp(img1 , flow)

    loss = mse_criterion(predict, img2)
    
    return loss

def finetune(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.DataParallel(RAFTGMA(args), device_ids=args.gpus)

    print(f"Parameter Count: {count_parameters(model)}")

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    data_path = 'data/viper_day_rain'
    path_n = 4
    t_loader = ViperLoader(data_path, split='train', path_num=path_n, bw=args.bwflow)

    train_loader = data.DataLoader(t_loader, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=True, num_workers=4, drop_last=True)
    
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=args.mixed_precision)

    cnt_iter = 0

    for data_blob in tqdm(train_loader):
        cnt_iter += 1

        preds, data_of = data_blob

        image1, image2, flow, valid = [x.cuda() for x in data_of]
        pred_vss1, pred_vss2 = [x for x in preds]

        optimizer.zero_grad()
        flow_pred = model(image1, image2)

        a_loss = compute_adjacency_loss(flow_pred[-1].permute(0,2,3,1), pred_vss1)  # loss 1
        seg_loss = compute_seg_loss(pred_vss1, flow_pred[-1], pred_vss2) # loss 2
        unsup_loss = compute_unsup_loss(image1, image2, flow_pred[-1]) # loss unsup

        loss = args.a*a_loss + args.b*seg_loss + args.c*unsup_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

    PATH = args.output + f'/{args.name}.pth'
    torch.save(model.state_dict(), PATH)
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints', help='output directory to save checkpoints and plots')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--a', default=10, type=int,
                        help='coefficient of adjacency loss')
    parser.add_argument('--b', default=5, type=int,
                        help='backward seg loss')
    parser.add_argument('--c', default=0.01, type=int,
                        help='backward unsup loss')
    parser.add_argument('--bwflow', default=False, action='store_true',
                        help='backward optical flow')


    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    finetune(args)