import argparse
import gc
import glob
import os
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
import math
import numpy as np
import torch.nn as nn
from scipy.linalg import sqrtm
from util.util import warp, AverageMeter, intersectionAndUnion
# from fcn import FCN8sAtOnce

ROOT_ADDRESS = '.'
TEST_NAME = 'day-to-rain'

class Metrics():
    def __init__(self):
        self.inceptionv3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.inceptionv3.fc = nn.Flatten()
        self.inceptionv3.eval()
        self.inceptionv3.to('cuda')

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_images(self, images):
        new_images = []
        for image in images:
            new_images.append(self.preprocess(image))

        return torch.stack(new_images).to('cuda')
    
    def cal_warping_error(self, source_frames, translated_frames, flows, alpha=0.075):
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
            gc.collect()
            
            warping_error = source_dis * translated_dis
            error += warping_error

        return error

    def cal_fid(self, images1, images2):
        images1 = self.preprocess_images(images1)
        images2 = self.preprocess_images(images2)

        with torch.no_grad():
            feature1 = self.inceptionv3(images1).cpu().numpy()
            feature2 = self.inceptionv3(images2).cpu().numpy()

        mu1, sigma1 = feature1.mean(axis=0), np.cov(feature1, rowvar=False)
        mu2, sigma2 = feature2.mean(axis=0), np.cov(feature2, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2)**2.0)
        
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
          covmean = covmean.real
        
        del images1, images2, feature1, feature2
        gc.collect()

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    # def cal_cls_l2v(self, epoch):
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--exp_name', action="store", type=str, default='{}/results/{}/test_{}/images/*fake_B.png'.format(ROOT_ADDRESS, TEST_NAME, str(epoch)))
    #     parser.add_argument('--target_path', action="store", type=str, default='{}/data/{}/val/A'.format(ROOT_ADDRESS, TEST_NAME))
    #     parser.add_argument('--pred_cache_dir', action="store", type=str, default="pred_lbl_ind_map")
    #     parser.add_argument('--lbl_cache_dir', action="store", type=str, default="gt_lbl_ind_map_l2v")
    #     parser.add_argument('--eval_h', action="store", type=int, default="256")
    #     parser.add_argument('--eval_w', action="store", type=int, default="256")
    #     parser.add_argument('--model_path', action="store", type=str, default='{}/saved_models/fcn_model.pt'.format(ROOT_ADDRESS))
    #     parser.add_argument('--n_class', action="store", type=int, default="32")
    #     parser.add_argument('--mean', action="store", type=bool, default="False")
    #     parser.add_argument('--verbose', action="store", type=bool, default="False")
    #     parser.add_argument('--output_name', action="store", type=str, default="")

    #     args = parser.parse_args("")

    #     device = "cuda" if torch.cuda.is_available() else "cpu"

    #     model = FCN8sAtOnce(n_class=args.n_class).to(device)
    #     model.load_state_dict(torch.load(args.model_path))
    #     model.eval()

    #     fs = glob.glob(args.exp_name)

    #     print("L2V experiment, Looking files from:", args.exp_name, ", with model:", args.model_path)

    #     rgb2id = {
    #         (0, 0, 0): (0, "unlabeled"),
    #         (111, 74, 0): (1, "ambiguous"),	
    #         (70, 130, 180): (2, "sky"), 
    #         (128, 64, 128): (3, "road"), 
    #         (244, 35, 232): (4, "sidewalk"), 
    #         (230, 150, 140): (5, "railtrack"),
    #         (152, 251, 152): (6, "terrain"), 
    #         (87, 182, 35): (7, "tree"), 
    #         (35, 142, 35): (8, "vegetation"), 
    #         (70, 70, 70): (9, "building"), 
    #         (153, 153, 153): (10, "infrastructure"), 
    #         (190, 153, 153): (11, "fence"), 
    #         (150, 20, 20): (12, "billboard"), 
    #         (250, 170, 30): (13, "traffic light"), 
    #         (220, 220, 0): (14, "traffic sign"), 
    #         (180, 180, 100): (15, "mobilebarrier"), 
    #         (173, 153, 153): (16, "firehydrant"),
    #         (168, 153, 153): (17, "chair"),
    #         (81, 0, 21): (18, "trash"),
    #         (81, 0, 81): (19, "trashcan"),
    #         (220, 20, 60): (20, "person"),
    #         (255, 0, 0): (21, "animal"),
    #         (119, 11, 32): (22, "bicycle"),
    #         (0, 0, 230): (23, "motorcycle"),
    #         (0, 0, 142): (24, "car"),
    #         (0, 80, 100): (25, "van"),
    #         (0, 60, 100): (26, "bus"),
    #         (0, 0, 70): (27, "truck"),
    #         (0, 0, 90): (28, "trailer"),
    #         (0, 80, 100): (29, "train"),
    #         (0, 100, 100): (30, "plane"),
    #         (50, 0, 90): (31, "boat"),
    #     }

    #     colors = torch.from_numpy(np.array(list(rgb2id.keys()))).to(device).float()
    #     ids = [rgb2id[i][0] for i in rgb2id.keys()]
    #     class_names = [rgb2id[i][1] for i in rgb2id.keys()]

    #     if not os.path.exists(args.pred_cache_dir):
    #         os.makedirs(args.pred_cache_dir)
    #     if not os.path.exists(args.lbl_cache_dir):
    #         os.makedirs(args.lbl_cache_dir)

    #     def cal_acc(data_list, classes, names):
    #         print(classes, names)
    #         intersection_meter = AverageMeter()
    #         union_meter = AverageMeter()
    #         target_meter = AverageMeter()

    #         if len(data_list) == 0:
    #             print("Empty list.")
    #             return

    #         for i, (image_path, target_path) in enumerate(data_list):
    #             pred = np.array(Image.open(image_path))
    #             target = np.array(Image.open(target_path))
    #             eval_size = (args.eval_w, args.eval_h)
    #             if pred.shape != eval_size:
    #                 pred = cv2.resize(pred, eval_size, cv2.INTER_NEAREST)
    #             if target.shape != eval_size:
    #                 target = cv2.resize(target, eval_size, cv2.INTER_NEAREST)
    #             intersection, union, target = intersectionAndUnion(pred, target, classes)
    #             intersection_meter.update(intersection)
    #             union_meter.update(union)
    #             target_meter.update(target)

    #         iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    #         accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    #         mIoU = np.mean(iou_class)
    #         mAcc = np.mean(accuracy_class)
    #         allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    #         print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    #         for i in range(classes):
    #             print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

    #     data_list = []

    #     img_mean = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float32)

    #     for f in (fs):

    #         im = Image.open(f).convert('RGB')
    #         im = np.array(im.resize((args.eval_w, args.eval_h), Image.LANCZOS), dtype=np.uint8)
    #         im = im[:, :, ::-1]  # RGB -> BGR
    #         im = im.astype(np.float64)
    #         if args.mean:
    #             im -= img_mean
    #         im = im.transpose(2, 0, 1)
    #         im = torch.from_numpy(im).float().cuda().unsqueeze(0).to(device)
    #         with torch.no_grad():
    #             lbl = model(im)[0]

    #         lbl = torch.argmax(lbl[0], 0)
    #         # save the map
    #         im = Image.fromarray(np.uint8(lbl.cpu().numpy()))
    #         save_dir = os.path.join(args.pred_cache_dir, os.path.basename(f))
    #         im.save(save_dir)

    #         vid_ind = os.path.basename(f).split("_")[0]
    #         img_ind = os.path.basename(f).split("_")[1]
    #         target_path = os.path.join(args.target_path, vid_ind + "_" + img_ind + ".png")

    #         data_list.append((save_dir, target_path))

    #     print("All:")
    #     cal_acc(data_list, len(class_names), class_names)


    # def cal_cls_v2l(self, epoch):
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--exp_name', action="store", type=str, default='{}/results/{}/test_{}/images/*fake_B.png'.format(ROOT_ADDRESS, TEST_NAME, str(epoch)))
    #     parser.add_argument('--target_path', action="store", type=str, default='{}/data/{}/val/A_cls'.format(ROOT_ADDRESS, TEST_NAME))
    #     parser.add_argument('--map_cache_dir', action="store", type=str, default="seg_lbl_ind_map")
    #     parser.add_argument('--lbl_cache_dir', action="store", type=str, default="gt_lbl_ind_map")
    #     parser.add_argument('--eval_h', action="store", type=int, default="256")
    #     parser.add_argument('--eval_w', action="store", type=int, default="256")
    #     parser.add_argument('--details', action="store", type=bool, default="False")

    #     args = parser.parse_args("")

    #     device = "cuda" if torch.cuda.is_available() else "cpu"

    #     fs = glob.glob(args.exp_name)
    #     print("V2L experiment, Looking files from:", args.exp_name)
    #     rgb2id = {
    #         (0, 0, 0): (0, "unlabeled"),
    #         (111, 74, 0): (1, "ambiguous"),	
    #         (70, 130, 180): (2, "sky"), 
    #         (128, 64, 128): (3, "road"), 
    #         (244, 35, 232): (4, "sidewalk"), 
    #         (230, 150, 140): (5, "railtrack"),
    #         (152, 251, 152): (6, "terrain"), 
    #         (87, 182, 35): (7, "tree"), 
    #         (35, 142, 35): (8, "vegetation"), 
    #         (70, 70, 70): (9, "building"), 
    #         (153, 153, 153): (10, "infrastructure"), 
    #         (190, 153, 153): (11, "fence"), 
    #         (150, 20, 20): (12, "billboard"), 
    #         (250, 170, 30): (13, "traffic light"), 
    #         (220, 220, 0): (14, "traffic sign"), 
    #         (180, 180, 100): (15, "mobilebarrier"), 
    #         (173, 153, 153): (16, "firehydrant"),
    #         (168, 153, 153): (17, "chair"),
    #         (81, 0, 21): (18, "trash"),
    #         (81, 0, 81): (19, "trashcan"),
    #         (220, 20, 60): (20, "person"),
    #         (255, 0, 0): (21, "animal"),
    #         (119, 11, 32): (22, "bicycle"),
    #         (0, 0, 230): (23, "motorcycle"),
    #         (0, 0, 142): (24, "car"),
    #         (0, 80, 100): (25, "van"),
    #         (0, 60, 100): (26, "bus"),
    #         (0, 0, 70): (27, "truck"),
    #         (0, 0, 90): (28, "trailer"),
    #         (0, 80, 100): (29, "train"),
    #         (0, 100, 100): (30, "plane"),
    #         (50, 0, 90): (31, "boat"),
    #     }

    #     colors = torch.from_numpy(np.array(list(rgb2id.keys()))).to(device).float()
    #     ids = [rgb2id[i][0] for i in rgb2id.keys()]
    #     class_names = [rgb2id[i][1] for i in rgb2id.keys()]

    #     if not os.path.exists(args.map_cache_dir):
    #         os.makedirs(args.map_cache_dir)
    #     if not os.path.exists(args.lbl_cache_dir):
    #         os.makedirs(args.lbl_cache_dir)

    #     def cal_acc(data_list, classes, names):
    #         intersection_meter = AverageMeter()
    #         union_meter = AverageMeter()
    #         target_meter = AverageMeter()

    #         if len(data_list) == 0:
    #             print("Empty list.")
    #             return

    #         for i, (image_path, target_path) in enumerate(data_list):
    #             pred = np.array(Image.open(image_path))
    #             target = np.array(Image.open(target_path))
    #             eval_size = (args.eval_w, args.eval_h)
    #             if pred.shape != eval_size:
    #                 pred = cv2.resize(pred, eval_size, cv2.INTER_NEAREST)
    #             if target.shape != eval_size:
    #                 target = cv2.resize(target, eval_size, cv2.INTER_NEAREST)

    #             intersection, union, target = intersectionAndUnion(pred, target, classes)
    #             intersection_meter.update(intersection)
    #             union_meter.update(union)
    #             target_meter.update(target)
    #             accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

    #         iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    #         accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    #         mIoU = np.mean(iou_class)
    #         mAcc = np.mean(accuracy_class)
    #         allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    #         print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))

    #         for i in range(classes):
    #             print('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

    #     data_list = []

    #     for f in (fs):
    #         img = torch.from_numpy(np.array(Image.open(f).convert('RGB'))).to(device) # (h, w, 3)
    #         img = img.unsqueeze(2).float()
    #         diff = torch.norm(img - colors, dim=3)
    #         ind_map = torch.argmin(diff, dim=2)
    #         lbl = 255 * torch.ones_like(ind_map)
    #         for i in range(len(colors)):
    #             lbl[ind_map == i] = ids[i]
    #         # save the map
    #         save_dir = os.path.join(args.map_cache_dir, os.path.basename(f))
    #         im = Image.fromarray(np.uint8(lbl.cpu().numpy()))
    #         im.save(save_dir)

    #         vid_ind = os.path.basename(f).split("_")[0]
    #         img_ind = os.path.basename(f).split("_")[1]
    #         target_path = os.path.join(args.target_path, vid_ind + "_" + img_ind + ".png")
    #         gt_cache_dir = os.path.join(args.lbl_cache_dir, vid_ind + "_" + img_ind + ".png")
    #         if not os.path.exists(gt_cache_dir):

    #             gt_rgb = torch.from_numpy(np.array(Image.open(target_path).convert('RGB'))).to(device) # (h, w, 3)
    #             gt_rgb = gt_rgb.unsqueeze(2).float()
    #             tgt_diff = torch.norm(gt_rgb - colors, dim=3)
    #             tgt_ind_map = torch.argmin(tgt_diff, dim=2)
    #             gt = 255 * torch.ones_like(tgt_ind_map)
    #             for i in range(len(colors)):
    #                 gt[tgt_ind_map == i] = ids[i]
    #             gt = Image.fromarray(np.uint8(gt.cpu().numpy()))
    #             gt.save(gt_cache_dir)

    #         data_list.append((save_dir, gt_cache_dir))

    #     print("All:")
    #     cal_acc(data_list, len(class_names), class_names)