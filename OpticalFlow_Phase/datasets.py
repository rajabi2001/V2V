import os
import torch
import numpy as np
import imageio
from PIL import Image
from torch.utils import data
import random
from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale
from torchvision import transforms
from tqdm import tqdm
import cv2

from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor
from ptsemseg.augmentations import get_composed_augmentations
from core.utils.frame_utils import readFlowNpz

class ViperLoader(data.Dataset):
    
    def __init__(
        self,
        root="data/viper",
        split="train",
        interval=2,
        path_num=4,
        bw = False,
        pred_vss = True,
        size = [256,512]
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.path_num= path_num
        self.interval = interval
        self.root = root
        self.split = split
        self.n_classes = 23
        self.bw = bw
        self.size = size
        self.pred_vss = pred_vss
        self.files = {}

        aug_params_of = {'crop_size': self.size, 'min_scale': -0.8, 'max_scale': 0.1, 'do_flip': False}
        self.augmentor = SparseFlowAugmentor(**aug_params_of)

        aug_params_vss = {'colorjtr': [0.5,0.5,0.5], 'scale': self.size}
        self.augmentations = get_composed_augmentations(aug_params_vss)

        self.images_base = os.path.join(self.root, self.split, "img")

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()

        vid_info = img_path.replace('\\','/').split('/')[-1].split('_')
        folder, frame = vid_info[0], vid_info[1].split('.')[0]
        
        f4_id = int(frame)
        f3_id = f4_id - 1

        f4_path = os.path.join(self.images_base, folder, ("%s_%05d.jpg" % (folder, f4_id)))
        f4_img = imageio.imread(f4_path)
        f4_img = np.array(f4_img, dtype=np.uint8)

        f3_path = os.path.join(self.images_base, folder, ("%s_%05d.jpg" % (folder, f3_id)))
        if not os.path.isfile(f3_path):
            f3_path = f4_path
        f3_img = imageio.imread(f3_path)
        f3_img = np.array(f3_img, dtype=np.uint8)

        if self.bw == True:
            img1 = f4_img
            img2 = f3_img
        else:
            img1 = f3_img
            img2 = f4_img

 
        if self.bw == True:
            pred1_path = f4_path.replace("img","pred")
            pred2_path = f3_path.replace("img","pred")
            flow_path = f4_path.replace("img","flowbw").replace("jpg","npz")
        else:
            pred1_path = f3_path.replace("img","pred")
            pred2_path = f4_path.replace("img","pred")
            flow_path = f3_path.replace("img","flow").replace("jpg","npz")

        if self.split == "train":
            pred1 = imageio.imread(pred1_path)
            pred1 = np.array(pred1, dtype=np.uint8)

            pred2 = imageio.imread(pred2_path)
            pred2 = np.array(pred2, dtype=np.uint8)


        flow, valid = readFlowNpz(flow_path)

        # grayscale images
        if len(f4_img.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        flow = np.array(flow).astype(np.float32)

        if self.split == "train":
            img1, img2, flow, valid, pred1, pred2 = self.augmentor(img1, img2, flow, valid, pred1, pred2)
        else:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        
        if self.split == "train":
            return [pred1, pred2], [img1, img2, flow, valid.float()]
        else:
            return [img1, img2, flow, valid.float()]


            


