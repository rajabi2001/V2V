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
        n_classes=23,
        pred_vss = False,
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
        self.n_classes = n_classes
        self.size = size
        self.pred_vss = pred_vss
        self.files = {}
        self.to_tensor = transforms.ToTensor()

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

        if self.pred_vss:
            f2_id = f3_id - 1
            f1_id = f2_id - 1

            f2_path = os.path.join(self.images_base, folder, ("%s_%05d.jpg" % (folder, f2_id)))
            if not os.path.isfile(f2_path):
                f2_path = f4_path
            f2_img = imageio.imread(f2_path)
            f2_img = np.array(f2_img, dtype=np.uint8)

            f1_path = os.path.join(self.images_base, folder, ("%s_%05d.jpg" % (folder, f1_id)))
            if not os.path.isfile(f1_path):
                f1_path = f4_path
            f1_img = imageio.imread(f1_path)
            f1_img = np.array(f1_img, dtype=np.uint8)

            [f4_img, f3_img, f2_img, f1_img], _ = self.augmentations([f4_img, f3_img, f2_img, f1_img], f4_img[:,:,0])

            f4_img = self.to_tensor(f4_img).float()
            f3_img = self.to_tensor(f3_img).float()
            f2_img = self.to_tensor(f2_img).float()
            f1_img = self.to_tensor(f1_img).float()

            return [f1_img, f2_img, f3_img, f4_img], f4_path

        img1 = f3_img
        img2 = f4_img
        pred1_path = f3_path.replace("img","pred")
        pred2_path = f4_path.replace("img","pred")

        if self.split == "train":
            pred1 = imageio.imread(pred1_path)
            pred1 = np.array(pred1, dtype=np.uint8)
            pred2 = imageio.imread(pred2_path)
            pred2 = np.array(pred2, dtype=np.uint8)

        # grayscale images
        if len(f4_img.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.split == "train":
            img1, img2, pred1, pred2 = self.augmentor(img1, img2, mask1=pred1, mask2=pred2)
        else:
            img1, img2 = self.augmentor(img1, img2)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        if self.split == "train":
            return [pred1, pred2], [img1, img2]
        else:
            return [img1, img2]


            


