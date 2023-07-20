import os.path
import numpy as np
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torch
import cv2

class UnalignedFlowDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # if opt.split == "":
        if opt.phase == "train":
            self.dir_A = os.path.join(opt.dataroot, "train/A")
            self.dir_B = os.path.join(opt.dataroot, "train/B")
        if opt.phase == "test":
            self.dir_A = os.path.join(opt.dataroot, "val/A")
            self.dir_B = os.path.join(opt.dataroot, "val/B")

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)


        # self.transform = get_transform(opt)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        
        index_A2 = index % self.A_size
        index_A1 = index_A2 - 1
        if index_A1 < 0:
            index_A1 = index_A2

        A2_path = self.A_paths[index_A2]
        A1_path = self.A_paths[index_A1]

        if self.opt.phase == "train":
            A2_video = A2_path.split(".")[0].split("/")[3].split("_")[0]
            A1_video = A1_path.split(".")[0].split("/")[3].split("_")[0]
            if A2_video != A1_video:
                A1_path = A2_path

        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A2_img = Image.open(A2_path).convert('RGB')
        A1_img = Image.open(A1_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # resize A*
        if self.opt.resize_mode == "scale_shortest":
            w, h = A2_img.size
            if w >= h: 
                scale = self.opt.loadSize / h
                new_w = int(w * scale)
                new_h = self.opt.loadSize
            else:
                scale = self.opt.loadSize / w
                new_w = self.opt.loadSize
                new_h = int(h * scale)
                
            A2_img = A2_img.resize((new_w, new_h), Image.BICUBIC)
            A1_img = A1_img.resize((new_w, new_h), Image.BICUBIC)
        elif self.opt.resize_mode == "square":
            A2_img = A2_img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            A1_img = A1_img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        elif self.opt.resize_mode == "rectangle":
            A2_img = A2_img.resize((self.opt.loadSizeW, self.opt.loadSizeH), Image.BICUBIC)
            A1_img = A1_img.resize((self.opt.loadSizeW, self.opt.loadSizeH), Image.BICUBIC)
        elif self.opt.resize_mode == "none":
            pass
        else:
            raise ValueError("Invalid resize mode!")
        
        A2_flow = A2_img.copy()
        A1_flow = A1_img.copy()
        A2_flow = np.array(A2_flow).astype(np.uint8)[..., :3]
        A1_flow = np.array(A1_flow).astype(np.uint8)[..., :3]
        A2_flow = torch.from_numpy(A2_flow).permute(2, 0, 1).float()
        A1_flow = torch.from_numpy(A1_flow).permute(2, 0, 1).float()

        A2_img = self.transform(A2_img)
        A1_img = self.transform(A1_img)

        # crop A*
        w = A2_img.size(2)
        h = A2_img.size(1)
        if self.opt.crop_mode == "square":
            fineSizeW, fineSizeH = self.opt.fineSize, self.opt.fineSize
        elif self.opt.crop_mode == "rectangle":
            fineSizeW, fineSizeH = self.opt.fineSizeW, self.opt.fineSizeH
        elif self.opt.crop_mode == "none":
            fineSizeW, fineSizeH = w, h
        else:
            raise ValueError("Invalid crop mode!")

        w_offset = random.randint(0, max(0, w - fineSizeW - 1))
        h_offset = random.randint(0, max(0, h - fineSizeH - 1))

        A2_img = A2_img[:, h_offset:h_offset + fineSizeH, w_offset:w_offset + fineSizeW]
        A1_img = A1_img[:, h_offset:h_offset + fineSizeH, w_offset:w_offset + fineSizeW]
        A2_flow = A2_flow[:, h_offset:h_offset + fineSizeH, w_offset:w_offset + fineSizeW]
        A1_flow = A1_flow[:, h_offset:h_offset + fineSizeH, w_offset:w_offset + fineSizeW]
        
        # resize B
        if self.opt.resize_mode == "scale_shortest":
            w, h = B_img.size
            if w >= h: 
                scale = self.opt.loadSize / h
                new_w = int(w * scale)
                new_h = self.opt.loadSize
            else:
                scale = self.opt.loadSize / w
                new_w = self.opt.loadSize
                new_h = int(h * scale)
                
            B_img = B_img.resize((new_w, new_h), Image.BICUBIC)
        elif self.opt.resize_mode == "square":
            B_img = B_img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        elif self.opt.resize_mode == "rectangle":
            B_img = B_img.resize((self.opt.loadSizeW, self.opt.loadSizeH), Image.BICUBIC)
        elif self.opt.resize_mode == "none":
            pass
        else:
            raise ValueError("Invalid resize mode!")

        B_img = self.transform(B_img)

        # crop B
        w = B_img.size(2)
        h = B_img.size(1)
        if self.opt.crop_mode == "square":
            fineSizeW, fineSizeH = self.opt.fineSize, self.opt.fineSize
        elif self.opt.crop_mode == "rectangle":
            fineSizeW, fineSizeH = self.opt.fineSizeW, self.opt.fineSizeH
        elif self.opt.crop_mode == "none":
            fineSizeW, fineSizeH = w, h
        else:
            raise ValueError("Invalid crop mode!")
        w_offset = random.randint(0, max(0, w - fineSizeW - 1))
        h_offset = random.randint(0, max(0, h - fineSizeH - 1))

        B_img = B_img[:, h_offset:h_offset + fineSizeH, w_offset:w_offset + fineSizeW]

        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        return {'A1': A1_img, 'A2': A2_img, 'B1': B_img, 'A1_flow': A1_flow, 'A2_flow': A2_flow, 'A1_paths': A1_path, 'A2_paths': A2_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedFlowDataset'
