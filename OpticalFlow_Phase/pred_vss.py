import os
import torch
import numpy as np
import oyaml as yaml
from torch.utils import data
from PIL import Image
import imageio
from ptsemseg.models import get_model
from tqdm import tqdm
from datasets import ViperLoader
import argparse

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="data/")
    parser.add_argument('--split', type=str, default="train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = args.root
    split = args.split
    s_path = f"{root}/{split}/img/"
    d_path = f"{root}/{split}/pred/"
    for i in os.listdir(s_path):
        os.makedirs(d_path+i, exist_ok=True)
  
    data_path = root
    path_n = 4

    t_loader = ViperLoader(data_path, split='train', path_num=path_n, pred_vss=True)

    train_loader = data.DataLoader(t_loader, batch_size=4,
                                   pin_memory=True, shuffle=True, num_workers=4, drop_last=True)
    

    model_vss_dict = {'arch': 'td4_psp', 'backbone': 'resnet18', 'syncBN': False, 'path_num': 4}
    model_vss = get_model(model_vss_dict, nclass=23).to(device)
    vss_ckp_path = "./checkpoints/viper_td4-psp18_base.pth"
    state = torch.load(vss_ckp_path)
    model_vss.load_state_dict(state, strict=False)
    print("Initialized vss networks with pretrained '{}'".format(vss_ckp_path))
    model_vss.eval()
    model_vss.to(device)


    cnt_iter = 0
    for data_blob in tqdm(train_loader):
        cnt_iter += 1

        imgs, imgs_path = data_blob
        imgs_vss = [ele.to(device) for ele in imgs]

        outputs = model_vss(imgs_vss,pos_id=cnt_iter%path_n)
        pred_vss = np.uint8(outputs.data.max(1)[1].cpu().numpy())

        for i in range(len(imgs_path)):
            img_path = imgs_path[i]
            img_path = img_path.replace("img","pred")
            imageio.imwrite(img_path, pred_vss[i])

        


   