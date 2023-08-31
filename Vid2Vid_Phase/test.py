import glob
import time
import os
import cv2
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.metrics import Metrics
from util import html
import tqdm
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import gc
from util.metrics import ROOT_ADDRESS, TEST_NAME
from util.util import read_flow

def test(output_original=False, count=-1):

    opt = TestOptions().parse()
    epoch = opt.which_epoch
    tag = opt.name

    output_frame_size = (512, 256)
    original_frame_size = (512, 256)

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    if count != -1:
        opt.how_many = count

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)

    # create website
    if opt.split != "":
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.phase, opt.split, opt.which_epoch))
    else:
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        if opt.model != 'flow_cyclegan' and i < 2: continue
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()

    frameSize = output_frame_size
    out = cv2.VideoWriter('{}/video/translated_{}.mp4'.format(ROOT_ADDRESS, tag), cv2.VideoWriter_fourcc(*'DIVX'), 15, frameSize)
    for filename in sorted(glob.glob('{}/results/{}/test_{}/images/*fake_B.png'.format(ROOT_ADDRESS, tag, str(epoch)))):
        image = cv2.imread(filename)
        out.write(image)
    out.release()

    if len(glob.glob('{}/results/{}/test_{}/images/*fake_B_enhanced.png'.format(ROOT_ADDRESS, tag, str(epoch)))) > 0:
        frameSize = output_frame_size
        out = cv2.VideoWriter('{}/video/translated_{}_{}_enhanced.mp4'.format(ROOT_ADDRESS, str(epoch), tag), cv2.VideoWriter_fourcc(*'DIVX'), 15, frameSize)
        for filename in sorted(glob.glob('{}/results/{}/test_{}/images/*fake_B_enhanced.png'.format(ROOT_ADDRESS, tag, str(epoch)))):
            image = cv2.imread(filename)
            out.write(image)
        out.release()
    
    if output_original:
        frameSize = original_frame_size
        out = cv2.VideoWriter('{}/video/original_video.mp4'.format(ROOT_ADDRESS), cv2.VideoWriter_fourcc(*'DIVX'), 15, frameSize)
        for filename in sorted(glob.glob('{}/results/{}/test_{}/images/*real_A.png'.format(ROOT_ADDRESS, tag, str(epoch)))):
            image = cv2.imread(filename)
            out.write(image)
        out.release()

    
def evaluate_warping_error(seq_size):
    metrics = Metrics()
    opt = TestOptions().parse()
    epoch = opt.which_epoch
    tag = opt.name

    transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)

    real_paths = sorted(glob.glob('{}/results/{}/test_{}/images/*real_A.png'.format(ROOT_ADDRESS, tag, str(epoch))))[:seq_size]
    fake_paths = sorted(glob.glob('{}/results/{}/test_{}/images/*fake_B.png'.format(ROOT_ADDRESS, tag, str(epoch))))[:seq_size]
    size = len(real_paths)

    dir_flow = os.path.join(opt.dataroot, "val/flow")
    flow_paths = sorted(glob.glob('{}/*.npz'.format(dir_flow)))

    scale_y = opt.loadSizeH / 1080
    scale_x = opt.loadSizeW / 1920
    flows = read_flow(flow_paths[:seq_size+2], scale_y, scale_x)[2:2+size]

    real_images = list(map(lambda path: Image.open(path).convert('RGB'), real_paths))
    fake_images = list(map(lambda path: Image.open(path).convert('RGB'), fake_paths))
    
    real_images = list(map(lambda image: transform(image), real_images))
    fake_images = list(map(lambda image: transform(image), fake_images))
    
    real_images = list(map(lambda image: torch.unsqueeze(image, 0), real_images))
    fake_images = list(map(lambda image: torch.unsqueeze(image, 0), fake_images))
    
    error = metrics.cal_warping_error(real_images, fake_images, flows, alpha=0.15)

    del flows
    del real_images
    del fake_images
    gc.collect()

    return error

def evaluate_mfid(target_domain, gt_size, fake_size):
    metrics = Metrics()
    opt = TestOptions().parse()
    epoch = opt.which_epoch
    tag = opt.name

    real_paths = sorted(glob.glob('{}/groundtruths/{}/*.jpg'.format(ROOT_ADDRESS, target_domain)))[:gt_size]
    fake_paths = sorted(glob.glob('{}/results/{}/test_{}/images/*fake_B.png'.format(ROOT_ADDRESS, tag, str(epoch))))[:fake_size]

    real_images = list(map(lambda path: cv2.imread(path), real_paths))
    fake_images = list(map(lambda path: cv2.imread(path), fake_paths))

    result = metrics.cal_fid(fake_images, real_images)

    del real_images
    del fake_images
    gc.collect()

    return result

if __name__ == '__main__':

    test(output_original=True)
    # print('Warping error:', evaluate_warping_error(500))
    # print('mFiD score:', evaluate_mfid('rain', 50, 300))

    # metrics = Metrics()
    # metrics.cal_cls_l2v(checkpoint_name)
    # metrics.cal_cls_v2l(checkpoint_name)