import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import cv2
import random
from models import vgg
from ofe_core.network import RAFTGMA
import torch.nn as nn


class FlowCycleGANModel(BaseModel):
    def name(self):
        return 'FlowCycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt

        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A1 = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A1_flow = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A2_flow = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.input_nc, size, size)

        self.loss_D = 0
        self.loss_G_A = 0
        self.loss_cycle_A = 0
        self.loss_idt_A = 0
        self.loss_idt_B = 0
        self.loss_perc_A = 0
        self.loss_flow = 0

        self.idt_A = None

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netF = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.vgg = vgg.vgg.cuda(self.gpu_ids[0])
            vgg_pretrained_path = 'saved_models/vgg_normalised.pth'
            self.vgg.load_state_dict(torch.load(vgg_pretrained_path))
            self.vgg.eval()
            self.vgg.requires_grad = False
            self.instance_norm_layer = torch.nn.InstanceNorm2d(
                512, affine=False)
            
        if self.isTrain:
            self.netOFE = nn.DataParallel(RAFTGMA(opt), device_ids=opt.gpu_ids)
            if opt.ofe_ckpt is not None:
                self.netOFE.load_state_dict(torch.load(opt.ofe_ckpt), strict=False)
            self.netOFE.cuda()
            self.netOFE.eval()

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netF, 'F', which_epoch)
                self.load_network(self.netD, 'D', which_epoch)
            print(f"Model has been loaded from  epoch {which_epoch} chekpoint")

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionPerc = torch.nn.MSELoss()
            self.criterionflow = torch.nn.MSELoss()
            self.sigmoid = torch.nn.Sigmoid()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(
            ), self.netF.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # print('---------- Networks initialized -------------')
        # networks.print_network(self.netG)
        # networks.print_network(self.netF)
        # if self.isTrain:
        #     networks.print_network(self.netD)
        # print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A2'] # key frame
        input_A1 = input['A1']
        input_A1_flow = input['A1_flow']
        input_A2_flow = input['A2_flow']
        input_B = input['B1']

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A1.resize_(input_A.size()).copy_(input_A1)
        self.input_A1_flow.resize_(input_A.size()).copy_(input_A1_flow)
        self.input_A2_flow.resize_(input_A.size()).copy_(input_A2_flow)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        # self.image_paths = input['A2_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_A1 = Variable(self.input_A1)
        self.real_B = Variable(self.input_B)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(real_A).data

    def translate(self):
        real_A = Variable(self.input_A)
        fake_B = self.netG(real_A).data
        return fake_B

    def get_vgg_features(self, image):
        return self.instance_norm_layer(self.vgg(image)).detach().cpu()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B)
        self.loss_D = loss_D.item()

    def backward_G(self):

        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_content_A = self.opt.lambda_content_A
        lambda_flow = self.opt.lambda_flow

        # GAN loss D_A(G_A(A))
        fake_B = self.netG(self.real_A)
        fake_B1 = self.netG(self.real_A1)
        pred_fake = self.netD(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netF(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG(self.real_B)
            loss_idt_A = self.criterionIdt(
                idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netF(self.real_A)
            loss_idt_B = self.criterionIdt(
                idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.item()
            self.loss_idt_B = loss_idt_B.item()
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # perceptual loss
        if lambda_content_A > 0:
            in_feat_fake_B = self.instance_norm_layer(self.vgg(fake_B))
            in_feat_real_A = self.instance_norm_layer(self.vgg(self.real_A))

            loss_perc_A = lambda_content_A * \
                self.criterionPerc(in_feat_fake_B, in_feat_real_A)
            self.loss_perc_A = loss_perc_A.item()
        else:
            loss_perc_A = torch.tensor(0.)
            self.loss_perc_A = 0

        # flow loss
        if lambda_flow > 0:
            _, flow_pr = self.netOFE(self.input_A1_flow, self.input_A2_flow, iters=6, test_mode=True)
            flow = flow_pr[0]

            flow_to_warp = flow.detach().cpu().numpy()
            warped_fake_B1 = util.warp(fake_B1.cpu(), flow_to_warp)

            loss_flow = lambda_flow * self.criterionflow(warped_fake_B1, fake_B.cpu())
            self.loss_flow = loss_flow.item()
            self.predicted_B = warped_fake_B1
        else:
            loss_flow = torch.tensor(0.)
            self.loss_flow = 0

        # combined loss
        loss_G = loss_G_A + loss_cycle_A + loss_idt_A + loss_idt_B + loss_perc_A + loss_flow
        loss_G.backward()

        self.fake_B = fake_B.data
        self.rec_A = rec_A.data

        self.loss_G_A = loss_G_A.item()
        self.loss_cycle_A = loss_cycle_A.item()
        self.loss_flow = loss_flow.item()

    def optimize_parameters(self, counter=0):
        # forward
        self.forward()
        if counter % self.opt.artifcial_batch == 0:
            # G_A and G_B
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            # D_A
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        else:
            self.backward_G()
            self.backward_D()

    def get_current_errors(self):
        ret_errors = OrderedDict(
            [('D', self.loss_D), ('G', self.loss_G_A), ('Cyc_A', self.loss_cycle_A)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        if self.opt.lambda_content_A > 0.0:
            ret_errors['perc_A'] = self.loss_perc_A
        if self.opt.lambda_flow > 0.0:
            ret_errors['flow'] = self.loss_flow
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

        if not self.idt_A is None and self.isTrain:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals
    
    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netF, 'F', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
