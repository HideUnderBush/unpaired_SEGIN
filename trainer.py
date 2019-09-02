"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler, Timer
from torch.autograd import Variable
from torchvision import transforms
from ContextualLoss import ContextualLoss 
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import sys
import time

class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.dis_sa = MsImageDis(hyperparameters['input_dim_a']*2, hyperparameters['dis'])  # discriminator for domain a
        self.dis_sb = MsImageDis(hyperparameters['input_dim_b']*2, hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(hyperparameters['display_size'])
        self.s_a = torch.randn(display_size, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(display_size, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        dis_style_params = list(self.dis_sa.parameters()) + list(self.dis_sb.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_style_opt = torch.optim.Adam([p for p in dis_style_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.dis_style_scheduler = get_scheduler(self.dis_style_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_sa.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        self.dis_sb.apply(weights_init('gaussian'))
        if hyperparameters['gen']['CE_method'] == 'vgg':
            self.gen_a.content_init()
            self.gen_b.content_init()
        self.criterion = nn.L1Loss().cuda()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
        self.kld = nn.KLDivLoss()
        self.contextual_loss = ContextualLoss()

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def kl_loss(self, input, target):
        return torch.mean(-self.kld(input, target)) 

    def normalize_feat(self, feat):
        bs, c, H, W = feat.shape
        feat = feat.view(bs, c, -1)
        feat_norm = torch.norm(feat, 2, 1, keepdim=True) + sys.float_info.epsilon
        feat = torch.div(feat, feat_norm)
        #print(max(feat))
        return feat

    def norm_two_domain(self, feat_c, feat_s):
        feat = torch.cat((feat_c, feat_s), 1)
        bs, c, H, W = feat.shape
        feat_norm = torch.norm(feat, 2, 1, keepdim=True)
        feat = torch.div(feat, feat_norm)
        feat_c = feat[:,0:256,:,:].view(bs, 256, -1)
        feat_s = feat[:,256:512,:,:].view(bs, 256, -1)
        return feat_c, feat_s


    def generate_map(self, corr_index, h, w):
        coor = []
        corr_map = []
        for i in range(len(corr_index)):
            x = corr_index[i] // h
            y = corr_index[i] % w 
            coor.append(x)
            coor.append(y)
            corr_map.append(list(np.asarray(coor)))
            coor.clear()
        corr_map_final = np.reshape(np.asarray(corr_map), (h, w, 2))
        return corr_map_final

    def warp_img(self, corr_map, ref_img):
        bs, c, h_img, w_img = ref_img.shape
        h, w, _ = corr_map.shape
        scale = h_img // h
        warped_img = torch.zeros(ref_img.shape)
        for i in range(h):
            for j in range(w):
                nnx = corr_map[i][j][0]
                nny = corr_map[i][j][1]
                warped_img[:, :, i*scale:(i+1)*scale, j*scale:(j+1)*scale] =  ref_img[:, :, nnx*scale:(nnx+1)*scale, nny*scale:(nny+1)*scale]
        return warped_img.cuda()

    def warp_style(self, cur_content, ref_content, ref_style):
        # normalize feature
        cur_content= self.normalize_feat(cur_content)
        ref_content= self.normalize_feat(ref_content)
        #cur_content, ref_content = self.norm_two_domain(cur_content, ref_content)
        cur_content = cur_content.permute(0, 2, 1) 

        # calculate similarity
        f = torch.matmul(cur_content, ref_content) # 1 x (H x W) x (H x W)
        #f_corr = F.softmax(f/0.005, dim=-1) # 1 x (H x W) x (H x W)
        f_corr = F.softmax(f, dim=-1) # 1 x (H x W) x (H x W)

        # get corr index replace softmax 
        bs, HW, WH = f_corr.shape
        corr_index = torch.argmax(f_corr, dim=-1).squeeze(0)


        # collect ref style
        bs, c, H, W = ref_style.shape
        ref_style = ref_style.view(bs, c, -1)
        ref_style = ref_style.permute(0, 2, 1) # 1 x (H x W) x c

        # warp ref style
        warped_style = torch.matmul(f_corr, ref_style) # 1 x (H x W) x c
        warped_style = warped_style.permute(0, 2, 1).contiguous()
        warped_style = warped_style.view(bs, c, H, W)

        return corr_index, warped_style

    def forward(self, x_a, x_b):
        self.eval()
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)

        # warp the ref_style to the content_style
        _, s_ab_warp = warp_style(c_a, c_b, s_b_fake)
        _, s_ba_warp = warp_style(c_b, c_a, s_a_fake)

        x_ba = self.gen_a.decode(c_b, s_ba_warp)
        x_ab = self.gen_b.decode(c_a, s_ab_warp)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, x_adf, x_bdf, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        c_a, s_a_prime = self.gen_a.encode(x_a)
        c_b, s_b_prime = self.gen_b.encode(x_b)

        # add style warp here
        _, s_ab = self.warp_style(c_a, c_b, s_b_prime)
        _, s_ba = self.warp_style(c_b, c_a, s_a_prime)

        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_ba)
        x_ab = self.gen_b.decode(c_a, s_ab)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba) # now the s_a_recon matches the structure of B
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)

        # decode again (if needed)
        # to warp style first
        _, s_aba = self.warp_style(c_b, c_a, s_a_recon)
        _, s_bab = self.warp_style(c_a, c_b, s_b_recon)
        # to reconstruct then
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # prepare paired data for adv generator
        pair_a_ffake = torch.cat((x_ba, x_a), 1)
        pair_b_ffake = torch.cat((x_ab, x_b), 1)

        # reconstruction loss
        self.loss_gen_recon_x_a = self.criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_aba, s_a_prime)
        self.loss_gen_recon_s_b = self.recon_criterion(s_bab, s_b_prime) # default is s_bab, need to test s_b_recon
        #self.loss_gen_recon_s_b = self.recon_criterion(s_ab, s_b_prime)
        #self.loss_gen_recon_s_a = self.recon_criterion(s_ba, s_a_prime)
        self.loss_gen_recon_s_a += self.triplet_loss(s_a_prime, s_aba, s_b_prime)
        self.loss_gen_recon_s_b += self.triplet_loss(s_b_prime, s_bab, s_a_prime)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_kl_ab = self.kl_loss(x_ab, x_b)
        self.loss_gen_kl_ba = self.kl_loss(x_ba, x_a)
        self.loss_gen_cx_a = self.contextual_loss(s_ba, s_a_prime)
        self.loss_gen_cx_b = self.contextual_loss(s_ab, s_b_prime)
        self.loss_gen_cycrecon_x_a = self.criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0

        # GAN loss
        self.loss_gen_adv_xa = self.gen_a.calc_gen_loss(self.dis_a.forward(x_ba))
        self.loss_gen_adv_xb = self.gen_b.calc_gen_loss(self.dis_b.forward(x_ab))
        self.loss_gen_adv_sxa = self.gen_a.calc_gen_loss(self.dis_sa.forward(pair_a_ffake))
        self.loss_gen_adv_sxb = self.gen_b.calc_gen_loss(self.dis_sb.forward(pair_b_ffake))

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss_new(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss_new(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_xa + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_xb + \
                              hyperparameters['gan_wp'] * self.loss_gen_adv_sxa + \
                              hyperparameters['gan_wp'] * self.loss_gen_adv_sxb + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_kl_ab + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_kl_ba + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b
                              #hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_s_a + \
                              #hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_s_b + \
        self.loss_gen_total.backward()
        
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def compute_vgg_loss_new(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_feat = vgg(img_vgg)
        target_feat = vgg(target_vgg)
        return self.recon_criterion(img_feat, target_feat)

    def sample(self, x_a, x_b, x_adf, x_bdf):
        self.eval()
        #s_a1 = Variable(self.s_a)
        #s_b1 = Variable(self.s_b)
        #s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        #s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        h = 64
        w = 64
        for i in range(x_a.size(0)):
            img_a = x_a[i].unsqueeze(0)
            img_b = x_b[i].unsqueeze(0)
            c_a, s_a_fake = self.gen_a.encode(img_a)
            c_b, s_b_fake = self.gen_b.encode(img_b)
            # reconstruction
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            print(x_a_recon[0].shape)
            # warp style
            corr_index_ab, s_ab = self.warp_style(c_a, c_b, s_b_fake)
            corr_index_ba, s_ba = self.warp_style(c_b, c_a, s_a_fake)
            # cross domain construction
            x_ba1.append(self.gen_a.decode(c_b, s_ba))
            ## output warped results x_ba2
            corr_map_ba = self.generate_map(corr_index_ba, h, w)
            x_ba2.append(self.warp_img(corr_map_ba, img_a))

            x_ab1.append(self.gen_b.decode(c_a, s_ab))
            ## output warped results x_ab2
            corr_map_ab = self.generate_map(corr_index_ab, h, w)
            x_ab2.append(self.warp_img(corr_map_ab, img_b))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_adf, x_a_recon, x_ab1, x_ab2, x_b, x_bdf, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, x_adf, x_bdf, hyperparameters):

        self.dis_opt.zero_grad()
        self.dis_style_opt.zero_grad()
        #s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        #s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a = self.gen_a.encode(x_a)
        c_b, s_b = self.gen_b.encode(x_b)

        # warp the style here
        _, s_ab_warp = self.warp_style(c_a, c_b, s_b)
        _, s_ba_warp = self.warp_style(c_b, c_a, s_a)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_ba_warp)
        x_ab = self.gen_b.decode(c_a, s_ab_warp)

        # prepare data for the paired discriminator
        # real fake data -> 0
        if(len(self.dis_sa.pool_) == 0):
            print(len(self.dis_sa.pool_))
            pair_a_rfake = torch.cat((x_b, x_a), 1)
        else:
            pair_a_rfake = torch.cat((self.dis_sa.pool('fetch'), x_a), 1)
        self.dis_sa.pool('push',x_a)

        if(len(self.dis_sb.pool_) == 0):
            print(len(self.dis_sb.pool_))
            pair_b_rfake = torch.cat((x_a, x_b), 1)
        else:
            pair_b_rfake = torch.cat((self.dis_sb.pool('fetch'), x_b), 1)
        self.dis_sb.pool('push',x_b)

        # real real data -> 1
        pair_a_rreal = torch.cat((x_a, x_adf), 1)
        pair_b_rreal = torch.cat((x_b, x_bdf), 1)
        # fake fake data -> 0
        pair_a_ffake = torch.cat((x_ba.detach(), x_a), 1)
        pair_b_ffake = torch.cat((x_ab.detach(), x_b), 1)

        # D loss
        #self.loss_dis_xa = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        #self.loss_dis_xb = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_xa = self.dis_a.calc_dis_loss(x_ba.detach(), self.dis_sa.pool('fetch'))
        self.loss_dis_xb = self.dis_b.calc_dis_loss(x_ab.detach(), self.dis_sb.pool('fetch'))
        #self.loss_dis_sxa = (self.dis_sa.calc_dis_loss(pair_a_rfake, pair_a_rreal) + self.dis_sa.calc_dis_loss(pair_a_ffake, pair_a_rreal)) / 2
        #self.loss_dis_sxb = (self.dis_sb.calc_dis_loss(pair_b_rfake, pair_b_rreal) + self.dis_sb.calc_dis_loss(pair_b_ffake, pair_b_rreal)) / 2
        self.loss_dis_sxa = self.dis_sa.calc_dis_loss(pair_a_ffake, pair_a_rreal)
        self.loss_dis_sxb = self.dis_sb.calc_dis_loss(pair_b_ffake, pair_b_rreal)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_xa + hyperparameters['gan_w'] * self.loss_dis_xb + hyperparameters['gan_wp'] * self.loss_dis_sxa + hyperparameters['gan_wp'] * self.loss_dis_sxb
        self.loss_dis_total.backward()
        self.dis_opt.step()
        self.dis_style_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.dis_style_scheduler is not None:
            self.dis_style_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.dis_style_scheduler = get_scheduler(self.dis_style_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)



