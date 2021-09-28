# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:27:08 2021

@author: hanqi
"""


import sys
import os
import glob
import random
import time

import numpy as np
import pandas as pd
from skimage.io import imread

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import argparse
import math

from torchvision import transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset,DataLoader
from torchvision import datasets

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch

from PIL import ImageDraw
from torchvision.transforms.functional import adjust_brightness



from PIL import Image
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve,average_precision_score
from sklearn import preprocessing 

class MyDataLoader(torch.utils.data.Dataset):
    def __init__(self,filepath=None,transform=None,mask_transform=None,gray=True,test=False):
        self.filepath = filepath
        self.test = test
        self.ano_level = 1
        #self.test_transform = get_test_transforms(160)
        #if self.test:
        #    self.mask_file = []
        #    m_file = os.listdir(self.filepath+'mask_x')
        #    for m in m_file:
        #        cur_path = os.path.join(self.filepath+'mask_x',m)
        #        for c in os.listdir(cur_path):
        #            self.mask_file.append(os.path.join(cur_path,c))
        #    self.test_file = []
        #    t_file = os.listdir(self.filepath+'x')
        #    for t in t_file:
        #        cur_path = os.path.join(self.filepath+'x',t)
        #        for c in os.listdir(cur_path):
        #            self.test_file.append(os.path.join(cur_path,c))
        if self.test:
            self.mask_file = []
            m_file = os.listdir(self.filepath+'mask')
            for m in m_file:
                self.mask_file.append(os.path.join(self.filepath+'mask',m))
            #print(self.mask_file)
            self.test_file = []
            t_file = os.listdir(self.filepath+'brats')
            for t in t_file:
                self.test_file.append(os.path.join(self.filepath+'brats',t))
        else:
            self.file = os.listdir(self.filepath)
            
    
        self.gray = gray
        self.transform = transform
        self.mask_transform = mask_transform
        #self.img_size=128

    def random_mask(self, size):
        """Generate mask tensor from bbox.
        Returns:
        tf.Tensor: output with shape [1, H, W, 1]
        """
        min_num_vertex = 8
        max_num_vertex = 32
        mean_angle = 1*math.pi / 2
        angle_range = 1*math.pi / 4
        min_width = 4
        max_width = 16
        H=256
        W=256
        img_size = 256
        mu=img_size//2
        sigma = img_size//16
        average_radius = math.sqrt(H*H+W*W) / 16
        mask = Image.new('L', (W, H), 0)
        for _ in range(np.random.randint(1, 2)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            #for i in range(num_vertex):
                #    if i % 2 == 0:
                    #        angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                    #    else:
                        #        angles.append(np.random.uniform(angle_min, angle_max))
                        
            angle_min = 0
            angle_max = math.pi/24
            init_angle = np.random.uniform(0,2*math.pi)
            angles.append(init_angle)
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(angles[-1] + math.pi + angle_max)#np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(angles[-1] - math.pi - angle_max)#np.random.uniform(angle_min, angle_max))
            h, w = mask.size
            vertex.append((np.random.normal(mu, sigma, 1).astype(int), np.random.normal(mu, sigma, 1).astype(int)))
            for i in range(num_vertex):
                draw_length = average_radius*np.random.uniform(1,3)
                r = draw_length
                #r = np.clip(
                #    np.random.normal(loc=draw_length, scale=draw_length//2),
                #    0, 2*draw_length)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=1)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        #mask = mask.resize((size, size),Image.ANTIALIAS)
        mask = np.asarray(mask, np.float32)
        mask = np.expand_dims(mask,2)
        mask = np.repeat(mask,3,axis=2)
        #mask = np.reshape(mask,(size,size))
        return mask
    
    def circle_mask(self,img):
        size = img.shape[0]
        img = np.zeros((size,size,3))
        color = (1,1,1)
        min_size = 20
        np.random.normal()
        #first_point=(
        #    np.clip(int((size+np.random.normal(0,0.25)*size)//2),min_size,size-min_size),
        #    np.clip(int((size+np.random.normal(0,0.25)*size)//2),min_size,size-min_size)
        #)
        first_point = (random.randint(min_size,size-min_size),
                       random.randint(min_size,size-min_size))
        region_size = random.randint(min_size//2,min_size*2)
        circle = cv2.circle(img,first_point,random.randint(min(region_size,
                                                                 first_point[0],
                                                                 first_point[1],
                                                                 size - first_point[0],
                                                                 size - first_point[1])
        ,min(region_size,
             first_point[0],
             first_point[1],
             size - first_point[0],
             size - first_point[1])),color,-1)
        return circle

    def normalization(self,img):
        min_val = np.min(img)
        max_val = np.max(img)
        output = (img-min_val)/(max_val-min_val)
        return output

    def const_img(self, img,mask):

        #random_scale = random.random()
        #random_switch = random.random()
        random_scale = round(random.random(),4)+0.1
        #while(1):
        #    random_scale = abs(round(random.random(),4)-0.5)
        #    if abs(random_scale)>0.1:
        #        break


        if random_scale<=0.6:
            img = img+random_scale*mask
        else:
            #img = np.clip(img*(1-mask)+(random_scale+0.5)*mask,0,1)
            img = img*(1-mask)+img*(random_scale+1)*mask
        return torch.clamp(img,0,1)

    def noise_img(self, img,mask):
        size = mask.shape[1]
        noise = torch.rand(1,size,size)
        img = torch.clamp(img+noise*mask,0,1)
    
        return img

    def shift_img(self, img,mask):
        size = mask.shape[1]
        shiftnum1 = random.randint(10,size-10)
        shiftnum2 = random.randint(10,size-10)
        matrix=torch.vstack((img[(size-shiftnum1):,:,:],img[:(size-shiftnum1),:,:]))
        matrix=torch.hstack((matrix[:,(size-shiftnum2):,:],matrix[:,:(size-shiftnum2),:]))
        img = img*(1-mask)+mask*(matrix+0.01)
        return img

        
    def __getitem__(self, index):
        if self.test:
            test_path = self.test_file[index]
            mask_path = self.mask_file[index]
            
            if self.gray:
                img = cv2.imread(test_path,-1)
                mask = cv2.imread(mask_path,-1)
            else:
                img = cv2.imread(test_path)
                mask = cv2.imread(mask_path)
        else:
            if self.gray:
                img_path = os.path.join(self.filepath,self.file[index])
                img = cv2.imread(img_path,-1)
                mask = self.random_mask(img)
                mask = mask*255
            else:
                img = cv2.imread(os.path.join(self.filepath,self.file[index]))
                mask = self.random_mask(img)
                mask = mask*255
                mask = np.repeat(mask,3,2)

        
        img = Image.fromarray(img.astype('uint8')).convert('L')
        #ano_img = Image.fromarray(ano_img.astype('uint8')).convert('L')
        mask = Image.fromarray(mask.astype('uint8')).convert('L')
        
        if self.test:
            if self.transform is not None:
                img = self.mask_transform(img)
                #img = adjust_contrast(img,1.1)
                img = adjust_brightness(img,1.5)
                mask = self.mask_transform(mask)
            #ano_img = self.transform(ano_img)
            return 1,mask,img

        if self.transform is not None:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            #ano_img = self.transform(ano_img)


        random_num = random.random()

        if random_num>0.25:
            ano_img = self.const_img(img,mask)
            while(torch.sum(ano_img-img)==0):
                #assert 1==2
                ano_img = self.const_img(img,mask)
        #elif random_num>0.25:
        #    ano_img = self.shift_img(img,mask)
        #    while(torch.sum(ano_img-img)==0):
        #        #assert 1==2
        #        ano_img = self.const_img(img,mask)
            
        else:
            ano_img = self.noise_img(img,mask)
            while(torch.sum(ano_img-img)==0):
                #assert 1==2
                ano_img = self.shift_img(img,mask)
                
        return ano_img,mask,img

    def __len__(self):
        if self.test:
            return len(self.test_file)
        else:
            return len(self.file)

#CNN model
class Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim,
                                128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):  
    def __init__(self,opt):
        super().__init__()
        self.lin1 = nn.Linear(opt.latent_dim, 40)
        self.drop1 = nn.Dropout2d(0.25)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.lin2 = nn.Linear(40, 40*2)
        self.drop2 = nn.Dropout2d(0.25)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.lin3 = nn.Linear(40*2, 1)
        self.act3 = nn.Identity()
        #self.act4 = nn.Sigmoid()
    def forward(self, x):
        x = self.drop1(self.lin1(x))
        x = self.act1(x)
        x = self.drop2(self.lin2(x))
        x = self.act2(x)
        x = self.lin3(x)
        x = self.act3(x)
        output = x
        #output = self.act4(x)
        return output,x


class Encoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *encoder_block(opt.channels, 16, bn=False),
            *encoder_block(16, 32),
            *encoder_block(32, 64),
            *encoder_block(64, 128),
        )



        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2,
                                                 opt.latent_dim),
                                       nn.Identity())
        
    def forward(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity
    
    
#gradient penalty computation of WGAN-GP
lambda_gp = 10
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(*real_samples.shape[:2], 1, 1, device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_i_output,d_interpolates = D(interpolates)
    fake = torch.ones(*d_interpolates.shape, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#weight initialization
def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor

def weight_init(net):
    for op in net.modules():
        if isinstance(op,nn.Conv2d):
            op.weight.data=truncated_normal_(op.weight.data,std=0.02)
            nn.init.constant_(op.bias.data, val=0)
        else:
            pass


def get_data_transforms(image_size):
    img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            #transforms.GaussianBlur(5, sigma=(2, 2)),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor(),
        ])

    return img_transform

def get_test_transforms(image_size):
    img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            #transforms.GaussianBlur(5, sigma=(2, 2)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return img_transform

def get_mask_transforms(image_size):
    img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            #transforms.GaussianBlur(, sigma=(1, 1)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return img_transform

#train aae
def train(opt,load_model=False):
    if type(opt.seed) is int:
        torch.manual_seed(opt.seed)
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    transform = transforms.Compose([transforms.Resize([opt.img_size]*2),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5), inplace=True),])
    
    train_path = './hpc'
    model_path = './result'
    
    data_transform = get_data_transforms(128)
    mask_transform = get_mask_transforms(128)
    train_data = MyDataLoader(filepath=train_path,transform=data_transform,mask_transform=mask_transform,gray=True)
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=16,shuffle=True)
    
    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)

    weight_init(generator)
    weight_init(discriminator)
    weight_init(encoder)


    generator.to(device)
    discriminator.to(device)
    encoder.to(device)

    # optimizers
    #optim_Gen = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_Dec = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_Enc = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_Gen = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optim_Dis = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    writer = SummaryWriter(model_path+'/tensorboard')

    for epoch in range(opt.n_epochs):
        begin_time = time.time()
        for i,(imgs,_lbl,_msk) in enumerate(train_dataloader):
            real_imgs = imgs.to(device)
            real_z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)
            optim_Enc.zero_grad()
            optim_Dec.zero_grad()
            optim_Dis.zero_grad()

            # reconstruct
            encoder_output=encoder(real_imgs)
            output = generator(encoder_output)
            autoencoder_loss = torch.mean(torch.sqrt(torch.sum(torch.square(output - real_imgs),[1, 2, 3])))

            #l2 loss
            l2_loss = nn.MSELoss()
            reconstruct_loss_l2 = l2_loss(real_imgs,output)
            reconstruct_loss = autoencoder_loss+0.5*reconstruct_loss_l2
            reconstruct_loss.backward()

            optim_Enc.step()
            optim_Dec.step()

            # discriminator
            encoder.eval()
            
            with torch.no_grad():
                z_fake = encoder(real_imgs)
            real_d,real_logits = discriminator(real_z)
            fake_d,fake_logits = discriminator(z_fake)
            gradient_penalty = compute_gradient_penalty(discriminator,
                                                        real_z.data,
                                                        z_fake.data,
                                                        device)
            d_loss = -torch.mean(real_logits) + torch.mean(fake_logits) + lambda_gp * gradient_penalty
            d_loss.backward()
            optim_Dis.step()

            # encoder
            encoder.train()
            z_fake = encoder(real_imgs)
            e_fake,e_fake_logits = discriminator(z_fake)
            g_loss = -torch.mean(e_fake_logits)
            #g_loss = -torch.mean(torch.log(e_fake_logits + 1e-8))
            g_loss.backward()
            optim_Gen.step()

            #print(f"[Epoch {epoch}/{opt.n_epochs}] "
            #      f"[Batch {i}/{len(train_dataloader)}] "
            #      f"[R loss: {reconstruct_loss.item():3f}] "
            #      f"[D loss: {d_loss.item():3f}] "
            #      f"[G loss: {g_loss.item():3f}]")

            #visualization
            writer.add_scalar('Reconstruction Loss',reconstruct_loss,epoch*opt.batch_size+i)
            writer.add_scalar('Discriminator Loss',d_loss,epoch*opt.batch_size+i)
            writer.add_scalar('Generator Loss',g_loss,epoch*opt.batch_size+i)
            writer.add_histogram('Encoder Distribution',encoder_output,epoch*opt.batch_size+i)
            writer.add_histogram('Real Distribution',real_z,epoch*opt.batch_size+i)
            #print(torch.max(real_imgs[0]))
            #print(torch.min(real_imgs[0]))
            #print(torch.max(output[0]))
            #print(torch.min(output[0]))
            input_image = real_imgs[0].reshape(opt.channels,opt.img_size,opt.img_size)
            generate_image = output[0].reshape(opt.channels,opt.img_size,opt.img_size)
            writer.add_image('input image',input_image,epoch*opt.batch_size+i)
            writer.add_image('generate image',generate_image,epoch*opt.batch_size+i)
        print(f"[Epoch {epoch}/{opt.n_epochs}] "
              f"[R loss: {reconstruct_loss.item():3f}] "
              f"[D loss: {d_loss.item():3f}] "
              f"[G loss: {g_loss.item():3f}]")
    torch.save(generator.state_dict(), model_path + "/aae_generator.pth")
    torch.save(discriminator.state_dict(), model_path+"/aae_discriminator.pth")
    torch.save(encoder.state_dict(), model_path+"/aae_encoder.pth")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3,
                        help="number of image channels")
    parser.add_argument("--seed", type=int, default=None,
                        help="value of a random seed")
    opt = parser.parse_args(['--seed',str(1),'--n_epochs',str(1000),'--lr',str(0.0001),
                             '--img_size',str(128),'--latent_dim',str(128),
                             '--channels',str(1)])

    train(opt)