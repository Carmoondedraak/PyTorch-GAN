'''
    implement the feature extractions for light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''

from __future__ import print_function
import argparse
import os
import shutil
import time
from PIL import Image, ImageOps
import glob

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import numpy as np
import cv2

from light_cnn import LightCNN_9Layers
from load_imglist import ImageList
from models import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--resume', default='saveslightCNN_80_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='LightCNN-9', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')
parser.add_argument('--root_path', default='', type=str, metavar='PATH', 
                    help='root path of face images (default: none).')
parser.add_argument('--img_list', default='CelebA/Anno/new_labels_test.txt', type=str, metavar='PATH', 
                    help='list of face images for feature extraction (default: none).')
parser.add_argument('--save_path', default='/save_features', type=str, metavar='PATH', 
                    help='save root path for features of face images.')
parser.add_argument('--num_classes', default=7417, type=int,
                    metavar='N', help='mini-batch size (default: 79077)')
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")


def main():
    global args
    args = parser.parse_args()

    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    else:
        print('Error model type\n')

    model.eval()
    # if args.cuda:
    print('CUDA!')
    model = torch.nn.DataParallel(model).cuda()

    generator = Generator(channels=args.channels)
    discriminator = Discriminator(channels=args.channels)
    model.eval()
    generator.eval()
    discriminator.eval()
    # model = torch.nn.DataParallel(model)
    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)
    # if cuda:
    generator.cuda()
    discriminator.cuda()

    model.cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint1 = torch.load(args.resume)
            model.load_state_dict(checkpoint1['state_dict'])

            checkpoint = torch.load('PyTorch-GAN/models/model_new_76.tar')
            generator.load_state_dict(checkpoint['modelG_state_dict'])
            discriminator.load_state_dict(checkpoint['modelD_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizerD_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizerG_state_dict'])
            d_loss = checkpoint['dloss']
            fake_loss = checkpoint['fakeLoss']
            real_loss = checkpoint['realLoss']
            g_loss = checkpoint['gloss']
            g_adv = checkpoint['gadv']
            pixelloss = checkpoint['pixelloss']
            last_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))





    img_list  = read_list(args.img_list)
    transform2 = transforms.Compose([transforms.Resize((128,128), Image.BICUBIC), transforms.ToTensor()])
    transform1 = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
    transform = transforms.Compose([
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
    count     = 0
    correct1 = 0
    correct5 = 0
    not_correct = 0
    num_pixels = []
    # min_dif = []
    # max_dif = []
    # average_dif = []
    input     = torch.zeros(1, 3, 128, 128)
    input1     = torch.zeros(1, 1, 128, 128)
    new_mask     = torch.zeros(1, 3, 128, 128)
    # for img_name, target in img_list:
    count = count + 1
    
    # creating an og_image object
    # dir_name = args.root_path + '/'+ img_name 
    dir_name = 'carmen2.jpeg'
    img_name = dir_name
    target =1
    img = Image.open(dir_name)
    target_im = Image.open(dir_name)
    img   = transform2(img)
    target_im = transform2(target_im)

    masked_img, _ = apply_center_mask(img) 
    input[0,:,:,:] = masked_img

    start = time.time()
    if args.cuda:
        input = input.cuda()
    # input_var   = torch.autograd.Variable(input, volatile=True)
    
    gen_parts = generator(input)

    new_mask[0,:,:,:], num_pix = pixel_change(gen_parts,img, target_im, img_name)
    # gray_gen = transform1(new_mask)
    print(num_pix, 'num pixels')
    img = Image.open( "withmask/%s.png" % img_name).convert('L')
    # img = Image.open(args.root_path +'/'+ img_name).convert('L')
    img   = transform(img)
    input1[0,:,:,:] = img

    start = time.time()
    if args.cuda:
        input1 = input1.cuda()
    input_var   = torch.autograd.Variable(input1, volatile=True)
    output, _ = model(input_var)

    end = time.time() - start
    print("{}({}/{}). Time: {}".format(os.path.join(args.root_path, img_name), count, len(img_list), end))
    # save_feature(args.save_path, img_name, features.data.cpu().numpy()[0])
    
    prec5, prec1 = accuracy(output.data, int(target))
    correct1 += prec1
    correct5 += prec5
    num_pixels.append(num_pix)
    # min_dif.append(min_d)
    # max_dif.append(max_d)
    # average_dif.append(avg_d)

    print(correct1, 'correct')
    print(correct5, 'correct 5')
    print(count, 'count')
    
    # print('max diff', max_d, 'min diff', min_d, 'average diff', avg_d )
    if prec1 == 0 and count < 30:
        input_im = Image.open("mask%s.png" % img_name)
        input_im =  transform2(input_im)
        save_image(input_im, "result_images4/%s.png" % img_name, normalize=True)
        # break
    print(correct1, 'correct')
    print(correct5, 'correct 5')
    print(count, 'count')
    print(num_pix, 'num pixels')
    print(min(num_pixels), 'minimum number of pixels')
    print(max(num_pixels), 'maximum number of pixels')
    print(sum(num_pixels)/len(num_pixels), 'average num of pixels')


def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append((img_path[0], img_path[1]))
    print('There are {} images..'.format(len(img_list)))
    return img_list

def save_feature(save_path, img_name, features):
    img_path = os.path.join(save_path, img_name)
    img_dir  = os.path.dirname(img_path) + '/';
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    fname = fname + '.feat'
    fid   = open(fname, 'wb')
    fid.write(features)
    fid.close()

def apply_center_mask(img):
    """Mask center part of image"""
    # Get upper-left pixel coordinate
    i = (128 - 64) // 2
    masked_img = img.clone()
    masked_img[:, i : i + 64, i : i + 64] = 1

    return masked_img, i

def pixel_change(img, img1,target, img_name):
    pixel_diff = img - target.cuda()
    average = torch.mean(pixel_diff)
    maxi = torch.max(pixel_diff)
    mini = torch.min(pixel_diff)
    # print('minmax', mini, maxi, average)
    z = 0
    count = 0
    target1 = target.clone()
    # target1 = target1.resize((180,180))
    # print(target1.size())
    # print(len(target[0]))
    for x in range(len(target[0])):
        for y in range(len(target[0])):
            if (x >= 32 and x <= 96) and  (y >= 32 and y <= 96):
                # print(x,y)
                if pixel_diff[0,:,:,:][z,x,y] > (maxi - 0.7) and pixel_diff[0,:,:,:][z+1,x,y] > (maxi - 0.7) and pixel_diff[0,:,:,:][z+2,x,y] > (maxi - 0.7):
                    target1[z,x,y] = 1
                    target1[z + 1,x,y] = 1
                    target1[z + 2,x,y] = 1
                    count += 1

                elif pixel_diff[0,:,:,:][z,x,y] < (mini + 0.7) and pixel_diff[0,:,:,:][z+1,x,y] < (mini + 0.7) and pixel_diff[0,:,:,:][z+2,x,y] < (mini + 0.7):
                    target1[z,x,y] = 1
                    target1[z + 1,x,y] = 1
                    target1[z + 2,x,y] = 1
                    count += 1
    # print(count, 'count')
    print('yowat')
    sample = torch.cat((target1.cuda(), img[0,:,:,:].cuda()), -2)
    save_image(sample, "mask2%s.png" % img_name, normalize=True)    
    return target1, count

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # maxk = max(topk)
    topk5 = 0
    topk1 = 0
    _, pred = output.topk(5, 1, True, True)
    # print(pred, 'daguck')
    if target in pred:
        # print('weetje dit zeker? ', target)
        topk5 += 1

    max_value1, prediction1 = torch.max(output, 1)
    # print([prediction1])
    if prediction1 == target:
        topk1 += 1

    # print(max_value1, prediction1, target)
    # print(pred, 'daguck')
    return topk5,topk1

if __name__ == '__main__':
    main()