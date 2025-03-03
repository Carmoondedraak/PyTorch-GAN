"""
Inpainting using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 context_encoder.py'
"""
import matplotlib.pyplot as plt
import argparse
import os
import sys
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *
from light_cnn import *
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.tensorboard import SummaryWriter

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--train_list", type=str, default="CelebA/Anno/new_labels_train.txt", help="list of train images")
parser.add_argument("--val_list", type=str, default="CelebA/Anno/new_labels_val.txt", help="list of val images")
parser.add_argument("--root_path", type=str, default="", help="path to the root directory of dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=128, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
parser.add_argument('--num_classes', default=7417, type=int, metavar='N', help='number of classes (default: 99891)')
parser.add_argument('--resume', default='saveslightCNN_80_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)

writer = SummaryWriter()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Loss function
adversarial_loss = torch.nn.MSELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
generator = Generator(channels=opt.channels)
discriminator = Discriminator(channels=opt.channels)
model = LightCNN_9Layers(num_classes=opt.num_classes)
model.eval()
model = torch.nn.DataParallel(model)



generator = torch.nn.DataParallel(generator)
discriminator = torch.nn.DataParallel(discriminator)
# if cuda:
generator.cuda()
discriminator.cuda()
adversarial_loss.cuda()
pixelwise_loss.cuda()
model.cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'])

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
#generator.load_state_dict(torch.load('PyTorch-GAN/models/generator_334000.pth'))
#discriminator.load_state_dict(torch.load('PyTorch-GAN/models/discriminator_334000.pth'))
# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

dataloader = DataLoader(
    ImageDataset(opt.root_path + "/train", opt.train_list, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

test_dataloader = DataLoader(
    ImageDataset(opt.root_path + "/val", opt.val_list,transforms_=transforms_),
    batch_size=12,
    shuffle=True,
    num_workers=1,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def correct(output, targets):
    max_values, predictions = torch.max(output, 1)
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == targets[i]:
            count += 1
    # print(targets, predictions)
    percentage = count / len(predictions)
    # print(percentage)
    return percentage


def save_sample(batches_done):
    samples, masked_samples, i, targets = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()  # Upper-left coordinate of mask
    # Generate inpainted image
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    # i = int(i)
    # filled_samples[:, :, i : i + opt.mask_size, i : i + opt.mask_size] = gen_mask
    # Save sample
    sample = torch.cat((masked_samples.data, gen_mask.data, samples.data), -2)
    save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)

checkpoint = torch.load('PyTorch-GAN/models/old_models/model_21.tar')
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



# ----------
#  Training
# ----------
for epoch in range(last_epoch, opt.n_epochs): 
    for i, (imgs, masked_imgs, aux, targets) in enumerate(dataloader):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

        # Configure input
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        # masked_parts = Variable(masked_parts.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()
        gen_parts = generator(masked_imgs)
        
        # -----------------
        #  recogntion_check
        # -----------------
        gray_gen = transform(gen_parts)
        output, _ = model(gray_gen)
        recognition_loss = correct(output, targets)
       
        
        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid)
        #print(discriminator(gen_parts).shape)
        g_pixel = pixelwise_loss(gen_parts, imgs)
	    # Total loss
        g_loss = 0.001 * g_adv + 0.999 * g_pixel + 3 * recognition_loss

        g_loss.backward()
        optimizer_G.step()
        writer.add_scalar('gLoss/train', g_loss, epoch)
        writer.add_scalar('gAdv/train', g_adv, epoch)
        writer.add_scalar('gPixel/train', g_pixel, epoch)
        writer.add_scalar('recognition_loss/train', recognition_loss, epoch)
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        writer.add_scalar('dLoss/train', d_loss, epoch)
        writer.add_scalar('fakeLoss/train', fake_loss,epoch)
        writer.add_scalar('realLoss/train', real_loss, epoch)
        d_loss.backward()

        optimizer_D.step()
        # ---------------------
        #  Recognition model check
        # ---------------------


        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
        )

        # Generate sample at sample interval
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_sample(batches_done)

    torch.save({
        'epoch': epoch,
        'modelG_state_dict' : generator.state_dict(),
        'modelD_state_dict' : discriminator.state_dict(),
        'optimizerG_state_dict' : optimizer_G.state_dict(),
        'optimizerD_state_dict' :  optimizer_D.state_dict(),
        'dloss': d_loss,
        'fakeLoss': fake_loss,
        'realLoss': real_loss,
        'gloss' : g_loss,
        'gadv' : g_adv,
        'pixelloss' : g_pixel
        }, 'PyTorch-GAN/models/model_new_{}.tar'.format(epoch))
writer.close()
