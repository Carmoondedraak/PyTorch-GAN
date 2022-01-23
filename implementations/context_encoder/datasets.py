import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
import matplotlib.pyplot as plt

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob("%s/*.png" % root))
        self.radius = 10
        file = '../../../CelebA/Anno/list_landmarks_align_celeba.txt'
        self.landmarks = self.read_landmarks(file)
        # self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        print(img.shape)
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def read_landmarks(self, file):
        '''reads in the landmarks from the file'''
        with open(file) as f:
            landmarks = defaultdict(list)
            lines = f.readlines()

            # skip the first line
            for i in range(len(lines)-1):
                # split into list and remove trailing spaces
                line = lines[i+1].split('\n')[0].split(" ")
                line = list(filter(None, line))

                # add to a list dictionary as tuples
                for i in range(1,len(line)-1,2):
                    landmarks[line[0]].append((int(line[i]),int(line[i+1])))
        return landmarks

    def delete_landmark_pixels(self, img, radius, im_name):
        ''' This function deletes landmark pixels given a certain raduis'''
        img1 = img
        masked_parts = []
        im_name = im_name.split('mirror')[-1].split('train')[-1].split('.')[0]
        im_name = im_name + '.jpg'
        # print(landmarks[im_name], 'landmarks')
        for landmark in self.landmarks[im_name]:
            rad = radius // 2
            x,y = landmark
            x = x-90
            y= y-50
            # find lower and upper bounds of landmarks and create radius to delete
            x1 = x - rad
            x2 = x + rad
            y1 = y - rad
            y2 = y + rad
            y_list = list(range(y1,y2))
            x_list = list(range(x1,x2))
            masked_img = img1.clone()
            # print(masked_img.shape, y_list, x_list)
            for x in x_list:
                for y in y_list:
                    for z in range(3):
                        masked_img[z,y,x] = 255 # or 0 if we want black
            # masked_part = img1[:, y1:y2, x1:x2]
            # masked_parts.append(masked_part)

            # masked_img[:, y1:y2, x1:x2] = 255
            # img1 = masked_img

        # img1.save("images/%d.png" % im_name)
        # masked_parts.save("images/maked%d.png" % im_name)
        plt.imshow(masked_img.detach().numpy().reshape(128,128,3))
        plt.show()
        return img1, masked_parts

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)

        print(self.mode)
        if self.mode == "train":
            # For training data perform random mask
            im_name = self.files[index % len(self.files)].split('/')[-1]
            # masked_img, aux = self.delete_landmark_pixels(img, self.radius, im_name)
            masked_img = img
            aux = img
        else:
            # For test data mask the center of the image
            # masked_img, aux = self.apply_random_mask(img)
            masked_img = img
            aux = img
        return img, masked_img, aux

    def __len__(self):
        return len(self.files)
