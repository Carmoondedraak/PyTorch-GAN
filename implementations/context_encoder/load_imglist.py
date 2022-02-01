import torch.utils.data as data

from PIL import Image
import os
import os.path
import glob
def default_loader(path):
    img = Image.open(path).convert('L')
    return img

def default_list_reader(fileList):
    imgDict = {}
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgDict[imgPath] = int(label)
    return imgDict

class ImageList(data.Dataset):
    def __init__(self, root, fileList, path, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.imgDict   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader
        self.path = path
        if self.path == 'train':
            self.files =  sorted(glob.glob("%s/train/*.png" % self.root))
        elif self.path == 'val':
            self.files = sorted(glob.glob("%s/val/*.png" % self.root))

    def __getitem__(self, index):
        imgPath = self.files[index]
        imgkey = imgPath.split('/')[-1]
        target = self.imgDict[imgkey]
        img = self.loader(os.path.join(imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgDict)
