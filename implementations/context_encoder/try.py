from collections import defaultdict
import matplotlib.pyplot as plt
import random
import numpy
import PIL
from numpy import asarray
from PIL import Image

im = Image.open('../../data/img_align_celeba/{}'.format('000025.png'))
implot = plt.imshow(im)
plt.show()
