import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import time
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

#importing necessary files
from synthesis import *
from utils import *

torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load vgg weights  
mydict = torch.load('vgg_conv.pth')

#time to argparse my soul away
parser = argparse.ArgumentParser(description='Arguments for the network')
parser.add_argument('--image_size', type = int, required = True, help = 'Size of texture. Should be a power of 2.')
parser.add_argument('--image_path', type = str, required = True, help = 'Input texture file path.')
parser.add_argument('--output_path', type = str, required = True, help = 'Path/name of file to place synthesized texture at.')
parser.add_argument('--file_type', type = str, required = True, help = 'jpg or png')
parser.add_argument('--num_scales', type = int, required = True, help = 'Number of scales for multiscale procedure.')

#arg parse
args = parser.parse_args()
size = args.image_size
image_path = args.image_path
output_path = args.output_path
file_type = args.file_type
num_scales = args.num_scales

#resize reference image to be desired size
resizer = torchvision.transforms.Resize((size,size))
if(file_type == 'jpg'):
    ref_img_orig = torch.Tensor(mpimg.imread(image_path)/255.0).permute(2,0,1)
elif(file_type == 'png'):
    ref_img_orig = torch.Tensor(mpimg.imread(image_path)).permute(2,0,1)
else:
    print('Error. You did not pick a valid option')
ref_img = resizer(ref_img_orig)
ref_img = torch.Tensor(ref_img)
print(ref_img.shape)

#initialize network
#layers_channels = [[]]*num_scales
layers_channels = [['r11', 'r12', 'r21', 'r22', 'r31', 'r32', 'r33', 'r34', 'r41', 'r42', 'r51', 'r52']]*num_scales
#layers_height = [['r11', 'r12', 'r21', 'r22', 'r31', 'r32', 'r41', 'r42', 'r51', 'r52']]*num_scales
layers_height = [[]]*num_scales
layers_width = [[]]*num_scales
ref_img = ref_img.to(device)
print('Network Initialized. Synthesis Beginning.')              

time_list = []                    
#do synthesis
for j in range(5):
  start = time.time()
  syn_image = gonthier_synthesis(mydict, 
                                 layers_channels, 
                                 layers_height,
                                 layers_width,
                                 device, 
                                 ref_img, 
                                 scales = num_scales, 
                                 epochs = [20]*num_scales, 
                                 lr = [1.0]*num_scales)                               
  end = time.time()
  time_list.append(end-start)
  
#detach and hist match
out = np.float64(syn_image.detach().cpu().permute(1,2,0).numpy())
ref = np.float64(ref_img.detach().cpu().permute(1,2,0). numpy())
plt.imsave(output_path, out)
print('Synthesis Complete. Image saved in folder')
print('Run Time for Algorithm:', time_list)