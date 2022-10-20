import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

#get vgg from file
from vgg import *

##Author liping1005##

#inspiration from https://github.com/xchhuang/pytorch_sliced_wasserstein_loss/tree/main/pytorch
class Slicing(nn.Module):
    """ 
    Slicing layer: computes projections and returns sorted vector 
    
    num_slices: number of slices; chosen to be number of feature maps in original paper
    """
    
    def __init__(self, num_slices, device):
        super().__init__()
        # Number of directions
        self.device = device
        self.num_slices = num_slices
        self.dim_slices = num_slices
        self.update_slices()
    
    def update_slices(self):
        '''
        Generate directions to project onto
        '''
        self.directions = torch.randn(size=(self.num_slices, self.dim_slices)).to(self.device)
        norm = torch.sqrt(torch.sum(torch.square(self.directions), axis=-1))
        norm = norm.view(self.num_slices, 1)
        self.directions = self.directions / norm

    def forward(self, input):
        '''
        Project image onto slices
        '''
        b,c,w,h = input.shape
        tensor = input.permute(0,2,3,1)
        tensor = tensor.contiguous().view(b, -1, c)
        # Project each pixel feature onto directions (batch dot product)
        sliced = self.directions @ tensor.permute(0, 2, 1)
        # Sort projections for each direction
        sliced, _ = torch.sort(sliced)
        sliced = sliced.view(b, -1)
        return sliced
        
class VGG19layers(nn.Module):
    def __init__(self, model_dict, H, W, channel_maps, width_maps, height_maps, device):
        '''
        INPUTS
        ------
        H: height of image (number of rows)
        W: width of image (number of columns)
        channel_maps: 
        width_maps: 
        height_maps:
        device: cuda or cpu
        '''
        super(VGG19layers,self).__init__()
        #load vgg model 
        self.model = VGG().to(device).eval()
        self.model.load_state_dict(model_dict)
        #maps for which layers to take
        self.channel_maps = channel_maps
        self.width_maps = width_maps
        self.height_maps = height_maps
        #slices for channels
        self.slicing_channels = []
        #slices for image height
        self.slicing_height = []
        #slices for image width
        self.slicing_width = []
        for item in self.channel_maps:
            if(item == 'r11' or item == 'r12'):
                self.slicing_channels.append(Slicing(64, device))
            elif(item == 'r21' or item == 'r22'):
                self.slicing_channels.append(Slicing(128, device))
            elif(item == 'r31' or item == 'r32'or item == 'r33' or item == 'r34'):
                self.slicing_channels.append(Slicing(256, device))
            elif(item == 'r41' or item == 'r42' or item == 'r43' or item == 'r44'):
                self.slicing_channels.append(Slicing(512, device))
            elif(item == 'r51' or item == 'r52' or item == 'r53' or item == 'r54'):
                self.slicing_channels.append(Slicing(512, device))
        for item in self.width_maps:
            if(item == 'r11' or item == 'r12'):
                self.slicing_width.append(Slicing(int(W), device))
            elif(item == 'r21' or item == 'r22'):
                self.slicing_width.append(Slicing(int(W/2), device))
            elif(item == 'r31' or item == 'r32'or item == 'r33' or item == 'r34'):
                self.slicing_width.append(Slicing(int(W/4), device))
            elif(item == 'r41' or item == 'r42' or item == 'r43' or item == 'r44'):
                self.slicing_width.append(Slicing(int(W/8), device))
            elif(item == 'r51' or item == 'r52' or item == 'r53' or item == 'r54'):
                self.slicing_width.append(Slicing(int(W/16), device))
        for item in self.height_maps:
            if(item == 'r11' or item == 'r12'):
                self.slicing_height.append(Slicing(int(H), device))
            elif(item == 'r21' or item == 'r22'):
                self.slicing_height.append(Slicing(int(H/2), device))
            elif(item == 'r31' or item == 'r32'or item == 'r33' or item == 'r34'):
                self.slicing_height.append(Slicing(int(H/4), device))
            elif(item == 'r41' or item == 'r42' or item == 'r43' or item == 'r44'):
                self.slicing_height.append(Slicing(int(H/8), device))
            elif(item == 'r51' or item == 'r52' or item == 'r53' or item == 'r54'):
                self.slicing_height.append(Slicing(int(H/16), device))
         
        #turn off gradients
        for param in self.model.parameters():
            param.requires_grad = False

        self.my_transform = transforms.Compose([ 
                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                std=[1,1,1]),
                           transforms.Lambda(lambda x: x.mul_(255))])

                           

    def forward(self, x):
        x_inp = self.my_transform(x.clone()).unsqueeze(0)
        feat_maps = self.model(x_inp)
        outputs = []
        count = 0
        for key in list(feat_maps.keys()):
            if(key in self.channel_maps):
                outputs += self.slicing_channels[count](feat_maps[key])
                count+=1             
        count = 0
        for key in list(feat_maps.keys()):
            if(key in self.width_maps):
                outputs += self.slicing_width[count](feat_maps[key].permute(0,2,3,1))
                count+=1
        count = 0
        for key in list(feat_maps.keys()):
            if(key in self.height_maps):
                outputs += self.slicing_height[count](feat_maps[key].permute(0,3,2,1))
                count+=1
        return outputs