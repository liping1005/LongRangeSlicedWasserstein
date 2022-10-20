import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from slicing import *

##Author: alchua1996##

def synthesis(network, device, ref_img, syn_img, epochs = 50, lr = 1):
    #harr = DWTForward(J=2, mode='zero', wave='haar').to(device) 
    #inverse_harr = DWTInverse(mode='zero', wave='haar').to(device)  
    optimizer = torch.optim.LBFGS([syn_img], 
                                   lr=lr, 
                                   max_iter=50, 
                                   max_eval=64, 
                                   tolerance_grad=0, 
                                   tolerance_change=0)
    #estimate the noise using the HH band of the reference image
    #low_ref, bandref = harr(ref_img.detach().unsqueeze(0))
    #noise = torch.median(torch.abs(torch.flatten(bandref[0][:,:,-1,:,:])))/.6745
    #ref_thres = noise * np.sqrt(2*np.log(3*ref_img.shape[-2]*ref_img.shape[-1]))
    
    #perform PCA here
    #PCA = PCA_image(ref_img)
    #ref_pca = torch.transpose(PCA.transform(), 0, 1).reshape((3, ref_img.shape[-2], ref_img.shape[-1]))
    #ref_dct = dct_2d(ref_img, norm = 'ortho')
    for k in range(epochs):
        ref_stat = network(ref_img)
        for item in ref_stat:
            item = item.detach()
        def closure(): 
            optimizer.zero_grad()   
            #syn_img.data.clamp_(0, 1)
            syn_maps = network(syn_img) 
            losses = [F.mse_loss(ref_stat[n], syn_maps[n]) for n in range(len(syn_maps))]    
            loss = torch.stack(losses, dim = 0).sum()/len(losses)/255**2
            loss.backward()
            return loss 
        optimizer.step(closure)
        for ele in network.slicing_channels:
            ele.update_slices()
        for ele in network.slicing_width:
            ele.update_slices()
        for ele in network.slicing_height:
            ele.update_slices()
        if((k+1) % 5 == 0):
            print('Epoch {} Complete!'.format(k+1))
            print('Loss is:', closure().item())  
    #low_syn, bandsyn = harr(syn_img.unsqueeze(0))
    #for coeff in bandsyn:
        #coeff = softThreshold(coeff, ref_thres)
        #syn_thres = inverse_harr((low_syn, bandsyn))
        #syn_img = torch.autograd.Variable(syn_thres.squeeze())
        
    #undo stupid pca now
    #syn_img = PCA.inverse_transform(torch.transpose(syn_img.reshape(3,-1),0,1))
    #syn_img = torch.transpose(syn_img, 0, 1).reshape((3, ref_img.shape[-2], ref_img.shape[-1]))
    
    #syn_img = idct_2d(syn_img, norm='ortho')
    syn_img.data.clamp_(0, 1)
    return syn_img

def gonthier_synthesis(mydict, layers_channel, layers_width, layers_height, device, ref_img, scales, epochs, lr):
    assert scales == len(lr)
    assert scales == len(epochs)
    #get sizes
    height = ref_img.shape[1]
    width = ref_img.shape[2]
    #initialize white noise and make it into a variable to use autograd on
    noise = torch.rand(3,int(width/2**(scales-1)),int(height/2**(scales-1))).to(device) *1e-2
    #make variable
    syn_image = torch.autograd.Variable(noise, requires_grad=True)
    #do the loop
    for j in range(scales):  
        #interpolate  
        ref = F.interpolate(ref_img.unsqueeze(0), scale_factor = 1/2**(scales-1-j), mode = 'bilinear').squeeze()
        #initalize a network here
        net =  VGG19layers(mydict, ref.shape[1], ref.shape[2], layers_channel[j], layers_width[j], layers_height[j], device)
        #do synthesis
        out = synthesis(net, device, ref, syn_image, epochs[j], lr[j])
        print('Layer {} synthesis complete!'.format(j+1))
        if(j != scales-1):
            out = F.interpolate(out.unsqueeze(0), scale_factor =2, mode = 'bilinear').squeeze()
            #add noise for regularization
            syn_image = torch.autograd.Variable(out.detach(), requires_grad=True)
    return syn_image