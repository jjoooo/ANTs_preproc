import os

import torch
from torch.autograd import Variable

from skimage import io
from sklearn.preprocessing import minmax_scale

from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
from sklearn.metrics import roc_auc_score

from glob import glob

import numpy as np
import time


        
def testing(args, test_batch, models, idx):

    print('Batch len = {}\n'.format(len(test_batch)))
    output_prob = np.zeros([args.volume_size, args.volume_size, args.volume_size])
    print('Patch segmentation test start...')

    tic = time.time()

    resnet_s = models[0]
    resnet_b = models[1]
    classifier = models[2]
    
    for img,_,p in test_batch:
        
        coord = []
        for pp in p:
            z,y,x,etc = pp.split('_',3)
            coord.append([z,y,x])

        mid = int(args.patch_size/2)

        x1 = Variable(img[:,:,:mid]).cuda()
        x2 = Variable(img[:,:,mid:]).cuda()
    
        out_s = resnet_s.forward(x1)
        out_b = resnet_b.forward(x2)

        concat_out = torch.cat([out_s,out_b],dim=1)
        out = classifier.forward(concat_out)

        out_arr = out.data.cpu().numpy()  
        tar_arr = _.numpy()
        
        for bc, cd in enumerate(coord):
            z = int(cd[0])
            y = int(cd[1])
            x = int(cd[2])

            h1 = y-int(args.patch_size/2)
            h2 = y+int(args.patch_size/2)
            w1 = x-int(args.patch_size/2)
            w2 = x+int(args.patch_size/2)

            if h1 < 0 or h2 > args.volume_size or w1 < 0 or w2 > args.volume_size:
                continue
            mid = int(args.patch_size/4) 
            output_prob[z, h1+mid:h2-mid, w1+mid:w2-mid] += out_arr[bc][0]


    print('Done. (prediction elapsed: %.2fs)' % (time.time() - tic))

    save_result(args, output_prob, idx)



    
def save_result(args, output_prob, idx):

    path = args.root + args.data_name + '/test_result_PNG/{}_{}_{}_{}'.format(idx, args.patch_size, args.n_patch, args.n_mode)
    if not os.path.exists(path):
        os.makedirs(path)

    thsd = 0.2
    output_prob[output_prob<thsd] = 0 

    output_prob = (output_prob-np.min(output_prob))/(np.max(output_prob)-np.min(output_prob))
    print('after minmax \noutput_prob : min={}, max={}\n'.format(np.min(output_prob),np.max(output_prob)))
    print('output_prob : mean = {}'.format(np.mean(output_prob)))

    origin_path = glob(args.root+args.data_name+'/test_origin_PNG/{}/**'.format(idx))

    origin_volume = np.zeros([args.volume_size, args.volume_size, args.volume_size])

    if args.data_name != 'YS':
        label_path = glob(args.root+args.data_name+'/test_label_PNG/{}/**'.format(idx))
        if label_path: 
            label_volume = np.zeros([args.volume_size, args.volume_size, args.volume_size])
            for ix in range(args.volume_size):
                origin_volume[ix] = io.imread(args.root+args.data_name+'/test_origin_PNG/{}/{}_origin.PNG'.format(idx,ix), plugin='simpleitk').astype(float)
                label_volume[ix] = io.imread(args.root+args.data_name+'/test_label_PNG/{}/{}_label.PNG'.format(idx,ix), plugin='simpleitk').astype(float)
    
    else:
        for ix in range(len(origin_path)):
            org = io.imread(args.root+args.data_name+'/test_origin_PNG/{}/{}_origin.PNG'.format(idx,ix), plugin='simpleitk').astype(float)
            h,w = org.shape
            origin_volume[ix][0:h, 0:w] = org[0:h, 0:w]

    vol = img_as_float(origin_volume)
    vol = (vol-np.min(vol))/(np.max(vol)-np.min(vol))
    vol = adjust_gamma(color.gray2rgb(vol), 0.5)


    red_mul = [1,0,0]

    infer_rgb = img_as_float(output_prob)
    infer_rgb = color.gray2rgb(infer_rgb)

    infer_rgb = red_mul * infer_rgb
    infer_rgb = adjust_gamma(infer_rgb, 0.8)
    
    vol_inf = vol+infer_rgb

    vol_inf = (vol_inf - np.mean(vol_inf)) / np.std(vol_inf)
    if np.max(vol_inf) != 0: # set values < 1
        vol_inf /= np.max(vol_inf)
    if np.min(vol_inf) <= -1: # set values > -1
        vol_inf /= abs(np.min(vol_inf))

    if args.data_name != 'YS':
        label_volume[label_volume>0] = 1
        label_rgb = img_as_float(label_volume)
        label_rgb = color.gray2rgb(label_rgb)
        label_rgb = red_mul * label_rgb
        label_rgb = adjust_gamma(label_rgb, 0.5)

        vol_label = vol+label_rgb

        vol_label = (vol_label - np.mean(vol_label)) / np.std(vol_label)
        if np.max(vol_label) != 0: # set values < 1
            vol_label /= np.max(vol_label)
        if np.min(vol_label) <= -1: # set values > -1
            vol_label /= abs(np.min(vol_label))

    
        i = 0
        for slice_ori, slice_inf, slice_label in zip(vol,vol_inf,vol_label):
            concat_img = np.concatenate((slice_ori,slice_inf,slice_label), axis=1)
            if np.max(concat_img) == 0:
                continue
            io.imsave(path+'/{}_predict.PNG'.format(i), concat_img)
            #io.imsave(path+'/{}_predict_inf.PNG'.format(i), slice_inf)
            #io.imsave(path+'/{}_predict_rgb_class.PNG'.format(i), slice_label)
            i += 1

    else:
        i = 0
        for slice_ori, slice_inf in zip(vol,vol_inf):
            concat_img = np.concatenate((slice_ori,slice_inf), axis=1)
            if i> len(origin_path)-1:
                break
            io.imsave(path+'/{}_predict.PNG'.format(i), concat_img)
            #io.imsave(path+'/{}_predict_inf.PNG'.format(i), slice_inf)
            #io.imsave(path+'/{}_predict_rgb_class.PNG'.format(i), slice_label)
            i += 1

    print('Volume saved.')
