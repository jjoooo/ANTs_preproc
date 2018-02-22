import os
import numpy as np
from glob import glob

import torch
import torch.nn as nn
from skimage import io

from resnet2d import Resnet, Classifier
import warnings
warnings.filterwarnings("ignore")


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def path_glob(ext, depth, data_name, path):
    if data_name == 'MICCAI2008':
        flair = glob(path + '/*FLAIR' + ext)
        t1 = glob(path + '/*T1' + ext)
        t2 = glob(path + '/*T2' + ext)
        gt = glob(path + '/*lesion' + ext)

    else:
        flair=[]; t1=[]; t2=[]; gt=[];
        for i in range(depth):
            img = glob(path + '/{}.nii'.format(i))
            mask = glob(path + '/{}_mask.PNG'.format(i))
            t1.append(img[0])
            gt.append(mask[0])

    return flair, t1, t2, gt


def save_img(idx, root_path, n_mode, slices_by_mode, volume_depth, label_bl=False):
    # Origin image
    o_path = root_path+'/test_origin_PNG/{}'.format(idx)
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    for m in range(n_mode-1):
        for slx in range(volume_depth):
            io.imsave(o_path+'/{}_origin.PNG'.format(slx), slices_by_mode[m][slx])

    # Label image (no label on real data)
    if label_bl:
        l_path = root_path+'/test_label_PNG/{}'.format(idx)
        if not os.path.exists(l_path):
            os.makedirs(l_path)
        for slx in range(volume_depth):
            io.imsave(l_path+'/{}_label.PNG'.format(slx), slices_by_mode[-1][slx])


def init_model(args, n4b):

    # Model init
    resnet_s = nn.DataParallel(Resnet(args.n_mode-1)).cuda() # center patch model
    resnet_b = nn.DataParallel(Resnet(args.n_mode-1)).cuda() # downsampling patch model
    classifier = Classifier(args.batch_size).cuda()
    models = [resnet_s, resnet_b, classifier]

    # If is_exists Pretrained model: loading
    if not os.path.exists('./trained_model'):
        os.makedirs('./trained_model')
    add_str = ''
    if n4b: add_str = '_n4'
    path = './trained_model/model_lr{}_ps{}'.format(args.learning_rate, args.patch_size) + add_str
    if not os.path.exists(path):
        os.makedirs(path)

    model_idx = 1
    for m in range(len(models)):
        models_path = glob(path+'/mode_{}/miccai_{}.pkl'.format(args.n_mode,m))

        # exist models
        if models_path:
            model_path = path+'/mode_{}/miccai_{}.pkl'.format(args.n_mode,m)
            print('pretrained s model loading: '+model_path)
            models[m].load_state_dict(torch.load(model_path))

    return models, path



def dcm2png(path):
    patients = glob(path + '/**')
    
    for patient in patients:

        im_path = glob(patient + '/**/*.dcm')
        for p in im_path:
            fn = p[:-3]+'PNG'
            img = io.imread(p, plugin='simpleitk').astype(float)
            t = np.percentile(img, 99)
            img = np.clip(img, 0, t)
            img = (img - np.mean(img)) / np.std(img)
            if np.max(img) != 0: # set values < 1
                img /= np.max(img)
            if np.min(img) <= -1: # set values > -1
                img /= abs(np.min(img))

            io.imsave(fn, img[0])

if __name__ == "__main__":
    dcm2png('/Users/jui/Downloads/Data/YS/MS')

    
