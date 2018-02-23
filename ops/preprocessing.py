import os
import numpy as np
from glob import glob
import random

# image data
from skimage import io
import SimpleITK as sitk

from ops.patch_extraction import MakePatches
from ops.util import create_folder, path_glob, save_img

import warnings
warnings.filterwarnings("ignore")

import subprocess

scripts_dir = '/mnt/disk2/source/ANTs/Scripts/'



# Preprocessing
class Preprocessing(object):

    def __init__(self, args, n4, n4_apply):
        self.args = args
        self.n4bias = n4
        self.n4bias_apply = n4_apply
        self.train_bool = True # Default training
        self.data_name = args.data_name 
        self.root_path = args.root + args.data_name

        self.path = ''
        self.ext = ''

        if self.data_name == 'YS':
            self.path = self.root_path + '/MS'
            self.ext = '.dcm'
            self.patients = glob(self.path + '/**')
            self.volume_depth = len(glob(self.patients[0]+'/*.nii'))
            print(self.patients[0])
            
        else:
            self.path = self.root_path + '/training'
            self.ext = '.nhdr'
            self.patients = glob(self.path + '/**')
            self.volume_depth =  args.volume_size
            
        self.slices_by_mode = np.zeros((self.args.n_mode, self.volume_depth, args.volume_size, args.volume_size))
        print(self.slices_by_mode.shape)
        self.data_dir = self.path + '/0/'
    # Bias correction using ANTs scripts
    def atroposN4_norm(self, img, mask, output, dim):
        subprocess.call(scripts_dir+'antsAtroposN4.sh -d '+dim+' -a '+img+' -x '+mask+ \
                        ' -p '+self.data_dir+'priorWarped%d.nii.gz -c 3 -y 2 -y 3 -w 0.25 -o '+output, shell = True)
        res_path = glob(output+'*N4.nii*')
        result = io.imread(res_path[0], plugin='simpleitk').astype(float)
        result = self._normalize(result)
        result /= np.max(result)
        return result

    # Bias correction using ITK
    def n4itk_norm(self, path):
        img=sitk.ReadImage(path)
        img=sitk.Cast(img, sitk.sitkFloat32)
        
        #img_mask = sitk.BinaryThreshold(img, 0, 0)
        img_mask = sitk.ReadImage(path[:-4]+'_mask.PNG')
        
        print('             -> Applyling bias correction...')
        #corrector = sitk.N4BiasFieldCorrectionImageFilter()
        #corrected_img = corrector.Execute(img, img_mask)
        corrected_img = sitk.N4BiasFieldCorrection(img, img_mask, 0.25, [50,50,30,20])
        print('             -> Done.')
        sitk.WriteImage(corrected_img, path.replace(self.ext, '__n.nii'))

    # Zero mean
    def _normalize(self, slice):
        # remove outlier
        if self.data_name != 'YS':
            b, t = np.percentile(slice, (1,99))
            slice = np.clip(slice, 0, t)
        slice[slice<0] = 0
        if np.std(slice) == 0: 
            return slice
        else:
            # zero mean norm
            return (slice - np.mean(slice)) / np.std(slice)

    # Volume to slices
    def volume2slices(self, patient):
        print('         -> Loading scans...')

        mode = []
        # directories to each protocol
        flair, t1, t2, gt = path_glob(self.ext, self.volume_depth, self.data_name, patient)
        
        if self.args.data_name == 'YS':

            mode = [t1, gt]
            for m in range(len(mode)-1):
                for slx in range(len(mode[0])):
                    result = self.atroposN4_norm(mode[m][slx],mode[-1][slx],patient+'/output/','2')
                    result = self._normalize(result)
                    result /= np.max(result)
                    
                    self.slices_by_mode[m][slx] = result
        else:
            mode = [flair[0], t1[0], t2[0], gt[0]]

            if self.args.n_mode < 3:
                mode = [t1[0], gt[0]]
                
            for mode_idx in range(len(mode)):
                result = io.imread(mode[mode_idx], plugin='simpleitk').astype(float)
                if mode_idx < len(mode)-1:
                    result = self._normalize(result)
                    result /= np.max(result)
                
                self.slices_by_mode[mode_idx] = result
    
        print('         -> Done.')

        return True

    # Preprocessing main func.
    def preprocess(self):
        print('\nCreate patches...')

        p_path = self.root_path+'/patch/mode_{}/patch_{}'.format(self.args.n_mode, self.args.patch_size)
        val_str = ''

        create_folder(p_path)

        if self.data_name == 'YS':
            if len(glob(p_path+'/test_ys/0/0/**')) > 10:
                print(p_path)
                print('YS Done.\n')
                return p_path, 0
        else:
            if len(glob(p_path+'/**')) > 1:
                print('MICCAI Done.\n')
                return p_path, 0

        len_patch = 0
        n_val = 1

        for idx, patient in enumerate(self.patients):

            if not self.volume2slices(patient):
                continue
         
            if self.data_name == 'YS':
                self.train_bool = False
                val_str = '/test_ys/{}'.format(idx)
                create_folder(p_path+val_str)
                create_folder(p_path+val_str+'/0')
                
            else:
                if idx > n_val and idx < n_val+3:
                    self.train_bool = False
                    val_str = '/validation/{}'.format(idx)
                    print(' --> test patch : '+ patient)
                else:
                    self.train_bool = True
                    val_str = '/train'
                
                create_folder(p_path+val_str)
                for i in range(self.args.n_class):
                    create_folder(p_path+val_str+'/{}'.format(i))

            # run patch_extraction
            pl = MakePatches(self.args, self.args.n_patch/len(self.patients), self.train_bool)

            if self.data_name == 'YS':
                save_img(idx, self.root_path, self.args.n_mode, self.slices_by_mode, self.volume_depth)
                l_p = pl.create_2Dpatches_YS(self.slices_by_mode, p_path+val_str, idx)
                len_patch += l_p
            else:
                save_img(idx, self.root_path, self.args.n_mode, self.slices_by_mode, self.volume_depth, True) 
                l_p = pl.create_2Dpatches(self.slices_by_mode, p_path+val_str, idx)
                len_patch += l_p
            
            print('-----------------------idx = {} & num of patches = {}'.format(idx, l_p))
            
        print('\n\nnum of all patch = {}'.format(len_patch))

        print('Done.\n')
        return p_path, len_patch

    
