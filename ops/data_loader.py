import numpy as np

import torchvision.datasets as dset
from ops.folder import ImageFolder
import torchvision.transforms as transforms

import torch
from torch.autograd import Variable
import torch.utils.data as data

from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io


class Create_Batch(object):

    def __init__(self, batch_size, patch_size, n_mode, img_dir):
        self.bs = batch_size
        self.ps = patch_size
        self.n_mode = n_mode
        self.img_dir = img_dir

    def flip(self, img):
        patch_mode = np.zeros([self.ps*2, self.ps, self.n_mode])
  
        w,h = img.size
        mid = int(w/2)
        ps_mid = int(self.ps/2)
        for m in range(self.n_mode):
            im = np.asarray(img)
            im = im[:,:,0]
            patch = im[w*m:w*(m+1), :]

            s = patch[mid-ps_mid:mid+ps_mid, mid-ps_mid:mid+ps_mid]
            b = downscale_local_mean(patch, (2,2))
  
            patch_mode[:,:,m] = np.concatenate([s, b],axis=0)
        
        return patch_mode

    def test_flip(self, img):
        patch_mode = np.zeros([self.n_mode, self.ps*2, self.ps])
                  
        h,w = img.shape
        mid = int(w/2)
        ps_mid = int(self.ps/2)
        for m in range(self.n_mode):
            im = np.asarray(img)
            patch = im[w*m:w*(m+1), :]

            s = patch[mid-ps_mid:mid+ps_mid, mid-ps_mid:mid+ps_mid]
            b = downscale_local_mean(patch, (2,2))

            patch_mode[m,:,:] = np.concatenate([s, b],axis=0)
        return patch_mode

    def db_load(self):
        img_data = ImageFolder(root=self.img_dir, transform = transforms.Compose([
                                                    transforms.Lambda(lambda x: self.flip(x)),
                                                    transforms.ToTensor()
                                                    ]))
        img_batch = data.DataLoader(img_data, batch_size=self.bs, shuffle=True, num_workers=2, drop_last = True)
        
        return img_batch
