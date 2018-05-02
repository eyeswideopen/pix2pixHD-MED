### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from .med_dataset import MedDataset, get_params, get_transform, normalize
from .med_image_folder import make_dataset, make_numpy_dataset
from PIL import Image
import torch
import numpy as np

class MedAlignedDataset(MedDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_numpy_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            raise RuntimeError("instance maps not supported with medical image data!!")                           


        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            raise RuntimeError("feature loading not supported with medical image data!!")                           


        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0


        ### input B (real images)
        if self.opt.isTrain:
            """
            Numpy array is loaded fron .npy file at B_path.
            Array is then converted from 1 channel 16 bit greyscale data to
            RGB with shape of C x H x W to be compatible with nvidias 3 channel approach
            """
            B_path = self.B_paths[index]   
            numpyArray = np.load(B_path)    #uint16 array 
            
            threeChannelArray = np.array([numpyArray, numpyArray, numpyArray]).astype(np.int32) #3dim channel int32 array


            B_tensor = torch.from_numpy(threeChannelArray).float().div(np.iinfo(np.uint16).max) #convert to float32 tensor and divide by 65535 (uint16 max value)

        ### instance maps        
        if not self.opt.no_instance:
            raise RuntimeError("Instance-maps not supported for medical image data!")                           

        return {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 'feat': feat_tensor, 'path': A_path}
                      

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'MedAlignedDataset'