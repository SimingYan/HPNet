import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import torch
import numpy as np
from torch.utils import data
import h5py
import random
from collections import Counter
from src.augment_utils import rotate_perturbation_point_cloud, jitter_point_cloud, \
    shift_point_cloud, random_scale_point_cloud, rotate_point_cloud

EPS = np.finfo(np.float32).eps

class ABCDataset(data.Dataset):
    def __init__(self, root, filename, opt, skip=1, fold=1):
        
        self.root = root
        self.data_path = open(os.path.join(root, filename), 'r')
        self.opt = opt
        self.augment_routines = [
            rotate_perturbation_point_cloud, jitter_point_cloud,
            shift_point_cloud, random_scale_point_cloud, rotate_point_cloud
        ]
        
        if 'train' in filename:
            self.augment = self.opt.augment
            self.if_normal_noise = self.opt.if_normal_noise
        else:
            self.augment = 0
            self.if_normal_noise = 0

       
        self.data_list = [item.strip() for item in self.data_path.readlines()]
        self.skip = skip
        
        self.data_list = self.data_list[::self.skip]
        self.tru_len = len(self.data_list)
        self.len = self.tru_len * fold
    
    def __getitem__(self, index):

        ret_dict = {}
        index = index % self.tru_len
        
        data_file = os.path.join(self.root, self.data_list[index] + '.h5')

        with h5py.File(data_file, 'r') as hf:
            points = np.array(hf.get("points"))
            labels = np.array(hf.get("labels"))
            normals = np.array(hf.get("normals"))
            primitives = np.array(hf.get("prim"))
            primitive_param = np.array(hf.get("T_param"))
        
        if self.augment:
            points = self.augment_routines[np.random.choice(np.arange(5))](points[None,:,:])[0]

        if self.if_normal_noise:
            noise = normals * np.clip(
                np.random.randn(points.shape[0], 1) * 0.01,
                a_min=-0.01,
                a_max=0.01)
            points = points + noise.astype(np.float32)
      
        ret_dict['gt_pc'] = points
        ret_dict['gt_normal'] = normals
        ret_dict['T_gt'] = primitives.astype(int)
        ret_dict['T_param'] = primitive_param
        
        # set small number primitive as background
        counter = Counter(labels)
        mapper = np.ones([labels.max() + 1]) * -1
        keys = [k for k, v in counter.items() if v > 100]
        if len(keys):
            mapper[keys] = np.arange(len(keys))
        label = mapper[labels]
        ret_dict['I_gt'] = label.astype(int)
        clean_primitives = np.ones_like(primitives) * -1
        valid_mask = label != -1
        clean_primitives[valid_mask] = primitives[valid_mask]
        ret_dict['T_gt'] = clean_primitives.astype(int)

        ret_dict['index'] = self.data_list[index]

        small_idx = label == -1
        full_labels = label
        full_labels[small_idx] = labels[small_idx] + len(keys)
        ret_dict['I_gt_clean'] = full_labels.astype(int)
        return ret_dict

    def __len__(self):
        return self.len

if __name__ == '__main__':

    abc_dataset = ABCDataset(
        root=
        '/home/ysm/project/2021_SIG_Primitive/Primitive_Detection/thirdparty/parsenet-codebase/data/shapes',
        filename='train_data.h5')

    for idx in range(len(abc_dataset)):
        example = abc_dataset[idx]
        import ipdb
        ipdb.set_trace()
