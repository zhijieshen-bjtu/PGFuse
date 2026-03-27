from __future__ import print_function
import os
import cv2
import numpy as np
import random

import torch
from torch.utils import data
from torchvision import transforms

from .util import Equirec2Cube


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
          rgb_depth_list.append(line.strip().split(" "))
    return rgb_depth_list


class Structure3D(data.Dataset):
    """The Matterport3D Dataset"""

    def __init__(self, root_dir, list_file, height=512, width=1024, disable_color_augmentation=False,
                 disable_LR_filp_augmentation=False, disable_yaw_rotation_augmentation=False, is_training=False):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
            disable_color_augmentation, disable_LR_filp_augmentation,
            disable_yaw_rotation_augmentation: augmentation options.
            is_training (bool): True if the dataset is the training set.
        """
        self.root_dir = root_dir
        self.rgb_depth_list = read_list(list_file)

        self.w = width
        self.h = height

        self.max_depth_meters = 10.0

        self.color_augmentation = not disable_color_augmentation
        self.LR_filp_augmentation = not disable_LR_filp_augmentation
        self.yaw_rotation_augmentation = not disable_yaw_rotation_augmentation

        self.is_training = is_training

        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)

        if self.color_augmentation:
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                self.color_aug= transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1
                self.color_aug = transforms.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #self.normalizegt = transforms.Normalize()

    def __len__(self):
        return len(self.rgb_depth_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_depth = gt_depth.astype(np.float)#/2048
        #gt_depth[gt_depth > 3] = 3
        
        if self.is_training and self.yaw_rotation_augmentation:
            # random yaw rotation
            roll_idx = random.randint(0, self.w)
            rgb = np.roll(rgb, roll_idx, 1)
            gt_depth = np.roll(gt_depth, roll_idx, 1)

        if self.is_training and self.LR_filp_augmentation and random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            gt_depth = cv2.flip(gt_depth, 1)

        if self.is_training and self.color_augmentation and random.random() > 0.5:
            aug_rgb = np.asarray(self.color_aug(transforms.ToPILImage()(rgb)))
        else:
            aug_rgb = rgb

        #cube_rgb, cube_gt_depth = self.e2c.run(rgb, gt_depth[..., np.newaxis])
        '''
        cube_rgb = self.e2c.run(rgb)
        cube_aug_rgb = self.e2c.run(aug_rgb)

        rgb = self.to_tensor(rgb.copy())
        cube_rgb = self.to_tensor(cube_rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)

        inputs["cube_rgb"] = cube_rgb
        inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)


        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["gt_depth"]))
        '''
                                
        shift_amount = self.w//8
        rrgb = np.concatenate((rgb[:, -shift_amount:], rgb[:, :-shift_amount]), axis=1)
        #rrgb = np.concatenate((rrgb[:, shift_amount:], rrgb[:, :shift_amount]), axis=1)
        aug_rrgb = np.concatenate((aug_rgb[:, -shift_amount:], aug_rgb[:, :-shift_amount]), axis=1)
        #aug_rrgb = np.concatenate((aug_rrgb[:, shift_amount:], aug_rrgb[:, :shift_amount]), axis=1)                         
        
        cube_rgb = self.e2c.run(rgb)
        cube_aug_rgb = self.e2c.run(aug_rgb)
        
        rcube_rgb = self.e2c.run(rrgb)
        rcube_aug_rgb = self.e2c.run(aug_rrgb)
        
        

        rgb = self.to_tensor(rgb.copy())
        rrgb = self.to_tensor(rrgb.copy())
        cube_rgb = self.to_tensor(cube_rgb.copy())
        rcube_rgb = self.to_tensor(rcube_rgb.copy())
        aug_rgb = self.to_tensor(aug_rgb.copy())
        cube_aug_rgb = self.to_tensor(cube_aug_rgb.copy())
        rcube_aug_rgb = self.to_tensor(rcube_aug_rgb.copy())

        inputs["rgb"] = rgb
        inputs["rrgb"] = rrgb
        inputs["normalized_rgb"] = self.normalize(aug_rgb)
        #inputs["normalized_rrgb"] = self.normalize(aug_rrgb)

        inputs["cube_rgb"] = cube_rgb
        inputs["normalized_cube_rgb"] = self.normalize(cube_aug_rgb)
        
        inputs["rcube_rgb"] = rcube_rgb
        inputs["rnormalized_cube_rgb"] = self.normalize(rcube_aug_rgb)


        gt_depth = torch.from_numpy(np.expand_dims(gt_depth, axis=0)) 
        #print(gt_depth.min())
        #print(gt_depth.max())
        #inputs["val_mask"] = ((gt_depth > 0) & ~torch.isnan(gt_depth))       
        gt_depth = (gt_depth-gt_depth.min())/(gt_depth.max() - gt_depth.min())
        inputs["gt_depth"] = gt_depth
        inputs["gt_depth"] = inputs["gt_depth"] * self.max_depth_meters
        #print(inputs["gt_depth"].max()-inputs["gt_depth"].min())
        inputs["val_mask"] = (inputs["gt_depth"] > 0)
        inputs["gt_depth"] = inputs["gt_depth"] * inputs["val_mask"]
        
                                
        #inputs["gt_depth"] = inputs["gt_depth"] * inputs["val_mask"].float()#*self.max_depth_meters
        
        #print(inputs["gt_depth"].max())

        """
        cube_gt_depth = torch.from_numpy(np.expand_dims(cube_gt_depth[..., 0], axis=0))
        inputs["cube_gt_depth"] = cube_gt_depth
        inputs["cube_val_mask"] = ((cube_gt_depth > 0) & (cube_gt_depth <= self.max_depth_meters)
                                   & ~torch.isnan(cube_gt_depth))
        """

        return inputs



