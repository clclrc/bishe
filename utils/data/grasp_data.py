import numpy as np

import torch
import torch.utils.data

import random

def simulate_occlusion(
    img, 
    occlusion_prob=1, 
    min_ratio=0.15, 
    max_ratio=0.25, 
    min_num=2, 
    max_num=3
):
    """
    在图像上随机添加多个较小的遮挡块（黑色矩形）。
    :param img: numpy.ndarray, (H, W) 或 (H, W, C)，灰度或彩色图像
    :param occlusion_prob: 添加遮挡的概率，如果没达到这个概率则不添加任何遮挡
    :param min_ratio: 单个遮挡块相对于图像宽/高的最小比例
    :param max_ratio: 单个遮挡块相对于图像宽/高的最大比例
    :param min_num: 最少添加多少个遮挡块
    :param max_num: 最多添加多少个遮挡块
    :return: 添加遮挡后的图像
    """
    if np.random.rand() < occlusion_prob:
        h, w = img.shape[:2]
        # 在 [min_num, max_num] 范围内随机选多少个遮挡块
        num_occ = np.random.randint(min_num, max_num + 1)

        for _ in range(num_occ):
            # 随机生成遮挡块的宽度和高度
            occ_w = np.random.randint(int(w * min_ratio), int(w * max_ratio) + 1)
            occ_h = np.random.randint(int(h * min_ratio), int(h * max_ratio) + 1)

            # 随机选择遮挡块在图像中的位置
            x1 = np.random.randint(0, w - occ_w + 1)
            y1 = np.random.randint(0, h - occ_h + 1)
            x2 = x1 + occ_w
            y2 = y1 + occ_h

            # 将区域置为 0（黑色），适用于灰度或彩色图像
            if len(img.shape) == 2:  # 灰度图
                img[y1:y2, x1:x2] = 0
            else:  # 彩色图
                img[y1:y2, x1:x2, :] = 0
    return img

class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, output_size=300, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        if np.random.rand() < 0.5:  # 50%概率遮挡
            depth_img = simulate_occlusion(depth_img)
            if self.include_rgb:
                rgb_img = simulate_occlusion(rgb_img)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, 150.0)/150.0

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)
