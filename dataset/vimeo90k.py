import random
import torch
import os.path as op
import numpy as np
from cv2 import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor


def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)  # (H W [BGR])
    img = img.astype(np.float32) / 255.
    return img


class Vimeo90KDataset(data.Dataset):
    """Vimeo90K dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super(Vimeo90KDataset, self).__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        self.gt_root = op.join(
            'data/Vimeo-90K/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/Vimeo-90K/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        with open(self.opts_dict['meta_info_fp'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']
        """
            self.is_lmdb = True
        elif self.type_ds == 'val':
            self.io_opts_dict['type'] = 'disk'
            self.is_lmdb = False
        """

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        """
        if self.type_ds == 'train' and ...
        """
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        img_gt_path = f'{key}/im4'
        """
        else:  # disk
            img_gt_path = op.join(
                self.gt_root, clip, seq, 'im4.png'
                )
        """
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W [BGR])

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}'
            """
            else:  # disk
                img_lq_path = op.join(
                    self.lq_root, clip, seq, f'im{neighbor}.png'
                )
            """
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W [BGR])
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        
        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
            )

        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )
        """
        elif self.type_ds == 'val':
            img_lqs.append(img_gt)
            img_results = img_lqs
        """

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.keys)


class VideoTestVimeo90KDataset(data.Dataset):
    """Video test dataset for Vimeo90k-Test dataset.

    For validation data: Disk IO is adopted.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """
    def __init__(self, opts_dict, radius):
        super(VideoTestVimeo90KDataset, self).__init__()
        
        self.opts_dict = opts_dict

        # file IO backend
        self.file_client = FileClient(type='disk')

        # dataset paths
        self.gt_root = op.join(
            'data/Vimeo-90K/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/Vimeo-90K/', 
            self.opts_dict['lq_path']
            )
        
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            }
        
        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]

        # record subfolders
        with open(opts_dict['meta_info_fp'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        
        # record img paths
        for subfolder in subfolders:
            # only 4-th
            gt_path = op.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)

            # all 7 frames
            lq_paths = [
                op.join(self.lq_root, subfolder, f'im{i}.png')
                for i in self.neighbor_list
                ]  # a list
            self.data_info['lq_path'].append(lq_paths)

    def __getitem__(self, index):
        gt_path = self.data_info['gt_path'][index]
        lq_path_list = self.data_info['lq_path'][index]

        # get gt 4-th frame
        img_bytes = self.file_client.get(gt_path)
        img_gt = _bytes2img(img_bytes)  # (H W [BGR])

        # get lq 7 frames
        img_lqs = []
        for lq_path in lq_path_list:
            img_bytes = self.file_client.get(lq_path)
            img_lq = _bytes2img(img_bytes)  # (H W [BGR])
            img_lqs.append(img_lq)

        # to tensor
        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.data_info['gt_path'])
