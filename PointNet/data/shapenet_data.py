import os
import numpy as np
import json

import paddle
from paddle.io import Dataset
from paddle.io import DataLoader

from .point_aug import *

class ShapeNetDataset(Dataset):
    def __init__(
    	self,
    	file_folder,
        npoints=2048, 
        mode='train',  
        use_normals=True,
        transform=None
        ):
        super().__init__()

        assert mode in ['train', 'val']

        self.npoints = npoints
        self.catfile = os.path.join(file_folder, 'synsetoffset2category.txt')
        self.cat = {}
        self.use_normals = use_normals
        self.transform = transform


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]


        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))


        self.meta = {}
        with open(os.path.join(file_folder, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(file_folder, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(file_folder, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])


        for item in self.cat:

            self.meta[item] = []
            dir_point = os.path.join(file_folder, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if mode == 'train':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif mode == 'val':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

        print(f"----- 📦 The ShapeNet size of {mode} data is {len(self.datapath)}")


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(fn[1]).astype(np.float32)

            if not self.use_normals:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]

            seg = data[:, -1].astype(np.int64)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        point_set[:, 0:3] = self.transform(point_set[:, 0:3])

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


def get_train_transforms(point_set):

    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    # point_set[:, 0:3] = random_rotate_point_cloud(point_set[:, 0:3])
    # point_set[:, 0:3] = jitter_point_cloud(point_set[:, 0:3])
    # point_set[:, 0:3] = random_point_dropout(point_set[:, 0:3])
    point_set[:, 0:3] = random_scale_point_cloud(point_set[:, 0:3])
    point_set[:, 0:3] = shift_point_cloud(point_set[:, 0:3])

    return paddle.to_tensor(point_set)


def get_val_transforms(point_set):
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    return paddle.to_tensor(point_set)


def get_dataset(file_folder, data='ShapeNet', npoints=2048, mode='train', use_normals=True):

    assert data in ['ShapeNet']

    if mode == 'train':
        dataset = ShapeNetDataset(file_folder=file_folder, mode=mode, npoints=npoints, transform=get_train_transforms, use_normals=use_normals)
    else:
        dataset = ShapeNetDataset(file_folder=file_folder, mode=mode, npoints=npoints, transform=get_val_transforms, use_normals=use_normals)

    return dataset


def get_dataloader(dataset, batch_size=128, mode='train'):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=(mode == 'train'))
    print(f"----- {mode} batch size is {batch_size}")
    return dataloader


if __name__ == "__main__":
    
    file_folder = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    train_dataset = get_dataset(data='ShapeNet', file_folder=file_folder, mode='train')
    train_dataloader = get_dataloader(train_dataset, batch_size=32, mode='train')
    val_dataset = get_dataset(data='ShapeNet', file_folder=file_folder, mode='val')
    val_dataloader = get_dataloader(val_dataset, batch_size=32, mode='val')

    for point_set, cls, seg in train_dataloader:
        print(point_set.shape) # [B,2048,6]
        print(cls.shape)       # [B,1]
        print(seg.shape)       # [B,2048]
        break
        
        