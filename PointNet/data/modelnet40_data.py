import os
import numpy as np

import paddle
from paddle.io import Dataset
from paddle.io import DataLoader

from .point_aug import *

class ModelNet40Dataset(Dataset):
    def __init__(
        self,
        file_folder,
        npoints=1024,
        use_farthest_point_sample=False,
        use_normals=True,
        mode="train",
        transform=None,
    ):
        super().__init__()
        assert mode in ["train", "val"]
        
        self.transform = transform
        self.npoints = npoints
        self.use_farthest_point_sample = use_farthest_point_sample
        self.use_normals = use_normals

        self.catfile = os.path.join(file_folder, "modelnet40_shape_names.txt")
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        shape_ids["train"] = [
            line.rstrip()
            for line in open(os.path.join(file_folder, "modelnet40_train.txt"))
        ]
        shape_ids["val"] = [
            line.rstrip()
            for line in open(os.path.join(file_folder, "modelnet40_test.txt"))
        ]

        
        shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids[mode]]
        self.datapath = [
            (
                shape_names[i],
                os.path.join(file_folder, shape_names[i], shape_ids[mode][i]) + ".txt",
            )
            for i in range(len(shape_ids[mode]))
        ]
        print(f"----- 📦 The ModelNet40 size of {mode} data is {len(self.datapath)}")


    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):

        fn = self.datapath[index]
        clas = self.classes[self.datapath[index][0]]
        label = np.array([clas]).astype(np.int64)
        point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)

        if self.use_farthest_point_sample:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0 : self.npoints, :]        

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        point_set[:, 0:3] = self.transform(point_set[:, 0:3])
        
        return point_set, label



def get_train_transforms(point_set):

    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    # point_set[:, 0:3] = random_rotate_point_cloud(point_set[:, 0:3])
    point_set[:, 0:3] = jitter_point_cloud(point_set[:, 0:3])
    # point_set[:, 0:3] = random_point_dropout(point_set[:, 0:3])
    point_set[:, 0:3] = random_scale_point_cloud(point_set[:, 0:3])
    point_set[:, 0:3] = shift_point_cloud(point_set[:, 0:3])

    return paddle.to_tensor(point_set)



def get_val_transforms(point_set):
    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    
    return paddle.to_tensor(point_set)


def get_dataset(file_folder, data='ModelNet40', mode='train', npoints=1024, use_normals=True):

    assert data in ['ModelNet40']

    
    if mode == 'train':
        dataset = ModelNet40Dataset(file_folder=file_folder, mode=mode, npoints=npoints, transform=get_train_transforms, use_normals=True)
    else:
        dataset = ModelNet40Dataset(file_folder=file_folder, mode=mode, npoints=npoints, transform=get_val_transforms, use_normals=True)

    return dataset


def get_dataloader(dataset, batch_size=128, mode='train'):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=(mode == 'train'))
    print(f"----- {mode} batch size is {batch_size}")
    return dataloader 


if __name__ == "__main__":
    
    file_folder = 'modelnet40_normal_resampled'
    train_dataset = get_dataset(data='ModelNet40', file_folder=file_folder, mode='train')
    train_dataloader = get_dataloader(train_dataset, batch_size=128, mode='train')
    val_dataset = get_dataset(data='ModelNet40', file_folder=file_folder, mode='val')
    val_dataloader = get_dataloader(val_dataset, batch_size=128, mode='val')

    for point_set, label in train_dataloader:
        print(point_set.shape)   # [B,1024, 6]
        print(label.shape)   # [B,1]
        break