import tarfile
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T

import config as conf
import results as res

def dataset_load(path=conf.path, path_unzip=conf.path_unzip):
    '''INPUT
            -> path: need path to dataset archive
            -> path_unzip: need path to unzip dir

        OUTPUT
            -> unzip dataset to unzip dir'''

    with tarfile.open(path, "r") as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path_unzip)
    return


def get_dataloader(path_work=conf.path_work, batch_size=conf.batch_size, shape_resize=conf.shape_resize):
    '''INPUT
            -> train_transform: params transformation train data
            -> val_transform: params transformation val data
            -> path_work: path to work directory
            -> batch_size: number of images in batch
        OUTPUT
            -> train_dataloader, val_dataloader'''

    # DataSet
    train_transform = T.Compose([
        T.Resize(shape_resize),
        T.ToTensor(),
        T.Normalize(
            mean=res.normalize_dict['train']['mean'],
            std=res.normalize_dict['train']['std'])])

    val_transform = T.Compose([
        T.Resize(shape_resize),
        T.ToTensor(),
        T.Normalize(
            mean=res.normalize_dict['val']['mean'],
            std=res.normalize_dict['val']['std'])])

    train_dataset = ImageFolder(
        f'{path_work}/train',
        transform=train_transform)

    val_dataset = ImageFolder(
        f'{path_work}/val',
        transform=val_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader

