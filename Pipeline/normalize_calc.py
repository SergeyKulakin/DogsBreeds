import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Модули
import config as conf

# функция для подсчета среднего и стандартного отклонения в выборке
def mean_std(loader=None):

    '''INPUT
            -> dataloader: str - values: 'train',
                                        'val'
      OUTPUT - > mean, standard deviation'''
    # преобразуем датасет в тензоры
    train_transform = T.Compose([T.Resize(conf.shape_resize), T.ToTensor()])
    train_dataset = ImageFolder(
        f'{conf.path_work}/train',
        transform=train_transform)

    val_dataset = ImageFolder(
        f'{conf.path_work}/val',
        transform=train_transform)

    # приравниваем размер batch к размеру выборки
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0)

    if loader == None:
        print('You should select "train" or "val"')
        return

    # shape of images = [b,c,w,h]
    if loader == 'train':

        images, lebels = next(iter(train_dataloader))

        mean, std = images.mean([0,2,3]), images.std([0,2,3])
        return mean.tolist(), std.tolist()

    elif loader == 'val':
        images, lebels = next(iter(val_dataloader))

        mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
        return mean.tolist(), std.tolist()

