import torch
import torchvision.transforms as T

# Модули
import results as res

path = "F:/Проекты/SBER CV/imagewoof2.tgz"
path_unzip = 'dataset'
path_work = 'dataset/imagewoof2'
shape_resize = (224, 224)
batch_size = 256
torch.manual_seed(17)
def_n_ep = 1 # epochs
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") # device
cr = torch.nn.CrossEntropyLoss() # criterion



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