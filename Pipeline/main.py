#from sklearn.metrics import accuracy_score
from torch import nn, optim
import torch

import os
import pickle

# Модули
import config as conf
import results as res
import dataloader as dl
import normalize_calc as norm
import models as md
import train

# Pipeline step 1 *unzip*
if not os.path.isdir(conf.path_unzip):
    dl.dataset_load(path=conf.path, path_unzip=conf.path_unzip) # with default args

# Pipeline 2 *mean, std train*
if not os.path.isfile(f'{conf.path_unzip}/normalize_dict.pickle'):
    res.normalize_dict['train']['mean'], res.normalize_dict['train']['std'] = norm.mean_std('train')
    res.normalize_dict['val']['mean'], res.normalize_dict['val']['std'] = norm.mean_std('val')
    # Сохраним словарик
    with open(f'{conf.path_unzip}/normalize_dict.pickle', 'wb') as f:
        pickle.dump(res.normalize_dict, f)
# Если файл уже есть
elif os.path.isfile(f'{conf.path_unzip}/normalize_dict.pickle'):
    with open(f'{conf.path_unzip}/normalize_dict.pickle', 'rb') as f:
        res.normalize_dict = pickle.load(f)
#print(res.normalize_dict)

# Pipeline 3 *get dataloaders*
train_dataloader, val_dataloader = dl.get_dataloader(conf.path_work, conf.batch_size, conf.shape_resize) # with default args
#print(train_dataloader.dataset)


# Запуск экспериментов
for num_experiment in range(1, 4): # всего 3 эксперимента
    # Pipeline 4
    if not os.path.isdir(f'{conf.path_unzip}/{num_experiment}'):
        os.mkdir(f'{conf.path_unzip}/{num_experiment}')
    model = md.chose_model(f'{num_experiment}')
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.1)
    train.train(model, optimizer, train_dataloader, val_dataloader, criterion=conf.cr, n_epochs=conf.def_n_ep, device=conf.dev)
    # Сохраним словарик результатов
    with open(f'{conf.path_unzip}/{num_experiment}/history_dict_{num_experiment}.pickle', 'wb') as f:
        pickle.dump(res.history_dict, f)
    # Сохраним модель
    torch.save(model, f'{conf.path_unzip}/{num_experiment}/model_{num_experiment}.pt')
    # запишем результаты эксперимента
    res.results[num_experiment] = res.best_score

# Итоговые результаты
with open(f'{conf.path_unzip}/results.pickle', 'wb') as f:
    pickle.dump(conf.results, f)


