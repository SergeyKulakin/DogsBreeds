import torch

path_model = 'static/model_2.pt'
shape_resize = (224, 224)
dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") # device


# Зададим словарь классов
dog_breeds = {0 : 'ши-тцу',
              1 : 'родезийский риджбек',
              2 : 'бигль',
              3 : 'английский фоксхаунд',
              4 : 'бордер-терьер',
              5 : 'австралийский терьер',
              6 : 'золотистый ретривер',
              7 : 'староанглийская овчарка',
              8 : 'самоед',
              9 : 'динго'}