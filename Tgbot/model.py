import torch
from torch.autograd import Variable
import torchvision.transforms as T

import config as conf

# Загрузим лучшую модель
model = torch.load(conf.path_model)
model.eval()


test_transforms = T.Compose([T.Resize(conf.shape_resize), T.ToTensor()])


def predict_image(image):
    '''INPUT
            -> image: PIL image
        OUTPUT
            -> predict idx_label, predict label
    '''

    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    inpt = Variable(image_tensor)
    inpt = inpt.to(conf.dev)
    output = model(inpt)
    _, predicted = torch.max(output, 1)
    label = conf.dog_breeds[predicted.item()]
    return label