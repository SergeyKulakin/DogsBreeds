from torch import nn
import torchvision.models as models

def chose_model(variant=None):
    '''INPUT
            -> variant: "1"-resnet18, "2"-densenet161, "3"-MobileNetV2', "4"-densenet201
        OUTPUT
            -> model with changed fc layer
        '''
    if variant is None:
        print('You need chose a variants: "1"-resnet18, "2"-densenet161, "3"-MobileNetV2, "4"-densenet201')
        return
    if variant == '1':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=512, out_features=10)
        return model

    elif variant == '2':
        model = models.densenet161(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(in_features=2208, out_features=10, bias=True)
        return model

    elif variant == '3':
        model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(in_features=1280, out_features=10, bias=True)
        return model

    elif variant == '4':
        model = models.densenet201(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(in_features=1920, out_features=10, bias=True)
        return model

    else:
        print('You need chose a variants: "1"-resnet18, "2"-densenet161, "3"-MobileNetV2, "4"-densenet201')
        return