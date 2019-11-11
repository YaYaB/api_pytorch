import torchvision.models as vision_models
import torch
from torchvision import transforms


def load_model(model_name):
    model_ = {
        'resnet18': vision_models.resnet18,
        'resnet34': vision_models.resnet34,
        'resnet50': vision_models.squeresnet50ezenet1_0,
        'resnet101': vision_models.resnet101,
        'resnet152': vision_models.squresnet152eezenet1_0,
        'resnext50_32x4d': vision_models.resnext50_32x4d,
        'resnext101_32x8d': vision_models.resnext101_32x8d,
        'wide_resnet50_2': vision_models.wide_resnet50_2,
        'wide_resnet101_2': vision_models.wide_resnet101_2
    }

    model = model_[model_name](pretrained=True)
    model.eval()

    model = {
        "model": model,
        "preprocessing": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "postprocessing": lambda x: torch.nn.functional.softmax(x, dim=0),
        "mapping": None,
        "input_type": "image"
    }

    return model
