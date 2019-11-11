import torchvision.models as vision_models
import torch
from torchvision import transforms


def load_model(model_name):
    model_ = {
        "vgg11": vision_models.vgg11,
        "vgg13": vision_models.vgg13,
        "vgg16": vision_models.vgg16,
        "vgg19": vision_models.vgg19,
        "vgg11_bn": vision_models.vgg11_bn,
        "vgg13_bn": vision_models.vgg13_bn,
        "vgg16_bn": vision_models.vgg16_bn,
        "vgg19_bn": vision_models.vgg19_bn
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
