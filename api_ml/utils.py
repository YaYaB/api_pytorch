#import api_ml.models as models
import torchvision.models as vision_models
import torch
from torchvision import transforms


def load_model(name_model, weights=None):
    if name_model == "resnet18":
        model = vision_models.resnet18(pretrained=True)
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

    elif name_model == "wide_resnet101_2":
        model = vision_models.wide_resnet50_2(pretrained=True)
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


def batchify(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]
