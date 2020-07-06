import torchvision.models as vision_models
import torch
from torchvision import transforms


def load_model(model_name):
    model_ = {
        "densenet121": vision_models.densenet121,
        "densenet169": vision_models.densenet169,
        "densenet201": vision_models.densenet201,
        "densenet161": vision_models.densenet161
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
