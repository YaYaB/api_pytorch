import torchvision.models as vision_models
import torch
from torchvision import transforms


def load_model(model_name):
    model_ = {
        "mnasnet0_5": vision_models.mnasnet0_5,
        "mnasnet0_75": vision_models.mnasnet0_75,
        "mnasnet1_0": vision_models.mnasnet1_0,
        "mnasnet1_3": vision_models.mnasnet1_3
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
