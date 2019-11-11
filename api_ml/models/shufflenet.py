import torchvision.models as vision_models
import torch
from torchvision import transforms


def load_model(model_name):
    model_ = {
        "shufflenetv2_x2.0": vision_models.shufflenet_v2_x2_0,
        "shufflenetv2_x1.5": vision_models.shufflenet_v2_x1_5,
        "shufflenet_v2_x1_0": vision_models.shufflenet_v2_x1_0,
        "shufflenetv2_x0.5": vision_models.shufflenet_v2_x0_5
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
