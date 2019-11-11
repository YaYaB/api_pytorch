import api_ml.models as models


def load_model(name_model, weights=None):
    if name_model == "resnet18":
        return models.resnet18.load_model()

    if name_model == "alexnet":
        return models.alexnet.load_model()

    if name_model in ["vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]:
        return models.vgg.load_model(name_model)

    if name_model in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2"]:
        return models.resnet.load_model(name_model)

    if name_model in ["squeezenet1_0", "squeezenet1_1"]:
        return models.squeezenet.load_model(name_model)

    if name_model in ["densenet121", "densenet169", "densenet161", "densenet201"]:
        return models.densenet.load_model(name_model)

    if name_model == "inception_v3":
        return models.inception_v3.load_model()

    if name_model == "googlenet":
        return models.googlenet.load_model()

    if name_model in ["shufflenetv2_x0.5", "shufflenet_v2_x1_0", "shufflenetv2_x1.5", "shufflenetv2_x2.0"]:
        return models.shufflenet.load_model(name_model)

    if name_model == "mobilenet_v2":
        return models.mobilenet_v2.load_model()

    if name_model in ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]:
        return models.mnasnet.load_model(name_model)

    return False


def batchify(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]
