import api_ml.models as models


def load_model(name_model, weights=None):
    if name_model == "resnet18":
        return models.resnet18.load_model()

    if name_model == "alexnet":
        return models.alexnet.load_model()

    if name_model == "vgg16":
        return models.vgg16.load_model()

    if name_model == "resnet18":
        return models.resnet18.load_model()

    if name_model == "squeezenet1_0":
        return models.squeezenet1_0.load_model()

    if name_model == "wide_resnet101_2":
        return models.wide_resnet101_2.load_model()

    if name_model in ["densenet121", "densenet169", "densenet161", "densenet201"]:
        return models.densenet.load_model(name_model)

    if name_model == "inception_v3":
        return models.inception_v3.load_model()

    if name_model == "googlenet":
        return models.googlenet.load_model()

    if name_model == "shufflenet_v2_x1_0":
        return models.shufflenet_v2_x1_0.load_model()

    if name_model == "mobilenet_v2":
        return models.mobilenet_v2.load_model()

    if name_model == "resnext50_32x4d":
        return models.resnext50_32x4d.load_model()

    if name_model == "wide_resnet50_2":
        return models.wide_resnet50_2.load_model()

    if name_model == "mnasnet1_0":
        return models.mnasnet1_0.load_model()
    
    return False


def batchify(iterable, batch_size=1):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx:min(ndx + batch_size, length)]
