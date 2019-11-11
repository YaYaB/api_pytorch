import falcon

from api_ml import models_utils
from .services import Services


def create_app():
    api = falcon.API()
    services = Services()
    api.add_route("/create", models_utils.LoadModel(services))
    api.add_route("/delete", models_utils.DeleteModel(services))
    api.add_route("/predict", models_utils.Predict(services))
    api.add_route("/available", models_utils.ListAvailableModels(services))
    api.add_route("/online", models_utils.ListOnlineModels(services))

    return api


def get_app():
    # storage_path = os.environ.get("LOOK_STORAGE_PATH", ".")
    return create_app()
