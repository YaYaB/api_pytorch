from api_ml.parameters import model_to_repository


class Services(object):
    def __init__(self):
        self.models_available = model_to_repository
        self.models_online = {}

    def create_service(self, name, model):
        if name not in self.models_online:
            self.models_online[name] = model
            return True
        else:
            # Service already existing
            return False

    def delete_service(self, name):
        if name not in self.models_online:
            # Service does not exist
            return False
        else:
            del self.models_online[name]
            return True
