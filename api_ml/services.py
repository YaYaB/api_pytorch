from api_ml.parameters import models_available


class Services(object):
    def __init__(self):
        self.models_available = models_available
        self.models_online = {}

    def create_service(self, model_infos, model):
        name = model_infos["service_name"]
        type_data = model_infos["type"]
        if name not in self.models_online:
            self.models_online[name] = {
                "model_name": model_infos["model_name"],
                "model": model,
                "type": type_data
            }

            if "batch_size" in model_infos:
                self.models_online[name]['batch_size'] = model_infos['batch_size']
            if "top_k" in model_infos:
                self.models_online[name]['top_k'] = model_infos['top_k']
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
