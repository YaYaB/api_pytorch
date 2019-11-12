from io import BytesIO
from PIL import Image
import requests

import falcon
import json

from api_ml.models.utils import load_model, batchify
from api_ml.utils import check_json
import torch


# Create class Origin that raises HTTPMethodNotAllowed
class Origin(object):
    def __init__(self):
        self.methods_allowed = []

    def on_get(self, req, resp):
        raise falcon.HTTPMethodNotAllowed(self.methods_allowed, description="GET method is not allowed. Methods allowed are {}".format(self.methods_allowed))

    def on_delete(self, req, resp):
        raise falcon.HTTPMethodNotAllowed(self.methods_allowed, description="DELETE method is not allowed. Methods allowed are {}".format(self.methods_allowed))

    def on_put(self, req, resp):
        raise falcon.HTTPMethodNotAllowed(self.methods_allowed, description="PUT method is not allowed. Methods allowed are {}".format(self.methods_allowed))

    def on_post(self, req, resp):
        raise falcon.HTTPMethodNotAllowed(self.methods_allowed, description="POST method is not allowed. Methods allowed are {}".format(self.methods_allowed))

    def on_options(self, req, resp):
        raise falcon.HTTPMethodNotAllowed(self.methods_allowed, description="OPTIONS method is not allowed. Methods allowed are {}".format(self.methods_allowed))


class ListAvailableModels(Origin):
    def __init__(self, services):
        Origin.__init__(self)
        self.services = services
        self.methods_allowed = ["GET"]

    def validate_json_input(self, req):
        input_read = req.bounded_stream.read()
        if input_read == b'':
            return

        json_input = json.loads(input_read)
        if json_input != {}:
            raise falcon.HTTPBadRequest('Bad request', "Json must be empty")

        return

    def on_get(self, req, resp):
        _ = self.validate_json_input(req)
        resp.body = json.dumps({"title": "success", "description": self.services.models_available}, ensure_ascii=False)
        resp.content_type = falcon.MEDIA_JSON
        resp.status = falcon.HTTP_200


class ListOnlineModels(Origin):
    def __init__(self, services):
        Origin.__init__(self)
        self.services = services
        self.methods_allowed = ["GET"]

    def validate_json_input(self, req):
        input_read = req.bounded_stream.read()
        if input_read == b'':
            return

        json_input = json.loads(input_read)
        if json_input != {}:
            raise falcon.HTTPBadRequest('Bad request', "Json must be empty")

        return

    def on_get(self, req, resp):
        _ = self.validate_json_input(req)
        online_models = [{x: {y: self.services.models_online[x][y] for y in self.services.models_online[x] if y != "model"}} for x in self.services.models_online]
        resp.body = json.dumps({"title": "success", "description": online_models if online_models != {} else "no models are online"}, ensure_ascii=False)
        resp.content_type = falcon.MEDIA_JSON
        resp.status = falcon.HTTP_200


class LoadModel(Origin):
    def __init__(self, services):
        Origin.__init__(self)
        self.services = services
        self.methods_allowed = ["PUT"]

    def validate_json_input(self, req):
        call_params = {
            "top_k": {
                "name": "top_k",
                "type": int,
                "mandatory": False,
                "specific_check": {"superior": lambda x: x == -1 or x > 0},
                "messages": {"none": "Please specify the top_k results wanted.", "type": "top_k must be an integer larger than 1 (or -1 to get all results)."}
            },
            "batch_size": {
                "name": "batch_size",
                "type": int,
                "mandatory": False,
                "specific_check": {"superior": lambda x: x > 0},
                "messages": {"none": "Please specify the batch_size to used.", "type": "batch_size must be an integer larger than 1."}
            },
            "service_name": {
                "name": "service_name",
                "type": str,
                "mandatory": True,
                "messages": {"none": "Please specify a service_name.", "type": "service_name must be a string."}
            },
            "model_name": {
                "name": "model_name",
                "type": str,
                "mandatory": True,
                "messages": {"none": "Please specify a model_name.", "type": "model_name must be a string."}
            },
            "type": {
                "name": "type",
                "type": str,
                "mandatory": True,
                "enum": ["image", "text"],
                "messages": {"none": "Please specify a type.", "type": "type must be an string between the following ('text', 'image')."}
            }
        }

        try:
            json_input = json.load(req.bounded_stream)
        except json.decoder.JSONDecodeError:
            raise falcon.HTTPBadRequest('Bad request', "Json seems malformed")

        success, message = check_json(json_input, call_params)
        if not success:
            raise falcon.HTTPBadRequest('Bad request', message)

        fields_needed = set(call_params.keys())
        if not set(json_input.keys()).issubset(fields_needed):
            msg = "{} are the only possible inputs.".format(fields_needed)
            raise falcon.HTTPBadRequest('Bad request', msg)

        return json_input

    def on_put(self, req, resp):
        json_input = self.validate_json_input(req)
        model_name = json_input["model_name"]
        service_name = json_input["service_name"]

        # Load model
        model = load_model(model_name)
        if model is False:
            raise falcon.HTTPNotFound(description="model {} is not an available model in the API".format(model_name))

        # Check if the service exists
        success = self.services.create_service(json_input, model)
        if not success:
            raise falcon.HTTPConflict("Conflict", "The service '{}' already exists".format(service_name))

        resp.body = json.dumps({"title": "sucess", "description": "service '{}' sucessfully created".format(service_name)}, ensure_ascii=False)
        resp.status = falcon.HTTP_201


class DeleteModel(Origin):
    def __init__(self, services):
        Origin.__init__(self)
        self.services = services
        self.methods_allowed = ["DELETE"]

    def validate_json_input(self, req):
        call_params = {
            "service_name": {
                "name": "service_name",
                "type": str,
                "mandatory": True,
                "messages": {"none": "Please specify a service_name to delete.", "type": "service_name must be a string."}
            }
        }

        try:
            json_input = json.load(req.bounded_stream)
        except json.decoder.JSONDecodeError:
            raise falcon.HTTPBadRequest('Bad request', "Json seems malformed")

        success, message = check_json(json_input, call_params)
        if not success:
            raise falcon.HTTPBadRequest('Bad request', message)

        fields_needed = set(call_params.keys())
        if not set(json_input.keys()).issubset(fields_needed):
            msg = "{} are the only possible inputs.".format(fields_needed)
            raise falcon.HTTPBadRequest('Bad request', msg)

        return json_input

    def on_delete(self, req, resp):
        json_input = self.validate_json_input(req)
        service_name = json_input["service_name"]

        # Check if service exists
        success = self.services.delete_service(service_name)
        if not success:
            raise falcon.HTTPNotFound(description="Service does not seem to exist")

        resp.body = json.dumps({"title": "sucess", "description": "service '{}' sucessfully deleted".format(service_name)})
        resp.status = falcon.HTTP_201


class Predict(Origin):
    def __init__(self, services):
        Origin.__init__(self)
        self.services = services
        self.methods_allowed = ["POST"]

    def validate_json_input(self, req):
        call_params = {
            "service_name": {
                "name": "service_name",
                "type": str,
                "mandatory": True,
                "messages": {"none": "Please specify a service_name.", "type": "service_name must be a string."}
            },
            "data": {
                "name": "data",
                "type": list,
                "mandatory": True,
                "specific_check": {"array_strings": lambda x: all([isinstance(y, str) for y in x])},
                "messages": {"none": "Please specify the field data.", "type": "data must be an array of strings."}
            }
        }

        try:
            json_input = json.load(req.bounded_stream)
        except json.decoder.JSONDecodeError:
            raise falcon.HTTPBadRequest('Bad request', "Json seems malformed")

        success, message = check_json(json_input, call_params)
        if not success:
            raise falcon.HTTPBadRequest('Bad request', message)

        fields_needed = set(call_params.keys())
        if not set(json_input.keys()).issubset(fields_needed):
            msg = "{} are the only possible inputs.".format(fields_needed)
            raise falcon.HTTPBadRequest('Bad request', msg)

        return json_input

    def predict(self, model, input, top_k=-1):
        # create mini batch
        input_batch = tuple(model['preprocessing'](x).unsqueeze(0) for x in input)
        input_batch = torch.cat(input_batch, 0)

        # Make prediction
        output = model['model'](input_batch)

        # Make postprocessing
        predictions = [model['postprocessing'](output[x,:]) for x in range(output.shape[0])]

        # Sort prediction and get top wanted
        res = []
        if top_k == -1:
            top_k = predictions[0].shape[0]
        for pred in predictions:
            topk_val, topk_indices = torch.topk(pred, top_k)
            if model["mapping"] is None:
                res.append({x.item(): y.item() for x, y in zip(topk_indices, topk_val)})
            else:
                res.append({
                    model["mapping"][x.item()]: y.item() for x, y in zip(topk_indices, topk_val)
                })

        return res

    def on_post(self, req, resp):
        json_input = self.validate_json_input(req)
        # Parse json
        service_name = json_input["service_name"]
        urls_image = json_input["data"]

        # Check if service exists
        if service_name not in self.services.models_online:
            raise falcon.HTTPNotFound(description="Service does not seem to exist")
        else:
            model = self.services.models_online[service_name]['model']
            batch_size = self.services.models_online[service_name].get("batch_size", 1)
            top_k = self.services.models_online[service_name].get("top_k", -1)
            predictions = []
            for batch in batchify(urls_image, batch_size):
                i = 0
                id_pred_to_res = {}
                imgs = []
                res = {}
                for j, url_image in enumerate(batch):
                    res[j] = {"name": url_image, "predictions": 'Input is not correct'}
                    try:
                        # Local url
                        if url_image[0] == '/':
                            img = Image.open(url_image)
                        # External url
                        else:
                            response = requests.get(url_image)
                            img = Image.open(BytesIO(response.content))

                        imgs.append(img)
                        id_pred_to_res[i] = j
                        i += 1
                    except Exception:
                        # TODO add logging
                        pass

                # Make predictions
                if len(imgs) > 0:
                    prediction = self.predict(model, imgs, top_k=top_k)

                    # Format output
                    for j, pred in enumerate(prediction):
                        res[id_pred_to_res[j]]['predictions'] = pred

                predictions.append(res)

            resp.body = json.dumps(predictions, ensure_ascii=False)
            resp.status = falcon.HTTP_201
