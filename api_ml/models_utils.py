from io import BytesIO
from PIL import Image
import requests

import falcon
import json

from api_ml.utils import load_model, batchify
import torch


ALLOWED_IMAGE_TYPES = (
    'image/gif',
    'image/jpeg',
    'image/png',
)


def validate_image_type(req, resp, resource, params):
    if req.content_type not in ALLOWED_IMAGE_TYPES:
        msg = 'Image type not allowed. Must be PNG, JPEG, or GIF'
        raise falcon.HTTPBadRequest('Bad request', msg)


def extract_project_id(req, resp, resource, params):
    """Adds `project_id` to the list of params for all responders.

    Meant to be used as a `before` hook.
    """
    params['project_id'] = req.get_header('X-PROJECT-ID')


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

    def on_get(self, req, resp):
        resp.body = json.dumps(self.services.models_available, ensure_ascii=False)
        resp.content_type = falcon.MEDIA_JSON

        resp.status = falcon.HTTP_200


class ListOnlineModels(Origin):
    def __init__(self, services):
        Origin.__init__(self)
        self.services = services
        self.methods_allowed = ["GET"]

    def on_get(self, req, resp):
        online_models = {x: self.services.models_available[x] for x in self.services.models_online}
        resp.body = json.dumps(online_models, ensure_ascii=False)
        resp.content_type = falcon.MEDIA_JSON

        resp.status = falcon.HTTP_200


class LoadModel(Origin):
    def __init__(self, services):
        Origin.__init__(self)
        self.services = services
        self.methods_allowed = ["PUT"]

    def validate_load_model(self, req):
        fields_needed = set(['model_name', 'service_name'])
        try:
            json_input = json.load(req.bounded_stream)
        except json.decoder.JSONDecodeError:
            raise falcon.HTTPBadRequest('Bad request', "Json seems malformed")
        if fields_needed != set(json_input.keys()):
            msg = "'model_name' and 'service_name' are the only input needed."
            raise falcon.HTTPBadRequest('Bad request', msg)

        return json_input

    def on_put(self, req, resp):
        json_input = self.validate_load_model(req)
        model_name = json_input["model_name"]
        service_name = json_input["service_name"]
        try:
            # Load model
            model = load_model(model_name)
            success = self.services.create_service(service_name, model)
            if not success:
                raise falcon.HTTP_CONFLICT()
        except Exception as e:
            # TODO add logging
            raise falcon.HTTPNotFound()

        resp.body = json.dumps(list(self.services.models_online.keys()), ensure_ascii=False)
        resp.status = falcon.HTTP_201


class DeleteModel(Origin):
    def __init__(self, services):
        Origin.__init__(self)
        self.services = services
        self.methods_allowed = ["DELETE"]

    def validate_delete_service(self, req):
        fields_needed = set(['service_name'])
        try:
            json_input = json.load(req.bounded_stream)
        except json.decoder.JSONDecodeError:
            # TODO add logging
            raise falcon.HTTPBadRequest('Bad request', "Json seems malformed")
        if fields_needed != set(json_input.keys()):
            msg = "'service_name' is the only input needed."
            raise falcon.HTTPBadRequest('Bad request', msg)

        return json_input

    def on_delete(self, req, resp):
        json_input = self.validate_delete_service(req)
        service_name = json_input["service_name"]
        try:
            success = self.services.delete_service(service_name)
            if not success:
                raise falcon.HTTPNotFound(description="Service does not seem to exist")
        except IOError:
            # TODO add logging
            raise falcon.HTTPNotFound()

        print(self.services.models_online)
        # TODO add messageindicating success
        resp.status = falcon.HTTP_201


class Predict(Origin):
    def __init__(self, services):
        Origin.__init__(self)
        self.services = services
        self.methods_allowed = ["POST"]

    def validate_predict_input(self, req):
        fields_needed = set(['service_name', 'urls_image'])
        try:
            json_input = json.load(req.bounded_stream)
        except json.decoder.JSONDecodeError:
            # TODO add logging
            raise falcon.HTTPBadRequest('Bad request', "Json seems malformed")
        if fields_needed != set(json_input.keys()):
            msg = "'service_name' and 'urls_image' are the only input needed."
            raise falcon.HTTPBadRequest('Bad request', msg)

        if not isinstance(json_input['urls_image'], list):
            raise falcon.HTTPBadRequest(
                'Bad request',
                "urls_image must be a list of urls")

        return json_input

    def predict(self, model, input, topk=10):
        # create mini batch
        input_batch = tuple(model['preprocessing'](x).unsqueeze(0) for x in input)
        input_batch = torch.cat(input_batch, 0)

        # Make prediction
        output = model['model'](input_batch)

        # Make postprocessing
        predictions = [model['postprocessing'](output[x,:]) for x in range(output.shape[0])]

        # Sort prediction and get top wanted
        res = []
        for pred in predictions:
            topk_val, topk_indices = torch.topk(pred, topk)
            if model["mapping"] is None:
                res.append({x.item(): y.item() for x, y in zip(topk_indices, topk_val)})
            else:
                res.append({
                    model["mapping"][x.item()]: y.item() for x, y in zip(topk_indices, topk_val)
                })

        return res

    #@falcon.before(validate_image_type)
    def on_post(self, req, resp):
        json_input = self.validate_predict_input(req)
        # Parse json
        service_name = json_input["service_name"]
        urls_image = json_input["urls_image"]

        # Check if service exists
        if service_name not in self.services.models_online:
            raise falcon.HTTPNotFound(description="Service does not seem to exist")
        else:
            model = self.services.models_online[service_name]
            predictions = []
            for batch in batchify(urls_image, 20):
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
                    prediction = self.predict(model, imgs)

                    # Format output
                    for j, pred in enumerate(prediction):
                        res[id_pred_to_res[j]]['predictions'] = pred

                predictions.append(res)

            resp.body = json.dumps(predictions, ensure_ascii=False)
            resp.status = falcon.HTTP_201
