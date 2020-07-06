import falcon
from collections import defaultdict


def list_routes(roots, parent=''):
    elems = []
    for root in roots:
        elem = []
        if root.method_map:
            elem = [parent + '/' + root.raw_segment]

        if root.children:
            return elem + list_routes(root.children, parent + '/' + root.raw_segment)
        else:
            elems += elem
    return elems


class DefaultRoute(object):
    def __init__(self, api):
        self.list_routes = list_routes(api._router._roots)

    def on_get(self, req, resp):
        raise falcon.HTTPNotFound(description="Route not found. Possible routes are {}".format(self.list_routes))


def check_parameter(json_content, param):
    val = json_content.get(param['name'], None)
    if "mandatory" in param and not param["mandatory"] and val is None:
        return True, ""
    if "mandatory" in param and param["mandatory"] and val is None:
        return False, param['messages']['none']
    if not isinstance(val, param["type"]):
        return False, param['messages']['type']
    if "enum" in param and val not in param["enum"]:
        return False, param['messages']['type']
    if "specific_check" in param:
        for check in param['specific_check']:
            if check == "boundaries":
                boundaries = param['specific_check'][check]
                if val < boundaries["min"] or val > boundaries['max']:
                    return False, param['messages']['type']
            else:
                if not param['specific_check'][check](val):
                    return False, param['messages']['type']

    return True, ""


def check_json(json_input, call_params):
    status = True
    message = defaultdict(str)

    # Check every fields of the api
    for param in call_params:
        st, mess = check_parameter(json_input, call_params[param])
        if not st:
            status = False
            message[param] = mess

    return status, message
