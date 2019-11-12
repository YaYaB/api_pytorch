import falcon


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
