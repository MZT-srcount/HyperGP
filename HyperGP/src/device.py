class BackendDevice:
    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __getattr__(self, item):
        return getattr(self.mod, item)

    def __eq__(self, other):
        return self.name == other.name
