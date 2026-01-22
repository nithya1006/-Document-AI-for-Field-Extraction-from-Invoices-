def __init__(self, config: dict = None):
    default = self._default_config()
    self.config = default

    if config:
        for key in config:
            self.config.setdefault(key, {}).update(config[key])

    self.dealer_master = []
    self.model_master = []
