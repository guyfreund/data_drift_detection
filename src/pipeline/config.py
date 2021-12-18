import os
import ruamel.yaml as ryaml
from easydict import EasyDict as edict

from src.pipeline.paths import DEFAULT_CONFIG_PATH
from src.pipeline.utils import Singleton


class Config(metaclass=Singleton):
    def __new__(cls, path: str = None):
        if not hasattr(cls, '_cfg'):
            if path is None:
                path = DEFAULT_CONFIG_PATH
            cls._cfg = cls.load_from_file(path)
        return cls._cfg

    @staticmethod
    def load_from_file(path: str) -> edict:
        assert os.path.isfile(path)
        with open(path, 'r') as f:
            cfg = ryaml.load(f)
        return edict(cfg)
