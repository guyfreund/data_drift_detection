from src.pipeline.config import Config


def test_config_is_singleton():
    x = Config()
    y = Config()
    assert x == y
