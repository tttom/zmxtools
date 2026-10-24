import logging
from types import ModuleType

import zmxtools

log = logging.getLogger(__name__)


def test_version():
    """Tests whether the version number is loaded correctly."""
    version = [int(_) for _ in zmxtools.__version__.split('.')[:3]]
    assert all(_ >= 0 for _ in version), f'Version {zmxtools.__version__} does not start with 3 non-negative digits.'

def test_logs():
    log = zmxtools.config_logger(__name__)
    log.info('Testing the logs')

    log = zmxtools.log_to_file(__file__)
    log.info('Logging to file...')

def test_modules():
    from zmxtools import agf, zar, zmx

    assert isinstance(agf, ModuleType)
    assert isinstance(zmx, ModuleType)
    assert isinstance(zar, ModuleType)
