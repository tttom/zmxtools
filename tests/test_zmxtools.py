import logging

from zmxtools import __version__

log = logging.getLogger(__name__)


def test_version():
    """Tests whether the version number is loaded correctly."""
    version = __version__
    log.info(f'Testing package version "{version}"...')
    assert isinstance(version, str), f'Version number "{version}" not a character string.'
    version_parts = version.split('-')
    version_number = version_parts[0].split('.')
    assert len(version_number) == 3, f'Version number "{version}" should consist of 3 numbers.'
    assert all(int(_) >= 0 for _ in version_number), (
        f'Version number "{version}" should consist of 3 non-negative numbers.',
    )
    assert any(int(_) > 0 for _ in version_number), f'Version number "{version}" should be greater than 0.'
