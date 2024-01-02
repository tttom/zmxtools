from pathlib import Path
from typing import Dict, Optional

from tests import log
log = log.getChild(__name__)

__all__ = ['MIN_FILES_IN_ARCHIVE', 'test_directory', 'test_zar_files', 'check_dir_and_remove', 'check_zip_and_remove']

MIN_FILES_IN_ARCHIVE = 3

test_directory = Path(__file__).resolve().parent.parent / 'data'

test_zar_files: Dict[Path, Optional[Path]] = {_: None for _ in test_directory.rglob('*') if _.suffix.lower() == ".zar"}
for _ in test_directory.rglob('*'):
    if _.suffix.lower() == ".zmx":
        zar_file = _.parent / (_.stem + '.zar')
        if zar_file in test_zar_files.keys():
            test_zar_files[zar_file] = _


def check_dir_and_remove(extraction_dir: Path, remove: bool = True):
    """Checks of the directory exists and removes it."""
    assert extraction_dir.exists, f'Extraction of zar file to {extraction_dir} failed'
    files_in_archive = tuple(extraction_dir.glob('*'))
    assert len(files_in_archive) >= MIN_FILES_IN_ARCHIVE, (
        f'Only found {files_in_archive} in {extraction_dir}. Expected {MIN_FILES_IN_ARCHIVE} files.'
    )

    if remove:
        # Delete files again and remove subdirectory
        log.debug(f'Deleting directory {extraction_dir} with the extracted contents...')
        for _ in files_in_archive:
            _.unlink()
        extraction_dir.rmdir()
        log.info(f'Deleted directory {extraction_dir} with the extracted contents.')


def check_zip_and_remove(zip_full_file: Path, remove: bool = True):
    """Checks if the zip file exists and deletes it unless otherwise specified."""
    assert zip_full_file.exists, f'Repacking of zar file as {zip_full_file} failed!'
    if remove:
        log.debug(f'Deleting file {zip_full_file}...')
        zip_full_file.unlink()
        log.info(f'Deleted file {zip_full_file}.')
