import logging
from pathlib import Path
from typing import Dict, List, Optional

from zmxtools import zar

log = logging.getLogger(__name__)

MIN_FILES_IN_ARCHIVE = 3

test_directory = Path(__file__).resolve().parent.parent / 'data'
test_files: Dict[Path, Optional[Path]] = {_: None for _ in test_directory.glob('**/*.zar')}
for _ in test_directory.glob('**/*.zmx'):
    zar_file = _.parent / (_.stem + '.zar')
    if zar_file in test_files.keys():
        test_files[zar_file] = _
paired_test_files = {zar_path: zmx_path for zar_path, zmx_path in test_files.items() if zmx_path is not None}


def test_read():
    """Tests the zmxtools.zar.read function."""
    assert len(paired_test_files) > 1, (
        f'No zar files found with matching zmx files in {test_directory}! Make sure that the extensions are lower case.'
    )

    for zar_full_file in paired_test_files.keys():
        log.info(zar_full_file)
        packed_files = []
        for packed_data in zar.read(zar_full_file.as_posix()):  # Use a str as argument, others already use pathlib.Path
            packed_files.append(packed_data.file_name)
            if packed_data.file_name.lower().endswith('.zmx'):
                try:
                    with open(paired_test_files[zar_full_file], 'rb') as zmx_file:
                        assert zmx_file.read() == packed_data.unpacked_contents, (
                            f'Data in {packed_data.file_name} is not what is expected.'
                        )
                except AssertionError as exc:  # Write out the actual file for later reference
                    with open(
                        paired_test_files[zar_full_file].parent / (
                            paired_test_files[zar_full_file].name[:-4] + '_actual.zmx'
                        ),
                        'wb',
                    ) as actual_zmx_file:
                        actual_zmx_file.write(packed_data.unpacked_contents)  # for debugging
                    raise exc
        assert len(packed_files) >= MIN_FILES_IN_ARCHIVE, (
            f'Expected more files than {packed_files} in {repr(zar_full_file)}'
        )


def _check_dir_and_remove(extraction_dir: Path):
    """Checks of the directory exists and removes it."""
    assert extraction_dir.exists, f'Extraction of zar file to {extraction_dir} failed'
    files_in_archive = tuple(extraction_dir.glob('*'))
    assert len(files_in_archive) >= MIN_FILES_IN_ARCHIVE, (
        f'Only found {files_in_archive} in {extraction_dir}. Expected {MIN_FILES_IN_ARCHIVE} files.'
    )

    # Delete files again and remove sub-directory
    log.debug(f'Deleting directory {extraction_dir} with the extracted contents...')
    for _ in files_in_archive:
        _.unlink()
    extraction_dir.rmdir()
    log.info(f'Deleted directory {extraction_dir} with the extracted contents.')


def test_extract():
    """Tests the zmxtools.zar.extract function."""
    assert len(test_files) > 1, (
        f'No zar files found in {test_directory}! Make sure that the extensions are lower case.'
    )

    for zar_full_file in test_files.keys():
        zar.extract(zar_full_file)
        output_dir = zar_full_file.parent / zar_full_file.stem
        _check_dir_and_remove(output_dir)
        zar.extract(zar_full_file.as_posix())  # Checking if str also work as input argument
        _check_dir_and_remove(output_dir)
        zar.extract(zar_full_file, output_path=output_dir.as_posix())  # also with str output argument?
        _check_dir_and_remove(output_dir)


def _check_zip_and_remove(zip_full_file: Path, remove: bool = True):
    """Checks if the zip file exists and deletes it unless otherwise specified."""
    assert zip_full_file.exists, f'Repacking of zar file as {zip_full_file} failed!'
    if remove:
        log.debug(f'Deleting file {zip_full_file}...')
        zip_full_file.unlink()
        log.info(f'Deleted file {zip_full_file}.')


def test_repack():
    """Tests the zmxtools.zar.repack function."""
    assert len(test_files) > 1, (
        f'No zar files found in {test_directory}! Make sure that the extensions are lower case.'
    )

    for zar_full_file in test_files.keys():
        zip_full_file = zar_full_file.with_suffix('.zip')
        zar.repack(zar_full_file)
        _check_zip_and_remove(zip_full_file)
        zar.repack(zar_full_file.as_posix())  # Checking if str also work as argument
        _check_zip_and_remove(zip_full_file)
        # also with str output argument
        zar.repack(zar_full_file, zip_full_file.as_posix())  # remove extension on purpose
        _check_zip_and_remove(zip_full_file)


def test_unzar_basic():
    """Do some basic tests on the code that is called as a script."""
    no_argv: List[str] = []
    test_args: List[List[str]] = [['-vv'], no_argv, ['-qq'], ['-q'], ['-v'], ['-vv'], ['-vvv'], ['-V'], ['--FATAL']]
    for argv in test_args:
        log.info(f'checking unzar{tuple(argv)}...')
        exit_code = zar.unzar(argv)
        assert exit_code == 2, f'unzar{tuple(argv)} returned error code {exit_code}, though 2 expected.'


def test_unzar_full():
    """Tests the code on actual archive files."""
    zar_full_file = list(test_files.keys())[0]
    output_path = test_directory / 'tmp'
    output_zip_full_file = output_path / (zar_full_file.stem + '.zip')

    log.info(f'Unpacking {zar_full_file}...')

    exit_code = zar.unzar(['-vvv', f'-i {zar_full_file}', '-z'])
    assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0.'
    _check_zip_and_remove(zar_full_file.with_suffix('.zip'))

    exit_code = zar.unzar(['-vvv', f'-i {zar_full_file}', f'-o {output_path}', '-z'])
    assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0.'
    _check_zip_and_remove(output_zip_full_file, remove=False)

    exit_code = zar.unzar(['-vvv', f'-i {zar_full_file}', f'-o {output_path}', '-z'])
    assert exit_code == 1, f'Unzar tool returned error code {exit_code}, though expected 1 since file already exist.'
    _check_zip_and_remove(output_zip_full_file, remove=False)

    exit_code = zar.unzar(['-vvv', f'-i {zar_full_file}', f'-o {output_path}', '-z', '-f'])
    assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0 because we used --force.'
    _check_zip_and_remove(output_zip_full_file)

    exit_code = zar.unzar(['-vvv', f'-i {zar_full_file}', f'-o {output_path}'])
    assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0.'
    _check_dir_and_remove(output_path / zar_full_file.stem)

    log.debug(f'Deleting {output_path} directory...')
    output_path.rmdir()
    log.info(f'Deleted {output_path} directory.')
