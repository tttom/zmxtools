import logging
import os
import shutil
from typing import List

from tests.zar import check_dir_and_remove, check_zip_and_remove, test_directory, test_files
from zmxtools import cli

log = logging.getLogger(__name__)


def test_unzar_basic():
    """Do some basic tests on the code that is called as a script."""
    no_argv: List[str] = []
    test_args: List[List[str]] = [['-vv'], no_argv, ['-qq'], ['-q'], ['-v'], ['-vv'], ['-vvv'], ['-V'], ['--FATAL']]
    for argv in test_args:
        log.info(f'checking unzar{tuple(argv)}...')
        exit_code = cli.unzar(argv)
        assert exit_code == 2, f'unzar{tuple(argv)} returned error code {exit_code}, though 2 expected.'


def test_unzar_full():
    """Tests the code on actual archive files."""
    zar_full_file = list(test_files.keys())[0]
    output_path = test_directory / 'tmp'
    output_zip_full_file = output_path / (zar_full_file.stem + '.zip')

    log.info(f'Unpacking {zar_full_file}...')

    exit_code = cli.unzar(['-vvv', f'-i {zar_full_file}', '-z'])
    assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0.'
    check_zip_and_remove(zar_full_file.with_suffix('.zip'))

    exit_code = cli.unzar(['-vvv', f'-i {zar_full_file}', f'-o {output_path}', '-z'])
    assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0.'
    check_zip_and_remove(output_zip_full_file, remove=False)

    exit_code = cli.unzar(['-vvv', f'-i {zar_full_file}', f'-o {output_path}', '-z'])
    assert exit_code == 1, f'Unzar tool returned error code {exit_code}, though expected 1 since file already exist.'
    check_zip_and_remove(output_zip_full_file, remove=False)

    exit_code = cli.unzar(['-vvv', f'-i {zar_full_file}', f'-o {output_path}', '-z', '-f'])
    assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0 because we used --force.'
    check_zip_and_remove(output_zip_full_file)

    exit_code = cli.unzar(['-vvv', f'-i {zar_full_file}', f'-o {output_path}'])
    assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0.'
    check_dir_and_remove(output_path / zar_full_file.stem)

    log.debug(f'Deleting {output_path} directory...')
    output_path.rmdir()
    log.info(f'Deleted {output_path} directory.')


def test_unzar_non_zar_extension():
    """Checks whether the command line interface is happy to take zar files with a different extension."""
    zar_full_file = list(test_files.keys())[0]
    non_zar_full_file = zar_full_file.with_name(zar_full_file.name[:-4] + '_zar_rename.nzr')
    shutil.copyfile(zar_full_file, non_zar_full_file)
    log.debug(f'Trying to read a zar file with the wrong extension {non_zar_full_file} as a .zar file...')
    exit_code = cli.unzar(['-vvv', f'-i {non_zar_full_file}', '-zf'])
    os.remove(non_zar_full_file)
    os.remove(non_zar_full_file.with_suffix('.zip'))
    assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0.'


def test_unzar_non_zar():
    """Test robustness of command line interface when a corrupted .zar file is presented."""
    zmx_full_file = list(test_directory.glob('**/LA1116-Zemax.zmx'))[0]
    non_zar_full_file = zmx_full_file.with_name(zmx_full_file.name[:-4] + '_zmx_rename.zar')
    shutil.copyfile(zmx_full_file, non_zar_full_file)
    log.debug(f'Trying to read a zar file with the wrong extension {non_zar_full_file} as a .zar file...')
    exit_code = cli.unzar(['-vvv', f'-i {non_zar_full_file}', '-zf'])
    os.remove(non_zar_full_file)
    os.remove(non_zar_full_file.with_suffix('.zip'))
    assert exit_code == -1, ('Unzar tool was expected to encounter an unexpected MemoryError and return with -1,' +
                             f' not {exit_code}.'
                             )
