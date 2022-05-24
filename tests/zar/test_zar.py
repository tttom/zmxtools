import logging
from pathlib import Path
from typing import Dict, Optional

from zmxtools import zar

log = logging.getLogger(__name__)

MIN_FILES_IN_ARCHIVE = 3

test_folder = Path(__file__).resolve().parent.parent / 'data'
test_files: Dict[Path, Optional[Path]] = {_: None for _ in test_folder.glob('**/*.zar')}
for _ in test_folder.glob('**/*.zmx'):
    zar_file = _.parent / (_.name[:-4] + '.zar')
    if zar_file in test_files.keys():
        test_files[zar_file] = _
paired_test_files = {zar_path: zmx_path for zar_path, zmx_path in test_files.items() if zmx_path is not None}


def test_read():
    """Tests the zmxtools.zar.read function."""
    assert len(paired_test_files) > 1, (
        f'No zar files found with matching zmx files in {test_folder}! Make sure that the extensions are lower case.'
    )

    for zar_file_path in paired_test_files.keys():
        log.info(zar_file_path)
        packed_files = []
        for packed_data in zar.read(zar_file_path):
            packed_files.append(packed_data.file_name)
            if packed_data.file_name.lower().endswith('.zmx'):
                try:
                    with open(paired_test_files[zar_file_path], 'rb') as zmx_file:
                        assert zmx_file.read() == packed_data.unpacked_contents, (
                            f'Data in {packed_data.file_name} is not what is expected.'
                        )
                except AssertionError as exc:  # Write out the actual file for later reference
                    with open(
                        paired_test_files[zar_file_path].parent / (
                            paired_test_files[zar_file_path].name[:-4] + '_actual.zmx'
                        ),
                        'wb',
                    ) as actual_zmx_file:
                        actual_zmx_file.write(packed_data.unpacked_contents)  # for debugging
                    raise exc
        assert len(packed_files) >= MIN_FILES_IN_ARCHIVE, (
            f'Expected more files than {packed_files} in {repr(zar_file_path)}'
        )


def test_extract():
    """Tests the zmxtools.zar.extract function."""
    assert len(test_files) > 1, (
        f'No zar files found in {test_folder}! Make sure that the extensions are lower case.'
    )

    def check(zar_file_path: Path):
        extraction_dir = zar_file_path.parent / zar_file_path.name[:-4]
        assert extraction_dir.exists, f'Extraction of zar file {repr(zar_file_path)} to {extraction_dir} failed'
        files_in_archive = tuple(extraction_dir.glob('*'))
        assert len(files_in_archive) >= MIN_FILES_IN_ARCHIVE, (
            f'Only found {files_in_archive} in {extraction_dir}. Expected {MIN_FILES_IN_ARCHIVE} files.'
        )

        # Delete files again and remove sub-directory
        for _ in files_in_archive:
            _.unlink()
        extraction_dir.rmdir()

    for zar_file_path in test_files.keys():
        zar.extract(zar_file_path)
        check(zar_file_path)
        zar.extract(zar_file_path.as_posix())  # Checking if str also work as argument
        check(zar_file_path)


def test_repack():
    """Tests the zmxtools.zar.repack function."""
    assert len(test_files) > 1, (
        f'No zar files found in {test_folder}! Make sure that the extensions are lower case.'
    )

    def check_and_clean(zar_file_path: Path):
        zip_file_path = zar_file_path.parent / (zar_file_path.name[:-4] + '.zip')
        assert zip_file_path.exists, f'Repacking of zar file {repr(zar_file_path)} as {zip_file_path} failed!'
        zip_file_path.unlink()

    for zar_file_path in test_files.keys():
        zar.repack(zar_file_path)
        check_and_clean(zar_file_path)
        zar.repack(zar_file_path.as_posix())  # Checking if str also work as argument
        check_and_clean(zar_file_path)


def test_unzar():
    """Tests the code that is called as a script."""
    exit_code = zar.unzar()
    assert exit_code != 0, f'Unzar tool returned error code {exit_code}, though -1 expected when running tests.'
    exit_code = zar.unzar('-qq')
    assert exit_code != 0, f'Unzar tool returned error code {exit_code}, though -1 expected when running tests.'
    exit_code = zar.unzar('-q')
    assert exit_code != 0, f'Unzar tool returned error code {exit_code}, though -1 expected when running tests.'
    exit_code = zar.unzar('-v')
    assert exit_code != 0, f'Unzar tool returned error code {exit_code}, though -1 expected when running tests.'
    exit_code = zar.unzar('-vv')
    assert exit_code != 0, f'Unzar tool returned error code {exit_code}, though -1 expected when running tests.'
    exit_code = zar.unzar('-vvv')
    assert exit_code != 0, f'Unzar tool returned error code {exit_code}, though -1 expected when running tests.'
    # exit_code = zar.unzar('-vvv', '-i tests/data/*.zar', '-zo tests/data/tmp/')
    # assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0.'
    # exit_code = zar.unzar('-vvv', '-i tests/data/*.zar', '-zo tests/data/tmp/')
    # assert exit_code == 2, f'Unzar tool returned error code {exit_code}, though expected 2 because the files already exist.'
    # exit_code = zar.unzar('-vvv', '-i tests/data/*.zar', '-zfo tests/data/tmp/')
    # assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0 because we used --force.'
    # exit_code = zar.unzar('-vvv', '-i tests/data/*.zar', '-o tests/data/tmp/')
    # assert exit_code == 0, f'Unzar tool returned error code {exit_code}, though expected 0.'
    # # clean up
    # tmp_dir = Path('tests/data/tmp/')
    # tmp_dir.unlink()

