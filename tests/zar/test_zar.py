import logging
from typing import List

from tests.zar import MIN_FILES_IN_ARCHIVE, check_dir_and_remove, check_zip_and_remove, test_directory, test_zar_files
from zmxtools import cli, zar

from tests.zar import log
log = log.getChild(__name__)
log.level = logging.DEBUG


paired_test_files = {zar_path: zmx_path for zar_path, zmx_path in test_zar_files.items() if zmx_path is not None}


def test_unpack():
    """Tests the zmxtools.zar.unpack function."""
    assert len(paired_test_files) > 1, (
        f'No zar files found with matching zmx files in {test_directory}! Make sure that the extensions are lower case.'
    )

    for zar_full_file in paired_test_files.keys():
        log.info(zar_full_file)
        packed_files = []
        for packed_data in zar.unpack(zar_full_file.as_posix()):  # Use a str as argument, others already use pathlib.Path
            packed_files.append(packed_data.name)
            if packed_data.name.lower().endswith('.zmx'):
                try:
                    with open(paired_test_files[zar_full_file], 'rb') as zmx_file:
                        assert zmx_file.read() == packed_data.read(), (
                            f'Data in {packed_data.name} is not what is expected.'
                        )
                except AssertionError as exc:  # Write out the unpacked file for later reference
                    with open(
                        paired_test_files[zar_full_file].parent / (
                            paired_test_files[zar_full_file].name[:-4] + '_unpacked.zmx'
                        ),
                        'wb',
                    ) as actual_zmx_file:
                        actual_zmx_file.write(packed_data.read())  # for debugging
                    raise exc
        assert len(packed_files) >= MIN_FILES_IN_ARCHIVE, (
            f'Expected more files than {packed_files} in {repr(zar_full_file)}'
        )


def test_extract():
    """Tests the zmxtools.zar.extract function."""
    assert len(test_zar_files) > 1, (
        f'No zar files found in {test_directory}! Make sure that the extensions are lower case.'
    )

    for zar_full_file in test_zar_files.keys():
        zar.extract(zar_full_file)
        output_dir = zar_full_file.parent / zar_full_file.stem
        check_dir_and_remove(output_dir)
        zar.extract(zar_full_file.as_posix())  # Checking if str also work as input argument
        check_dir_and_remove(output_dir)
        zar.extract(zar_full_file, output_path=output_dir.as_posix())  # also with str output argument?
        check_dir_and_remove(output_dir)


def test_repack():
    """Tests the zmxtools.zar.repack function."""
    assert len(test_zar_files) > 1, (
        f'No zar files found in {test_directory}! Make sure that the extensions are lower case.'
    )

    for zar_full_file in test_zar_files.keys():
        zip_full_file = zar_full_file.with_suffix('.zip')
        zar.repack(zar_full_file)
        check_zip_and_remove(zip_full_file)
        zar.repack(zar_full_file.as_posix())  # Checking if str also work as argument
        check_zip_and_remove(zip_full_file)
        # also with str output argument
        zar.repack(zar_full_file, zip_full_file.as_posix())  # remove extension on purpose
        check_zip_and_remove(zip_full_file)


def test_unzar_basic():
    """Do some basic tests on the code that is called as a script."""
    no_argv: List[str] = []
    test_args: List[List[str]] = [['-vv'], no_argv, ['-qq'], ['-q'], ['-v'], ['-vv'], ['-vvv'], ['-V'], ['--FATAL']]
    for argv in test_args:
        log.info(f'checking unzar{tuple(argv)}...')
        exit_code = cli.unzar(argv)
        assert exit_code == 2, f'cli.unzar{tuple(argv)} returned error code {exit_code}, though 2 expected.'


def test_unzar_full():
    """Tests the code on actual archive files."""
    zar_full_file = list(test_zar_files.keys())[0]
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


def test_load():
    """Tests the zmxtools.zar.unpack function."""
    for zar_full_file in test_zar_files.keys():
        log.info(zar_full_file)
        optical_design = zar.load(zar_full_file.as_posix())[0]
        assert optical_design.unit == 1e-3, f"unit = {optical_design.unit}, not mm."
        assert len(optical_design.wavelengths) > 0, f"No wavelengths specified in {zar_full_file}."
        assert all(150e-9 <= _ <= 15e-6 for _ in optical_design.wavelengths), f"Wavelengths not optical: {optical_design.wavelengths}"
        assert optical_design.sequential, f"Non-sequential optical model {zar_full_file}."
        assert len(optical_design.material_libraries), f"No material libraries specified in {zar_full_file}."
        assert len(optical_design.surfaces) >= 3, f"Number of surfaces ({len(optical_design.surfaces)}) is expected to be larger."
        total_track = sum(_.distance for _ in optical_design.surfaces[1:-1])
        assert 1e-3 <= total_track <= 1.0, f"Total track {total_track} too extreme in {zar_full_file}."
        assert any(_.curvature != 0.0 for _ in optical_design.surfaces), f"No curved surfaces found in in {zar_full_file}."
        # log.info(optical_design.surfaces[1].material.wavelength_limits)
