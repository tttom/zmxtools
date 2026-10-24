import os
import shutil

from tests.zar import check_dir_and_remove, check_zip_and_remove, test_directory, test_zar_files
from zmxtools import cli, config_logger

log = config_logger(__name__)





def test_unzar_show(monkeypatch):
    """Tests that --show passes the unpacked output on to oshow."""
    from zmxtools import gui

    zar_full_file = next(iter(test_zar_files))
    output_path = test_directory / "tmp_view"
    output_zip_full_file = output_path / (zar_full_file.stem + ".zip")

    calls: list[list[str]] = []
    return_code = [0]

    def fake_oshow(paths):
        calls.append(list(paths))
        return return_code[0]

    monkeypatch.setattr(gui, "oshow", fake_oshow)

    exit_code = cli.unzar(["-vvv", f"-i {zar_full_file}", f"-o {output_path}", "--show"])
    assert exit_code == 0, f"Unzar tool returned error code {exit_code}, though expected 0."
    assert calls == [[str(output_path / zar_full_file.stem)]]
    check_dir_and_remove(output_path / zar_full_file.stem)

    return_code[0] = 1
    exit_code = cli.unzar(["-vvv", f"-i {zar_full_file}", f"-o {output_path}", "-z", "--show"])
    assert exit_code == 0, f"Unzar tool returned error code {exit_code}, though expected 0."
    assert calls == [[str(output_path / zar_full_file.stem)], [str(output_zip_full_file)]]
    check_zip_and_remove(output_zip_full_file, remove=False)

    calls.clear()
    exit_code = cli.unzar(["-vvv", f"-i {zar_full_file}", f"-o {output_path}", "-z", "--show"])
    assert exit_code == 1, f"Unzar tool returned error code {exit_code}, though expected 1 since file exists."
    assert calls == []
    check_zip_and_remove(output_zip_full_file)

    output_path.rmdir()


def test_unzar_non_zar_extension():
    """Checks whether the command line interface is happy to take zar files with a different extension."""
    zar_full_file = next(iter(test_zar_files))
    non_zar_full_file = zar_full_file.with_name(zar_full_file.name[:-4] + "_zar_rename.nzr")
    shutil.copyfile(zar_full_file, non_zar_full_file)
    log.debug(f"Trying to read a zar file with the wrong extension {non_zar_full_file} as a .zar file...")
    exit_code = cli.unzar(["-vvv", f"-i {non_zar_full_file}", "-zf"])
    os.remove(non_zar_full_file)
    os.remove(non_zar_full_file.with_suffix(".zip"))
    assert exit_code == 0, f"Unzar tool returned error code {exit_code}, though expected 0."


def test_unzar_non_zar():
    """Test robustness of command line interface when a corrupted .zar file is presented."""
    zmx_full_file = next(iter(test_directory.glob("**/LA1116-Zemax.zmx")))
    non_zar_full_file = zmx_full_file.with_name(zmx_full_file.name[:-4] + "_zmx_rename.zar")
    shutil.copyfile(zmx_full_file, non_zar_full_file)
    log.debug(f"Trying to read a zar file with the wrong extension {non_zar_full_file} as a .zar file...")
    exit_code = cli.unzar(["-vvv", f"-i {non_zar_full_file}", "-zf"])
    os.remove(non_zar_full_file)
    os.remove(non_zar_full_file.with_suffix(".zip"))
    assert exit_code == -1, (
        "Unzar tool was expected to encounter an unexpected MemoryError and return with -1," + f" not {exit_code}."
    )
