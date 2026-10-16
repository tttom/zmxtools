import io
import unittest
from pathlib import Path

from numpy import testing as npt

from zmxtools import zar
from zmxtools.utils.io import BytesFile

ZAR_FILE = Path(__file__).resolve().parent.parent / "data" / "LA1116-Zemax.zar"


class TestZarPaths(unittest.TestCase):
    def test_extract_str_path(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = zar.extract(str(ZAR_FILE), output_path=tmp)
            npt.assert_equal(out is None, True)
            files = list(Path(tmp).rglob("*"))
            npt.assert_equal(len([_ for _ in files if _.is_file()]) >= 3, True)

    def test_extract_stream_raises(self):
        with open(ZAR_FILE, "rb") as f:
            with self.assertRaises(TypeError):
                zar.extract(f, output_path="/tmp/does-not-matter")

    def test_repack_str_output(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            zar.repack(ZAR_FILE, output_path_or_stream=f"{tmp}/out.zip")
            npt.assert_equal(Path(tmp, "out.zip").exists(), True)

    def test_repack_path_output(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "out.zip"
            zar.repack(ZAR_FILE, output_path_or_stream=out)
            npt.assert_equal(out.exists(), True)

    def test_repack_directory_output(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dir"
            zar.repack(ZAR_FILE, output_path_or_stream=out)
            zips = list(out.rglob("*.zip"))
            npt.assert_equal(len(zips) >= 1, True)

    def test_repack_stream_output(self):
        class NamedBytesIO(io.BytesIO):
            name = "memory.zip"

        stream = NamedBytesIO()
        zar.repack(ZAR_FILE, output_path_or_stream=stream)
        npt.assert_equal(len(stream.getvalue()) > 0, True)

    def test_unpack_stream(self):
        with open(ZAR_FILE, "rb") as f:
            files = list(zar.unpack(f))
        npt.assert_equal(len(files) >= 3, True)

    def test_unpack_bytesfile(self):
        bf = BytesFile(ZAR_FILE)
        files = list(zar.unpack(bf))
        npt.assert_equal(len(files) >= 3, True)

    def test_load_stream(self):
        with open(ZAR_FILE, "rb") as f:
            designs = zar.load(f)
        npt.assert_equal(len(designs) >= 1, True)


if __name__ == "__main__":
    unittest.main()
