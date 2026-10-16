from typing import cast

import pytest
from numpy import testing as npt

from zmxtools.parser import Command, OrderedCommandDict
from zmxtools.utils import array as arr
from zmxtools.utils.io import BytesFile


class TestArrayHelpers:
    def test_concatenate(self):
        npt.assert_array_equal(arr.concatenate([1, 2], [3, 4]), [1, 2, 3, 4])

    def test_stack(self):
        npt.assert_array_equal(arr.stack([1, 2], [3, 4]), [[1, 3], [2, 4]])

    def test_dot(self):
        npt.assert_allclose(arr.dot([1, 2, 3], [4, 5, 6]), 32)

    def test_cross(self):
        npt.assert_array_equal(arr.cross([1, 0, 0], [0, 1, 0]), [0, 0, 1])

    def test_einsum(self):
        npt.assert_allclose(arr.einsum("...i,...i->...", [1, 2], [3, 4]), 11)

    def test_norm(self):
        npt.assert_allclose(arr.norm([3, 4]), 5)

    def test_to_length(self):
        npt.assert_array_equal(arr.to_length([1, 2], 4), [1, 2, 0, 0])
        npt.assert_array_equal(arr.to_length([1, 2, 3, 4, 5], 2), [1, 2])
        npt.assert_array_equal(arr.to_length([1], 3, value=9), [1, 9, 9])


class TestBytesFile:
    def test_from_path(
        self,
    ):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/x.bin"
            with open(path, "wb") as f:
                f.write(b"hello")
            bf = BytesFile(path)
            npt.assert_equal(bf.read(), b"hello")
            npt.assert_equal(bf.name, path)

    def test_from_contents(self):
        bf = BytesFile("fictitious/name.bin", b"data")
        npt.assert_equal(bf.read(2), b"da")
        with bf as opened:
            npt.assert_equal(opened.read(), b"data")
        bf.open()
        bf.close()
        bf.close()  # closing twice is fine


class TestParserExtras:
    def test_base_from_str_not_implemented(self):
        with pytest.raises(NotImplementedError):
            OrderedCommandDict.from_str("X 1")

    def test_from_file_text_stream(self):
        import tempfile

        from zmxtools.zmx import ZmxOrderedCommandDict

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/x.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write("X 1\n  Y 2\n")
            with open(path, encoding="utf-8") as f:
                commands = ZmxOrderedCommandDict.from_file(f)
            npt.assert_equal("X" in commands, True)
            npt.assert_equal(commands.command("X").command("Y").text, "2")

    def test_contains_command(self):
        c = Command("A", "1")
        d = OrderedCommandDict([c])
        npt.assert_equal(c in d, True)
        npt.assert_equal(Command("A", "1") in d, False)
        npt.assert_equal("A" in d, True)

    def test_getitem(self):
        d = OrderedCommandDict([Command("A", "1"), Command("B", "2"), Command("A", "3")])
        npt.assert_equal(cast("Command", d[0]).name, "A")
        npt.assert_equal(len(d[0:2]), 2)
        npt.assert_equal(len(d["A"]), 2)
        npt.assert_equal(cast("Command", d["A", 1]).argument, "3")
        with pytest.raises(IndexError):
            d[17]

    def test_words_and_numbers(self):
        c = Command("X", "'a b' \"c d\" '''e\nf''' plain")
        npt.assert_array_equal(c.words, ["a b", "c d", "e\nf", "plain"])
        n = Command("Y", "1 2.5 x 3")
        npt.assert_array_equal(n.numbers, [1, 2.5, 3])
        npt.assert_equal(repr(c).startswith("Command("), True)

    def test_sort_and_merge(self):
        d = OrderedCommandDict(
            [
                Command("PARM", "0 1.0"),
                Command("PARM", "0 2.0"),
                Command("PARM", "1 3.0"),
                Command("OTHER", "text"),
            ],
        )
        merged = d.sort_and_merge("PARM")
        npt.assert_equal(len(merged), 2)
        npt.assert_array_equal(cast("Command", merged[0]).numbers, [1.0, 2.0])
        npt.assert_array_equal(cast("Command", merged[1]).numbers, [3.0])
        # Commands with non-integer arguments are skipped
        npt.assert_equal(len(d.sort_and_merge("OTHER")), 0)

    def test_append_and_len(self):
        d = OrderedCommandDict()
        d.append(Command("A", "1"))
        npt.assert_equal(len(d), 1)
        npt.assert_equal(d.__sizeof__(), 1)
        npt.assert_equal(str(d), "A 1")
        npt.assert_equal(repr(d).startswith("OrderedCommandDict("), True)

    def test_command_children_helpers(self):
        parent = Command("P", "0")
        parent.append(Command("C", "1"))
        npt.assert_equal(parent.command("C").text, "1")
        npt.assert_equal(len(parent.commands_of("C")), 1)
        npt.assert_equal("C" in parent, True)
        npt.assert_equal(len(parent), 1)
        with pytest.raises(IndexError):
            Command("NoChildren")[0]
        npt.assert_equal(str(Command("B")), "B")
