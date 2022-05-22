import argparse
import logging
import sys
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Union

import zmxtools

log = logging.getLogger(__name__)

__all__ = ['read', 'UnpackedData', 'extract', 'repack']


def _decompress_lzw(compressed: bytes) -> bytes:
    """
    Decompresses bytes using the variable LZW algorithm, starting with code strings of length 9.

    This function is used internally by the read function.
    General information about LZW: https://en.wikipedia.org/wiki/Lempel%E2%80%93Ziv%E2%80%93Welch
    Adapted partly from https://gist.github.com/BertrandBordage/611a915e034c47aa5d38911fc0bc7df9

    :param compressed: The compressed bytes without header.
    :return: The decompressed bytes.
    """
    # Convert input to bits
    compressed_bits: str = bin(int.from_bytes(compressed, 'big'))[2:].zfill(len(compressed) * 8)
    # convert to binary string and pad to 8-fold length

    code_word_length = 8
    words: List[bytes] = [_.to_bytes(1, 'big') for _ in range(2**code_word_length)]
    # integer codes refer to a words in an expanding dictionary

    bit_index = 0
    previous_word: bytes = b''
    decompressed: List[bytes] = []

    while True:
        if 2**code_word_length <= len(words):  # If the dictionary is full
            code_word_length += 1              # increase the code word length
        if bit_index + code_word_length > len(compressed_bits):
            break  # stop when the bits run out
        # Get the next code word from the data bit string
        code = int(compressed_bits[bit_index:bit_index + code_word_length], 2)
        bit_index += code_word_length

        # If word in dictionary, use it; else add it as a new word
        latest_word: bytes = words[code] if code < len(words) else previous_word + previous_word[:1]
        decompressed.append(latest_word)  # Update result
        if len(previous_word) > 0:  # Skip first iteration
            words.append(previous_word + latest_word[:1])  # Add as new encoding

        previous_word = latest_word

    return b''.join(decompressed)  # convert to bytes


@dataclass
class UnpackedData(object):
    """A structure to represent the file blocks in a zar-archive.

    Parameters:
        name: A string with the name of the file contained in the archive.
        unpacked_contents: The unpacked (decompressed) bytes of this file.
    """

    file_name: str
    unpacked_contents: bytes


def read(input_file_path: Union[Path, str]) -> Generator[UnpackedData, None, None]:
    """
    Reads a zar archive file and generates a series of (unpacked file name, unpacked file contents) tuples.

    The returned Generator produces tuples in the order found in the archive.

    :param input_file_path: The archive or the path to the archive.
    :return: A Generator of name-data tuples.
    """
    # Make sure that the input arguments are both pathlib.Path-s
    if isinstance(input_file_path, str):
        input_file_path = Path(input_file_path)
    with open(input_file_path, 'rb') as input_file:
        version_length = 2
        while True:
            version = input_file.read(version_length)
            if len(version) < version_length:
                break  # end of file
            if version[0] == 0xEC:
                header_length = 0x288 - version_length
            elif version[0] == 0xEA:
                header_length = 0x14C - version_length
            else:
                log.warning(f'Unknown ZAR header "{version.hex()}"!')
                header_length = 0x288 - version_length
                version = 0xEC03.to_bytes(2, 'big')  # override and cross fingers

            header = input_file.read(header_length)

            # flag1 = int.from_bytes(header[0x4-version_length:0x8-version_length], byteorder='little', signed=False)
            # flag2 = int.from_bytes(header[0x8-version_length:0x10-version_length], byteorder='little', signed=False)
            # log.info(f'Header {flag1} {flag2} {header[0x20-version_length:0x30-version_length].hex()}')
            if version[0] == 0xEC:
                packed_file_size = int.from_bytes(
                    header[0x10 - version_length:0x18 - version_length],
                    byteorder='little',
                    signed=False,
                )
                # unpacked_file_size = int.from_bytes(header[0x18 - version_length:0x20 - version_length],
                #                                     byteorder='little', signed=False,
                #                                     )
                packed_file_name = header[0x30 - version_length:].decode('utf-16-le')
                packed_file_name = packed_file_name[:packed_file_name.find('\0')]  # ignore all 0's on the right
            else:
                packed_file_size = int.from_bytes(
                    header[0xC - version_length:0x10 - version_length],
                    byteorder='little',
                    signed=False,
                )
                # unpacked_file_size = int.from_bytes(header[0x10 - version_length:0x14 - version_length],
                #                                     byteorder='little', signed=False)
                packed_file_name_bytes = header[0x20 - version_length:]
                packed_file_name_bytes = packed_file_name_bytes[:packed_file_name_bytes.find(0x0)]
                packed_file_name = packed_file_name_bytes.decode('utf-8')
            log.debug(f'Version {version.hex()}. Packed file {packed_file_name} has size {packed_file_size} bytes.')

            # Read and process data
            archive_data = input_file.read(packed_file_size)
            if packed_file_name[-4:].upper() == '.LZW':
                archive_data = _decompress_lzw(archive_data)
                packed_file_name = packed_file_name[:-4]

            # Yield a series of tuples from the Generator
            yield UnpackedData(file_name=packed_file_name, unpacked_contents=archive_data)


def extract(input_file_path: Union[Path, str], output_path: Union[Path, str, None] = None) -> None:
    """
    Imports the data from a zar archive file and writes it as a regular directory.

    :param input_file_path: The path to zar-file.
    :param output_path: The path where the files should be saved. Default: the same as the input_file_path but
        without the extension.
    """
    # Make sure that the input arguments are both pathlib.Path-s
    if isinstance(input_file_path, str):
        input_file_path = Path(input_file_path)
    # By default, just drop the .zar extension
    if output_path is None:
        output_path = input_file_path.name
        if output_path.lower().endswith('.zar'):
            output_path = output_path[:-4]
        output_path = input_file_path.parent / output_path
    elif isinstance(output_path, str):
        output_path = Path(output_path)
    Path.mkdir(output_path, exist_ok=True, parents=True)
    log.debug(f'Extracting {input_file_path} to directory {output_path}/...')

    # Unpack and store the recovered data
    for unpacked_data in read(input_file_path):
        with open(output_path / unpacked_data.file_name, 'wb') as unpacked_file:
            unpacked_file.write(unpacked_data.unpacked_contents)

    log.info(f'Extracted {input_file_path} to directory {output_path}/.')


def repack(input_file_path: Union[Path, str], output_file_path: Union[Path, str, None] = None) -> None:
    """
    Imports the data from a zar archive file and writes it as a regular zip file.

    :param input_file_path: The path to zar-file.
    :param output_file_path: The path to the zip file. Default: the same as the input_file_path but with the extension
        changed to 'zip'.
    """
    # Make sure that the input arguments are both pathlib.Path-s
    if isinstance(input_file_path, str):
        input_file_path = Path(input_file_path)
    # By default, just change .zar to .zip
    if output_file_path is None:
        output_file_name = input_file_path.name
        if output_file_name.lower().endswith('.zar'):
            output_file_name = output_file_name[:-4]
        output_file_name += '.zip'
        output_file_path = input_file_path.parent / output_file_name
    else:
        if isinstance(output_file_path, str):
            if not output_file_path.lower().endswith('.zip'):
                output_file_path += '/' + input_file_path.name + '.zip'
            output_file_path = Path(output_file_path)
        elif isinstance(output_file_path, Path) and not output_file_path.name.lower().endswith('.zip'):
            output_file_path /= input_file_path.name + '.zip'
        Path.mkdir(output_file_path.parent, exist_ok=True, parents=True)
    log.debug(f'Converting {input_file_path} to zip archive {output_file_path}...')

    # Open the output archive and start storing unpacked files
    repack_directory = output_file_path.name[:-4]  # all but the extension
    with zipfile.ZipFile(
        output_file_path,
        mode='a',
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=False,
        compresslevel=9,
    ) as archive_file:
        # Unpack and store the recovered data
        for unpacked_data in read(input_file_path):
            archive_file.writestr(f'{repack_directory}/{unpacked_data.file_name}', unpacked_data.unpacked_contents)

    log.info(f'Converted {input_file_path} to zip archive {output_file_path}.')


def unzar() -> int:
    """Function that can be called as a script."""
    # Parse the input arguments
    try:
        input_parser = argparse.ArgumentParser(
            description='Zemax archive files (.zar) unpacker and zipper.',
            usage="""
     Examples of usage:
    > unzar filename.zar
    > unzar filename.zar -z
    > unzar -i filename.zar -o filename.zip
    > unzar -i filename.zar -o subfolder/filename.zip
    > unzar -i filename.zar -vv
    > unzar -h
    """)
        input_parser.add_argument(
            '-v',
            '--verbosity',
            help='the path to the PNG image file describing the simulation structure',
            default=0,
            action='count',
        )
        input_parser.add_argument(
            '-V',
            '--version',
            help='show version information before proceeding',
            action='store_true',
        )
        input_parser.add_argument(
            '-i',
            '--input',
            type=str,
            nargs='*',
            help='the input archive',
        )
        input_parser.add_argument('-o', '--output', type=str, help='the output archive or directory')
        input_parser.add_argument('-z', '--zip', help='create an archive instead of a directory', action='store_true')
        input_parser.add_argument('archive_file_name', type=str, nargs='*', help='one or more names of input archives')
        input_args = input_parser.parse_args()

        log_levels = [logging.FATAL, logging.ERROR, logging.WARNING, logging.DEBUG]
        log.setLevel(log_levels[min(3, input_args.verbosity)])
        for _ in log.handlers:
            _.setLevel(log.level)

        log.debug(f'Parsed input arguments: {input_args}')

        version_message = (f'Running zmxtools unzar version {zmxtools.__version__}. ' +
                           'More info at: https://github.com/tttom/zmxtools'
                           )
        log.debug(version_message)
        log.debug(version_message)
        if input_args.version:
            sys.stdout.write(version_message)  # i.e. a print that lints

        all_input_files = input_args.input if input_args.input is not None else []
        all_input_files += input_args.archive_file_name if input_args.archive_file_name is not None else []
        if len(all_input_files) >= 1:
            log.info(f'Unpacking {len(all_input_files)} archives...')
            log.debug(f'File names {all_input_files}')
            for input_file in all_input_files:
                log.info(f'Loading {input_file}...')
                input_file_path = Path(input_file)

                if input_args.zip:
                    repack(input_file_path, input_args.output)
                else:
                    extract(input_file_path, input_args.output)
        else:
            log.error('No input zar archives specified.')
            input_parser.print_help()
            return 1
    except Exception as exc:
        log.fatal(exc)
        log.info(traceback.format_exc())
        # raise exc
        return -1
    # No error
    return 0


if __name__ == '__main__':
    sys.exit(unzar())
