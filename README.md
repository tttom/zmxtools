# ZmxTools

[![Build Status](https://github.com/tttom/zmxtools/workflows/test/badge.svg?branch=main)](https://github.com/tttom/zmxtools/actions?query=workflow%3Atest)
[![Documentation Status](https://readthedocs.org/projects/zmxtools/badge/?version=latest)](https://readthedocs.org/projects/zmxtools)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zmxtools)](https://www.python.org/downloads)
[![PyPI - License](https://img.shields.io/pypi/l/zmxtools)](https://opensource.org/licenses/AGPL-3.0)
[![PyPI](https://img.shields.io/pypi/v/zmxtools?label=version&color=808000)](https://github.com/tttom/ZmxTools/tree/master/python)
[![PyPI - Status](https://img.shields.io/pypi/status/zmxtools)](https://pypi.org/project/zmxtools/tree/master/python)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/zmxtools?label=python%20wheel)](https://pypi.org/project/zmxtools/#files)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/zmxtools)](https://pypi.org/project/zmxtools/)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/tttom/ZmxTools)](https://github.com/tttom/ZmxTools)
[![GitHub last commit](https://img.shields.io/github/last-commit/tttom/ZmxTools)](https://github.com/tttom/ZmxTools)
[![Libraries.io dependency status for latest release](https://img.shields.io/librariesio/release/pypi/zmxtools)](https://libraries.io/pypi/zmxtools)
[![wemake-python-styleguide](https://img.shields.io/badge/style-wemake-000000.svg)](https://github.com/wemake-services/wemake-python-styleguide)
[![codecov](https://codecov.io/gh/tttom/zmxtools/branch/main/graph/badge.svg)](https://codecov.io/gh/tttom/zmxtools)

### A toolkit to read Zemax files.

This provides a command line tool to unpack Zemax ZAR archives into its constituent files, or repack it as standard zip 
files. A Python API is provided to integrate this into your own projects. Further processing of the archive's contents, 
such as ZMX optical design or AGF glass files, is now also possible with the Python API; however, support is limited and 
experimental. You may find the [list of related software](#related-software) below may be complimentary for further parsing.

## Features
- Unpack a Zemax OpticStudio® Archive ZAR file using the `unzar` command.
- Repack a ZAR file as a standard zip file using the `unzar -z` command.
- Integrate into your project as a pure Python 3 library.
- Parse ZMX and AGF files to Python classes for further processing.
- Fully typed with annotations and checked with mypy, [PEP561 compatible](https://www.python.org/dev/peps/pep-0561/)

## Installation
### Prerequisites
- Python 3.11 or higher
- pip, the Python package manager

To install `zmxtools`, just run the following command in a command shell:
```bash
pip install zmxtools
```

## Usage
This package can be used directly from a terminal shell or from your own Python code.
Example files can be found on manufacturer's sites such as [Thorlabs Inc](https://www.thorlabs.com).

### Command line shell
The command `unzar` is added to the path upon installation. It permits the extraction of the zar-file to a sub-directory
as well as its conversion to a standard zip-file. For example, extracting to the sub-directory `mylens` is done using 
```console
unzar mylens.zar
```
Repacking the same zar-archive as a standard zip-archive `mylens.zip` is done with:
```console
unzar mylens.zar -z
```
Multiple input files and an alternative output directory can be specified: 
```console
unzar -i *.zar -o some/where/else/
```
Find out more information and alternative options using:
```console
unzar -h
```

### As a Python library
Extraction and repacking can be done programmatically as follows:
```python
from zmxtools import zar

zar.extract('mylens.zar')
zar.repack('mylens.zar')
optical_design = zar.load('mylens.zar')[0]
```
Note that you may use Python `pathlib.Path` objects instead of strings.

Further processing of its contents is possible using the `zmx` sub-module.
The contents of the files can then be extracted as follows:
```python
from zmxtools import zmx

optical_design = zmx.ZmxOpticalDesign.from_file('mylens.zmx', 
                                                ['schott.agf', 'infrared.agf'])
wavelength = optical_design.source.wavelengths.ravel()[0]
for surface in optical_design.surfaces:
  print(surface.type + (' STOP' if surface.stop else '') + ' surface with ' +
        (f'{1 / surface.curvature / 1e-3:0.3f} mm' if surface.curvature != 0.0 else '∞') +
        f' radius of curvature and {surface.radius * 2 / 1e-3:0.3f} mm aperture.')
  if surface.distance != 0.0:
    print(f'  {surface.distance / 1e-3:0.3f} mm spacing with {surface.material.name}' +
          f' n={surface.material.complex_refractive_index(wavelength=wavelength)}' +
          f' @ {wavelength / 1e-9:0.1f}nm.')
```
Note that `surface.material` are glass objects that model the optical properties over a range of wavelengths, not just 
those of the optical model.

## Online documentation
The latest version of the API Documentation is published on https://zmxtools.readthedocs.io/.
The documentation is generated automatically in the [docs/ directory](docs) from the source code. 

## Contributing to the source code
The complete source code can be found on [github: https://github.com/tttom/zmxtools](https://github.com/tttom/zmxtools).
Check out [Contributing](CONTRIBUTING.md) for details.

## License
This code is distributed under the
[agpl3: GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html)

## Credits
- [Wouter Vermaelen](https://github.com/m9710797) for decoding the ZAR header and finding LZW compressed contents.
- [Bertrand Bordage](https://github.com/BertrandBordage) for sharing this [gist](https://gist.github.com/BertrandBordage/611a915e034c47aa5d38911fc0bc7df9).
- This project was generated with [`wemake-python-package`](https://github.com/wemake-services/wemake-python-package). Current template version is: [cfbc9ea21c725ba5b14c33c1f52d886cfde94416](https://github.com/wemake-services/wemake-python-package/tree/cfbc9ea21c725ba5b14c33c1f52d886cfde94416). See what is [updated](https://github.com/wemake-services/wemake-python-package/compare/cfbc9ea21c725ba5b14c33c1f52d886cfde94416...master) since then.

## Related Software
- [Optical ToolKit](https://github.com/draustin/otk) reads Zemax .zmx files.
- [RayTracing](https://github.com/DCC-Lab/RayTracing) reads Zemax .zmx files.
- [Zemax Glass](https://github.com/nzhagen/zemaxglass) reads Zemax .agf files.
- [RayOptics](https://github.com/mjhoptics/ray-optics) reads Zemax .zmx and CODE-V .seq files.
- [RayOpt](https://github.com/quartiq/rayopt) reads Zemax .zmx as well as OSLO files.
- [OpticsPy](https://github.com/Sterncat/opticspy) does not read Zemax .zmx files but reads CODE-V .seq files and
  glass information from data downloaded from https://www.refractiveindex.info/.
- [OpticalGlass](https://github.com/mjhoptics/opticalglass) reads glass manufacturer Excel sheets.
