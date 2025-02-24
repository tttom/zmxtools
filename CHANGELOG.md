# Version history

We follow [Semantic Versions](https://semver.org/).


## Releases 0.2

### Version 0.2.0
- Breaking changes to the API:
  - `zar.load` is renamed to `zar.unpack` and yields `BytesFile` objects instead
  - all functions now take file-like objects
- Added optical_design module to interpret `.zmx` and `.agf` glass files. Key functionality:
  - An optic.OpticalDesign object is returned for each `.zmx` file, refering to `Material` objects.
  - A `material.MaterialLibrary` object is returned for each `.agf` file, containing a set of `Materials`s.
  - Added utility functions to represent and fit Zernike polynomials for the corresponding surface types.
- Fixed issue https://github.com/tttom/zmxtools/issues/2

## Releases 0.1
The first release series provides basic decompression and conversion tools,
both as command line tool and as a Python3 library.

### Version 0.1.5
- Made all non-standard dependencies optional.

### Version 0.1.4
- Security update of dependencies.
- Automated API-documentation generation.

### Version 0.1.3
- Refactored command-line interface code and the unit tests.

### Version 0.1.2
- Proper API docs, big fixes, and automated testing improved.

### Version 0.1.1
- Bug and documentation fixes.

## Version 0.1.0
- Initial release
