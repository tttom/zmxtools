# Version history

We follow [Semantic Versions](https://semver.org/).


## Releases 0.2

### Version 0.2.0
- Ability to interpret the lens model and materials as a Python object.
- Display the raytraced model on screen using the new oshow (or unzar -s) command.
- API changes: 
  - A new optical_design module interprets `.zmx` and `.agf` glass files. Key functionality:
    - An optic.OpticalDesign object is returned for each `.zmx` file, refering to `Material` objects.
    - A `material.MaterialLibrary` object is returned for each `.agf` file, containing a set of `Materials`s.
    - Added utility functions to represent and fit Zernike polynomials for the corresponding surface types.
  - Breaking changes to the API since versions 0.1:
    - `zar.load` is renamed to `zar.unpack` and yields a `BytesFile` objects instead
    - all functions now take file-like objects
    - requires at least Python 3.13.
- Fixed issue https://github.com/tttom/zmxtools/issues/2
- Development: 
  - Teplaced Poetry with `uv` + `hatchling`. Type checking with `ty`. Linting and formatting with `ruff`.
  - Test suite coverage >95%.
  - Updated the logging.
- Documentation:
  - apidocs converted to Google style

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
