# Version history

We follow [Semantic Versions](https://semver.org/).


## Releases 0.2

### Version 0.2.0
- Breaking changes to the API:
  - `zar.load` is renamed to `zar.unpack` and yields `BytesFile` objects instead
  - all functions now take file-like objects
- Added new module to interpret `.zmx` files


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
