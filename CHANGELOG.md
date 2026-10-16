# Version history

We follow [Semantic Versions](https://semver.org/).


## Releases 0.3

### Version 0.3.0
- Packaging: replaced Poetry with `uv` + `hatchling`; type checking with `ty`; linting
  and formatting with `ruff`. Requires Python 3.13.
- Docstrings converted to Google style (types live in the annotations).
- Bug fixes:
  - `Quaternion.__mul__` j-component sign; `Quaternion.__str__` formatting;
    `Quaternion.exp`/`__pow__` at zero rotation.
  - `Rotation.matrix` returned the transposed matrix.
  - `SphericalTransform.point` and `InverseSphericalTransform.point` used `np.stack`
    incorrectly and reconstructed points with the wrong sign.
  - `CompoundTransform` did not implement `homogeneous`, breaking composed transforms.
  - `SnellInterface.refract_into` crashed for single rays; `SphericalSurface.distance`
    picked the far intersection root.
  - `CompoundElement.__repr__` crashed; `InhomogeneousMedium` lost its transform.
  - `zar.extract`/`repack` crashed on string paths and streams; `repack` output zip
    pairing fixed.
  - `parser.sort_and_merge` crashed on non-integer command arguments.
  - `material.PolynomialMaterial` default adjust function returned the function
    instead of calling it; `MaterialLibrary` regex `in`-checks.
  - `zmx.zernike_sag` with empty indices; `ZmxSurface.conic_constant` scalar.
  - `zernike.Polynomial.__truediv__` no longer mutates in place; removed dead
    `__idiv__`.
  - `tests/zar.check_zip_and_remove` asserted a bound method (always true).
- Renamed `ZmxSurface.distance` (spacing) to `ZmxSurface.spacing` and `reflect` to
  `is_mirror` to stop shadowing the `Element.distance`/`Element.reflect` methods.
- Added the `oshow` command-line tool: loads a `.zar`/`.zmx` design (with optional
  AGF libraries), builds a JSON scene (stop-centered axes, semi-transparent lens
  meshes with opaque rim markings, traced ray bundle, image plane), and serves it as
  an interactive Three.js page on a local `http.server` with live reload; supports
  `--save-scene`, `--watch`, `--rays`, `--port`, and standalone `--output` HTML.
  The viewer exits by default once the model has been displayed; `--keep` or
  `--watch` keep the server running. A ◐ button toggles light/dark mode (OS
  color scheme by default, dark fallback); the aperture stop ring is dark silver.
- `oshow` also loads the output of `unzar`: a directory with the extracted `.zmx`
  and `.agf` files, or the `.zip` archive written by `unzar -z`.
- Added `unzar --show` (`-s`) to display the unpacked module directly in `oshow`.
- When the source of the image or the image is at infinity, `oshow` draws the rays
  as parallel rays sticking out beyond the first or final surface by the maximum of
  the rest of the path length and five times the aperture diameter.
- Simplifications: shared `Transformed` base for `Interface`/`Aperture`, shared
  path-coercion in `zar`, shared regex helper in `material`.
- Test suite extended to >95% coverage (215 tests).


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
