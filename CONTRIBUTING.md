# How to contribute

## Code structure

`zmxtools` is a pure-Python toolkit that reads Zemax OpticStudio files. It is organized
into a small number of cohesive sub-packages.

```
src/zmxtools/
  __init__.py               Package version + shared colored logging setup (log, console_log_handler)
  cli.py                    The `unzar` command-line entry point (argparse wrapper)
  gui/
    - __init__.py           The Python code to serve the web-based gui.
    - oshow_template.py     The html template for the graphical user interface.
    - oshow.css             The colors and styles for the interface.
    - oshow.js              The javascript code for the interactivity of the display.
  parser.py                 OrderedCommandDict / Command: an ordered, indexable parse tree
  zar.py                    ZAR archive reader: unpack / extract / repack / load
  zmx.py                    ZMX reader: ZmxOrderedCommandDict, ZmxOpticalDesign, ZmxSurface, ZmxSource
  agf.py                    AGF glass-catalog reader: AgfMaterialLibrary + dispersion formulas
  utils/
    __init__.py             Shared constants (const_c) and re-exports
    array.py                NumPy typing helpers (array_like / array_type) and array utilities
    io.py                   BytesFile + FileLike / PathLike type aliases
    polar.py                cart2pol / pol2cart coordinate conversion
    script.py               Unicode superscript / subscript formatting (sup / sub)
    factorial_fraction.py   Overflow-safe factorial ratios
    multivariate.py         Generic (Laurent) multivariate polynomials + PolarPolynomial
    zernike.py              Zernike basis polynomials, index conversions, and least-squares fit
  optical_design/
    __init__.py             Sub-package logger
    header.py               Abstract Element / Medium base classes (import-cycle breakers)
    geometry.py             Transforms: Translation, Scaling, Rotation, Quaternion, Spherical...
    light.py                Wavefront and LightPath ray bookkeeping
    material.py             Material model: Vacuum, air, ModelGlass, Polynomial/Function materials
    medium.py               HomogeneousMedium / InhomogeneousMedium
    surface.py              Interface, SnellInterface, Aperture, Surface
    optic.py                CompoundElement, Detector, SurfaceDetector, OpticalDesign
    source.py               Light Source base class
```

The data flow is: `zar.unpack` yields in-memory `BytesFile`s; `zmx.ZmxOpticalDesign`
and `agf.AgfMaterialLibrary` parse `.zmx` / `.agf` text through the shared
`parser.OrderedCommandDict`; the parsed surfaces reference `optical_design.material`
objects. `utils` holds the numeric helpers (Zernike, polynomials, array typing) that
the optical-design layer builds on.

The public API is intentionally small: `zmxtools.zar` (archive handling),
`zmxtools.zmx` (optical designs), and `zmxtools.agf` (glass catalogs).


## Dependencies

We use [uv](https://docs.astral.sh/uv/) to manage the dependencies and the virtual
environment. All development commands below are run through `uv run ...` so that they
always use the pinned, locked versions from `uv.lock`.

To install the development environment (including the test, lint, and type-check
tooling), run:

```bash
uv sync
```

The package itself is built with `hatchling` and configured in `pyproject.toml`
(PEP 621 `[project]` table); there is no Poetry, `setup.py`, or `setup.cfg`.

On NixOS (or any Nix-enabled machine), `nix develop` enters a shell with Python 3.13
and `uv`; entering the shell creates `.venv` with `uv sync` if it does not exist and
updates it from `uv.lock` otherwise. Run `uv-update` inside the shell to upgrade all
locked packages. `nix run .#python` gives a Python interpreter with the runtime
dependencies preinstalled.


## One magic command

Run `make test` to run all quality checks in one go: type checking, linting,
formatting, packaging, and the unit tests with coverage.


## Tests

We use `pytest` (with `pytest-cov`) for quality control:

```bash
uv run pytest
```

The test suite enforces a minimum of 95% statement-and-branch coverage
(`--cov-fail-under=95` in `pyproject.toml`). Tests live in the `tests/` directory,
mirroring the package structure, and real-world sample files live in `tests/data/`.
Docstring examples are executed as doctests (`--doctest-modules`).

To run a single test file:

```bash
uv run pytest tests/zmx/test_zmx.py
```


## Linting and formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check zmxtools tests
uv run ruff format --check zmxtools tests
```

Ruff enforces the `pydocstyle` conventions with the **Google** docstring style
(`[tool.ruff.lint.pydocstyle] convention = "google"`) and its default string style
(double quotes; single quotes only when the content contains a double quote).
Docstrings document arguments under an `Args:` section and results under a
`Returns:` section, **without types**: the types live in the Python annotations.

```python
def to_length(vector, length, value=0):
    """
    Pads a 1D vector to a given length, crops it if it is too long.

    Args:
        vector: The input vector.
        length: The length of the output vector.
        value: The value to use for padding.

    Returns:
        An output vector of the specified length.
    """
```


## Type checks

We use [ty](https://github.com/astral-sh/ty) to run type checks on our code:

```bash
uv run ty check zmxtools tests
```

This step is mandatory during continuous integration. Public functions must be fully
annotated; numpy arrays are typed with the `array_type` / `array_like` helpers from
`zmxtools.utils.array`. Suppress a check only with a targeted
`# ty: ignore[rule-code]` comment when there is no better option.


## Submitting your code

We use [trunk based](https://trunkbaseddevelopment.com/) development (we also sometimes call it `wemake-git-flow`).

What the point of this method?

1. We use protected `main` branch,
   so the only way to push your code is via a pull request
2. We use issue branches: to implement a new feature or to fix a bug
   create a new branch named `issue-$TASKNUMBER`
3. Then create a pull request to `main` branch
4. We use `git tag`s to make releases, so we can track what has changed
   since the latest release

So, this way we achieve an easy and scalable development process
which frees us from merging hell and long-living branches.

In this method, the latest version of the app is always in the `main` branch.

### Before submitting

Before submitting your code:

1. Add tests for code changes; keep the coverage gate at 95% green.
2. Include or update the inline documentation (Google-style docstrings, no types).
3. Update [README.md](README.md) as necessary.
4. Increase the version number in `pyproject.toml` under the `[project]` header
5. Update the change log, [CHANGES.md](CHANGES.md), with a brief summary of your changes
6. Run `make test` to:
   1. check code behaviour with `pytest`
   2. ensure that types are correct with `ty`
   3. enforce code style using `ruff`
   4. verify the documentation with `doc8`
7. Run `make html` in the `docs` folder and check for documentation errors.
8. Commit the changes with a descriptive message.
9. Tag the release with the version number in `pyproject.toml` using `git tag 0.0.0`.
10. Raise a pull-request.


## Other help

You can contribute by spreading a word about this library.