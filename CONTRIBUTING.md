# How to contribute


## Dependencies

We use [poetry](https://github.com/python-poetry/poetry) to manage the dependencies.

To install them you would need to run `install` command:

```bash
poetry install
```

To activate your `virtualenv` run `poetry shell`.


## One magic command

Run `make test` to run everything we have!


## Tests

We use `pytest` and `flake8` for quality control.
We also use [wemake_python_styleguide](https://github.com/wemake-services/wemake-python-styleguide) to enforce the code quality.

To run all tests:

```bash
pytest
```

To run linting:

```bash
flake8 .
```
Keep in mind: default virtual environment folder excluded by flake8 style checking is `.venv`.
If you want to customize this parameter, you should do this in `setup.cfg`.
These steps are mandatory during the CI.


## Type checks

We use `mypy` to run type checks on our code.
To use it:

```bash
mypy zmxtools tests/**/*.py
```

This step is mandatory during the CI.


## Submitting your code

We use [trunk based](https://trunkbaseddevelopment.com/)
development (we also sometimes call it `wemake-git-flow`).

What the point of this method?

1. We use protected `main` branch,
   so the only way to push your code is via pull request
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

1. Add tests for code changes
2. Include or update the inline documentation.
3. Update [README.md](README.md) as necessary.
4. Increase the version number in `pyproject.toml` under the `[tool.poetry]` header
5. Update [CHANGELOG.md](CHANGELOG.md) with a quick summary of your changes
6. Run `make test` to:
   1. check code behaviour with `pytest`
   2. ensure that types are correct with `mypy`
   3. enforce code style using `flake8`
   4. verify the documentation with `doc8`
7. Run `make clean html` in the `docs` folder and check for documentation errors. 
8. Commit the changes with a descriptive message.
9. Tag the release with the version number in `pyproject.toml` using `git tag 0.0.0`.
10. Raise a pull-request.


## Other help

You can contribute by spreading a word about this library.
It would also be a huge contribution to write
a short article on how you are using this project.
You can also share your best practices with us.
