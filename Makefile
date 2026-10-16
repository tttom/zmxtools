SHELL:=/usr/bin/env bash

.PHONY: lint
lint:
	uv run ty check src/zmxtools tests
	uv run ruff check src/zmxtools tests
	uv run ruff format --check src/zmxtools tests
	uv run doc8 README.md CONTRIBUTING.md CHANGELOG.md docs

.PHONY: unit
unit:
	uv run pytest

.PHONY: package
package:
	uv build
	uv run python -c "import zmxtools; print(zmxtools.__version__)"

.PHONY: test
test: lint package unit
