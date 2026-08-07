[group('dev')]
run *ARGS:
  uv run deckle {{ARGS}}

# Tests that do not need a scan.
[group('dev')]
test *ARGS:
  uv run pytest -m "not scans" {{ARGS}}

# Tests that run on scans
[group('dev')]
test-scans *ARGS:
  uv run pytest -m scans {{ARGS}}

[group('dev')]
lint:
  #!/bin/bash
  set -euo pipefail

  uv run ruff check src tests
  uv run ruff format --check src tests

[group('dev')]
fmt:
  uv run ruff format src tests
