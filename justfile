[group('dev')]
run *ARGS:
  uv run deckle {{ARGS}}

[group('dev')]
test *ARGS:
  uv run pytest {{ARGS}}

[group('dev')]
lint:
  #!/bin/bash
  set -euo pipefail

  uv run ruff check src tests
  uv run ruff format --check src tests

[group('dev')]
fmt:
  uv run ruff format src tests
