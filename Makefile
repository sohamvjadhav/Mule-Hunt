PYTHON := .venv/bin/python
UPIFRAUD := .venv/bin/upifraud

.PHONY: help lint test demo build release-check mcp

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

lint: ## run ruff over src and tests
	.venv/bin/ruff check src tests

test: ## run the test suite
	$(PYTHON) -m pytest -q -p no:warnings

demo: ## end-to-end smoke run on the built-in toy graph
	$(UPIFRAUD) demo --toy

mcp: ## start the MCP investigation server (stdio) for models/
	$(UPIFRAUD) mcp --out-dir models

build: ## build sdist + wheel
	$(PYTHON) -m build

release-check: ## fail if pyproject version differs from the newest tag
	@VERSION=$$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml); \
	LATEST=$$(git tag -l 'v*' | sort -V | tail -1 | sed 's/^v//'); \
	if [ "$$VERSION" != "$$LATEST" ]; then \
		echo "pyproject version $$VERSION vs latest tag v$$LATEST"; exit 1; fi; \
	echo "version $$VERSION matches latest tag"
