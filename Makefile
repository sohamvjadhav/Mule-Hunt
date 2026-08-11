PYTHON := .venv/bin/python
UPIFRAUD := .venv/bin/upifraud

.PHONY: help lint test demo build release-check release review mcp

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

release: ## verify lint+tests+version and print the release checklist
	@make lint && make test && make release-check; \
	echo; \
	echo "Release checklist (repo main):"; \
	echo "  1. gh pr create (version bump ships with the feature)"; \
	echo "  2. merge after the lint-and-test check passes"; \
	echo "  3. git tag v$$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml) && git push origin v$$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)"; \
	echo "  4. the publish workflow uploads to PyPI (trusted publishing)"

review: ## review the current diff with the Codex CLI (needs npm @openai/codex + OPENAI_API_KEY)
	@if ! command -v codex >/dev/null 2>&1; then \
		echo "codex CLI not found — install with: npm install -g @openai/codex"; exit 1; fi; \
	git diff HEAD -- . | codex exec --full-auto --skip-git-repo-check \
		"Review this diff for correctness bugs, security issues, honest reporting, and missing tests (follow AGENTS.md). Keep it under 250 words, bullets with file:line refs, finish with 'Verdict: LGTM' or 'Verdict: changes requested'."
