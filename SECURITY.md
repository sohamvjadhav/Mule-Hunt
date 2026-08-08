# Security Policy

Mule-Hunt is a research-oriented fraud-detection tool for **synthetic**
payment graphs. It does not process real financial data, but the code is
shipped on PyPI and may be pointed at production-shaped data, so we take
reports seriously.

## Reporting a vulnerability

- Do **not** open a public issue for security problems.
- Report privately via GitHub's Security Advisories:
  <https://github.com/sohamvjadhav/Mule-Hunt/security/advisories/new>
- Alternatively email `sohamvjadhav` at GitHub (link your advisory in the
  subject) if the advisory form is unavailable.

Please include: the affected version, the `upifraud` subcommand or API
endpoint involved, a minimal reproduction, and the impact.

## Scope

In scope: code execution, path traversal or arbitrary file reads in the CLI
or FastAPI service, unsafe deserialization (e.g. `torch.load` / `joblib` on
untrusted checkpoints), SSRF or data leakage via the API.

Out of scope: model-quality limitations (false positives/negatives), missing
hardening in a research demo, or issues in upstream dependencies.

## Response

Acknowledgment within 3 business days; a fix and a patch release as soon as a
reproduction is confirmed. Disclosure happens after a public release, with
credit unless anonymity is requested.

## Supported versions

Security fixes land on the latest release only.
