# Security policy

## Reporting a vulnerability

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/DeharengOlivier/rag-engine/security/advisories/new)
rather than opening a public issue. Include what you did, what happened, and
what you expected. A proof of concept helps but is not required.

Expect an acknowledgement within a week. If the report is valid, you will be
told what the fix is and when it ships, and you will be credited in the advisory
unless you prefer otherwise.

## Scope

This is a library and a CLI. It has no server, no authentication and no
multi-tenancy, so the usual web attack surface does not apply. What does:

- **Anything that lets a document control the process.** Ingestion reads files
  the operator points it at. A crafted document should never do more than
  produce bad chunks.
- **Anything that leaks a corpus or a key.** The engine deliberately keeps user
  content out of its logs, keeps API keys out of the configuration object, and
  redacts PII at ingestion when an anonymizer is selected. A path that defeats
  any of those is in scope.
- **Anything that makes an outbound call unbounded.** Every API-backed call
  carries a timeout and a finite retry budget; a way around that is in scope.
- **The supply chain.** A dependency with a known vulnerability, or a way to
  make the package execute code at install or import time.

Out of scope: the behaviour of the optional third-party backends themselves
(sentence-transformers, presidio, the provider SDKs), and the quality of the
answers a language model generates.

## What the project does on its side

- Secret scanning and push protection are enabled on the repository, and no
  secret has ever been committed to it.
- `pip-audit` and `bandit` run in CI on every push and pull request.
- API keys are read from the environment at call time only. They are never read
  into the configuration object, never logged, and never written to the index.
- The engine runs fully offline by default: no network call happens unless an
  API-backed provider is explicitly selected.
