# Readiness: native Mistral provider

Standards: workspace CODING-RULES / SECURITY-CHECKLIST **2026-10-05.1**.
Assessment: 2026-10-07, scope `feat/mistral-provider`, owner Olivier Dehareng.
Recheck on provider, SDK, credential, prompt or release changes.

## Profile

L2 reusable Python library and local CLI; no hosted service, accounts, database,
remote tools or automatic deployments. Users own their local document index.
This change adds optional outbound calls to the fixed Mistral HTTPS endpoint,
using an operator-supplied credential. Tests use synthetic documents and no live
credentials. Installing or choosing the provider is explicit; default stays offline.
No new tenant authorization, inbound API, payment processing or persistent store.

## Engineering evidence

| Control | Status | Evidence / follow-up |
|---|---|---|
| Lint, formatting, strict types | PASS | ruff check, ruff format --check, mypy; CI quality job |
| Coverage and contract tests | PASS | pytest --cov --cov-fail-under=97; fake SDK and real SDK with MockTransport; offline CI matrix plus dedicated SDK job |
| Protected main | PASS | GitHub branch protection read 2026-10-07: six existing required checks, strict updates, linear history, no force pushes/deletion |
| Independent review | PASS | Independent agent review of initial change and CLI fix; one P2 corrected, 42 relevant tests rerun, no remaining code findings; see PR #3 |
| CI isolation | PASS | GitHub-hosted ubuntu-latest, read-only contents, no secrets in deterministic tests |
| Native SDK dependency audit | PASS | mistralai 2.10.1, pip-audit --skip-editable: no known vulnerabilities |
| SDK version reproducibility | PARTIAL | Compatible range >=2.10.1,<3; contract job detects drift; not a frozen application deployment |

## Security evidence

This scoped register was missing before this change. It does not certify the
entire pre-existing library or a consumer's deployment. Runtime credential
permissions, retention, revocation, quotas and provider data-processing approval
remain operator obligations; no production configuration was inspected.

| Control | Priority | Status | Evidence / applicability |
|---|---|---|---|
| SEC-GOV-01 | P0 | NOT_VERIFIED | Scoped evidence recorded here; full library release assessment remains outstanding |
| SEC-17-018 | P0 | PASS (test scope) | Synthetic fixtures only; README states selected question/chunks leave the machine; no real user documents sent |
| SEC-17-013 | P0 | PASS (changed code) | test_native_request_and_safe_logs; provider HTTP error bodies suppressed |
| SEC-18-004 | P0 | PASS (changed code) | 1024 output tokens/attempt, <=5 retries, SDK retries disabled; deterministic exhaustion tests |
| SEC-19-005 | P0 | PASS (changed code) | Finite per-request timeout; explicit transient statuses only, jitter, capped Retry-After; ambiguous transport errors not retried |
| SEC-22-INVARIANTS | P0 | PASS (changed code) | Missing dependency/key, permanent/transient errors, empty output, retry cap, safe logs, pipeline pre-generation refusal |
| SEC-22-LLM | P0 | NOT_VERIFIED | Deterministic tests passed; real-model evaluation not run: no MISTRAL_API_KEY available. No claim of factual grounding or injection resistance |
| Hosted auth/server/DB/backup controls | — | NOT_APPLICABLE (change scope) | No such new surfaces; consumer deployments need their own assessment |

## Release gate

**BLOCKED for activation/merge**: SEC-22-LLM live evidence and SEC-GOV-01 full
release assessment are incomplete. Publish as a draft PR; do not present this as
a released provider on the profile README yet. Existing branch protection is
unchanged; this register is evidence, not an automated security enforcement gate.

The opt-in `.github/workflows/mistral-live-eval.yml` runs versioned synthetic cases
in `scripts/evaluate_mistral.py`: grounded answer with citation, missing evidence,
and an instruction embedded in a retrieved document. Maximum three calls, no
retries, 15-second request timeout, 3072 total output tokens. It fails if the key
is absent or any rubric fails. These are a small sample, not a universal guarantee.
Run on the candidate branch with a repository MISTRAL_API_KEY secret; attach the
run and candidate SHA to the PR. A workflow_dispatch workflow may first need to
exist on the default branch; alternatively run the same script in a trusted local
environment on this candidate. Do not put the credential in an issue or README.
