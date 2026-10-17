#!/usr/bin/env bash
# Decide whether the paid evals can run. Fails closed: if the API keys are missing, the eval
# check goes red, so a run that never scored the app can never look like a pass.
# Inputs (env): ANTHROPIC_API_KEY, OPENAI_API_KEY, EVENT_NAME, HEAD_REPO (the pull request's
# head repository, empty for other events), GITHUB_REPOSITORY, GITHUB_OUTPUT.
set -euo pipefail

trimmed() { printf '%s' "${1:-}" | tr -d '[:space:]'; }

if [ -n "$(trimmed "${ANTHROPIC_API_KEY:-}")" ] && [ -n "$(trimmed "${OPENAI_API_KEY:-}")" ]; then
  echo "available=true" >> "$GITHUB_OUTPUT"
  exit 0
fi

if [ "${EVENT_NAME:-}" = "pull_request" ] && [ "${HEAD_REPO:-}" != "${GITHUB_REPOSITORY:-}" ]; then
  echo "::error::The evals did not run: GitHub does not share secrets with pull requests from a fork (${HEAD_REPO:-unknown}). Re-open this change from a branch in this repository so the quality gate can score it."
else
  echo "::error::The evals did not run: the ANTHROPIC_API_KEY and OPENAI_API_KEY secrets must both be set for a ${EVENT_NAME:-unknown} run."
fi
exit 1
