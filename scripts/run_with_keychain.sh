#!/usr/bin/env bash
# run_with_keychain.sh — Run evoforge with API key from macOS Keychain
#
# Setup (one-time):
#   security add-generic-password -s evoforge -a anthropic-api-key -w "<YOUR_KEY>"
#
# Usage:
#   bash scripts/run_with_keychain.sh
#   EVOFORGE_SERVICE=my-svc EVOFORGE_ACCOUNT=my-acct bash scripts/run_with_keychain.sh

set -euo pipefail

SERVICE="${EVOFORGE_SERVICE:-evoforge}"
ACCOUNT="${EVOFORGE_ACCOUNT:-anthropic-api-key}"

cleanup() {
    unset ANTHROPIC_API_KEY 2>/dev/null || true
}
trap cleanup EXIT

ANTHROPIC_API_KEY="$(security find-generic-password -s "$SERVICE" -a "$ACCOUNT" -w 2>/dev/null)" || {
    echo "ERROR: Could not retrieve API key from Keychain." >&2
    echo "Store it first:" >&2
    echo "  security add-generic-password -s $SERVICE -a $ACCOUNT -w <YOUR_KEY>" >&2
    exit 1
}
export ANTHROPIC_API_KEY

exec uv run python scripts/run.py --config configs/lean_default.toml --max-generations 200 "$@"
