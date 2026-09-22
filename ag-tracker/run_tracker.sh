#!/usr/bin/env bash
# run_tracker.sh - Automated runner for Antigravity Usage Tracker
set -euo pipefail

TRACKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${TRACKER_DIR}/data"
LOG_FILE="${DATA_DIR}/tracker.log"
LOCK_FILE="${DATA_DIR}/tracker.lock"

mkdir -p "${DATA_DIR}"

# Ensure standard user environment variables for cron
export HOME="${HOME:-/home/mizzlr}"
export USER="${USER:-mizzlr}"
export PATH="/usr/local/bin:/usr/bin:/bin:/home/mizzlr/.local/bin:${PATH:-}"

# Auto-detect DBus session bus address for secret-service keyring access
if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ]]; then
    UID_NUM="$(id -u)"
    if [[ -e "/run/user/${UID_NUM}/bus" ]]; then
        export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${UID_NUM}/bus"
    fi
fi

# Acquire lock to prevent duplicate runs
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another tracker run is already in progress. Skipping." >> "${LOG_FILE}"
    exit 0
fi

# Run the tracker recording step
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Antigravity usage snapshot..." >> "${LOG_FILE}"
if /usr/bin/python3 "${TRACKER_DIR}/tracker.py" record >> "${LOG_FILE}" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Snapshot recorded successfully." >> "${LOG_FILE}"
else
    EXIT_CODE=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Tracker run failed with exit code ${EXIT_CODE}" >> "${LOG_FILE}"
    exit ${EXIT_CODE}
fi
