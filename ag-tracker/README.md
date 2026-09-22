# Antigravity Quota & Token Burn Monitor (AG Tracker)

An automated telemetry system that periodically tracks Antigravity quota consumption, token burn rate, and running CLI processes every 15 minutes.

---

## Why Quota Depletes on the Google AI Ultra (20x) Plan

If you have the **Google AI Ultra** plan ($250/mo subscription) with the **20x higher quota** benefit, your account is properly recognized by Antigravity under the **Antigravity** tier (the top tier).

Here is why your quota may look like it is depleting quickly:

1. **Rolling 5-Hour Smoothing Window vs. Monthly Hard Cap**:
   - The quota shown in `agl` under `Gemini 5h` is a **rolling sliding 5-hour window** designed to smooth peak aggregate demand across users.
   - It is **not** your monthly allowance.
   - When the 5h bar reads **93% left**, you have only used **7%** of your 5-hour burst window. It continuously restores headroom as older requests roll out of the 5-hour window.
   - Your weekly allowance (`Gemini 7d`) is at **98.8% left** (virtually untouched).
   - Claude and GPT quotas are at **100% left**.

2. **Concurrency Multiplier (13–14 Active CLI Instances)**:
   - There are currently 13–14 simultaneous `agy` CLI processes running across `astralane-quant`, `whisper-typer`, and `trailblazer`.
   - Active multi-step sessions like `tb-data` (3,849 steps), `sept-attrib` (1,762 steps), and `Fix TTS Hook Issues` (1,408 steps) execute extensive tool loops and maintain large conversation contexts.

3. **High Thinking Budget**:
   - The model is set to **Gemini 3.8 Flash (High)**.
   - The **High** thinking budget allocates thousands of internal reasoning tokens per response turn. Multiplied across 14 concurrent sessions, this easily burns 300,000–600,000 tokens every few minutes (~31M tokens used today alone).

---

## Directory Structure

```text
whisper-typer/ag-tracker/
├── run_tracker.sh            # Cron entry wrapper with lockfile & DBus resolution
├── tracker.py                # Core Python collector, SQLite/JSONL writer & HTML generator
├── index.html                # Self-contained, responsive dark-mode dashboard
├── .gitignore                # Ignores local databases, logs, and lockfiles
├── README.md                 # Documentation and operational guide
└── data/
    ├── usage_history.db      # SQLite database with indexed snapshots & process tables
    ├── usage_history.jsonl   # Append-only JSON Lines history file
    ├── latest.json           # Fast-read JSON cache of the most recent snapshot
    ├── tracker.lock          # Flock mutex ensuring no duplicate cron runs
    └── tracker.log           # Timestamped execution log
```

---

## How to View the Dashboard

### Option 1: Open Directly in Your Browser (Zero Server Required)
The generated `index.html` embeds the latest snapshot and historical records directly into the page payload. You can open it instantly in Chrome, Firefox, or Brave without running a local web server:

```bash
xdg-open file:///home/mizzlr/whisper-typer/ag-tracker/index.html
```

Or open `file:///home/mizzlr/whisper-typer/ag-tracker/index.html` in your browser URL bar.

### Option 2: Run Built-In Local Web Server with Live Auto-Reload
If you prefer a persistent browser tab that automatically refreshes every 30 seconds:

```bash
python3 /home/mizzlr/whisper-typer/ag-tracker/tracker.py serve --port 8989
```
Then visit: `http://localhost:8989/`

*(Note: The server includes automatic port detection—if port 8989 is ever in use, it will automatically bind to the next available free port without failing).*

---

## CLI Commands

### 1. Check Current Status
```bash
python3 /home/mizzlr/whisper-typer/ag-tracker/tracker.py status
```
Prints account tier, 5h quota, 7d quota, today's tokens, 15m delta, and active instances count.

### 2. Manually Trigger a Snapshot
```bash
/home/mizzlr/whisper-typer/ag-tracker/run_tracker.sh
```

### 3. Query the SQLite Database via CLI
Since the host does not have `sqlite3` installed as a system binary, `tracker.py` includes a built-in SQL interface:

```bash
# View database metrics, tables, and row counts
python3 /home/mizzlr/whisper-typer/ag-tracker/tracker.py db-info

# Query the hourly burn view
python3 /home/mizzlr/whisper-typer/ag-tracker/tracker.py sql "SELECT * FROM v_hourly_burn"

# Query active instance runtimes
python3 /home/mizzlr/whisper-typer/ag-tracker/tracker.py sql "SELECT * FROM v_instance_activity LIMIT 10"

# Run any custom SQL query
python3 /home/mizzlr/whisper-typer/ag-tracker/tracker.py sql "SELECT timestamp, gemini_5h_pct, tokens_delta FROM snapshots ORDER BY id DESC LIMIT 5"
```

### 4. Export to CSV
```bash
python3 /home/mizzlr/whisper-typer/ag-tracker/tracker.py export-csv --table snapshots
python3 /home/mizzlr/whisper-typer/ag-tracker/tracker.py export-csv --table v_hourly_burn
```

### 5. Regenerate HTML Dashboard
```bash
python3 /home/mizzlr/whisper-typer/ag-tracker/tracker.py generate
```

---

## Cron Configuration

The cron job is registered in the user's crontab (`crontab -l`):
```cron
*/15 * * * * /home/mizzlr/whisper-typer/ag-tracker/run_tracker.sh
```
It runs every 15 minutes, automatically detects the current user's DBus session bus at `/run/user/1000/bus` to access Secret Service OAuth credentials, writes records to both SQLite and JSONL, and updates `index.html`.

---

## Database Architecture (`usage_history.db`)

- **Concurrency & Reliability**: Configured in **WAL** (Write-Ahead Logging) mode with `busy_timeout = 5000` to allow non-blocking concurrent reads while the cron job writes snapshots.
- **Tables**:
  - **`snapshots`**: Every 15-minute telemetry point (timestamp, epoch, email, plan, 5h & 7d quotas, tokens, requests, deltas, instance count, raw JSON).
  - **`running_instances`**: Individual process states for each snapshot (PID, runtime, working directory, session title, arguments).
- **Pre-Built Analytics Views**:
  - **`v_hourly_burn`**: Hourly aggregation of tokens burned, requests made, min/max/avg 5h quota, and average instance count.
  - **`v_instance_activity`**: Summary of active sessions, repositories, total snapshots seen, and runtime.

