#!/usr/bin/env python3
"""Antigravity Usage Tracker.

Monitors Antigravity quota burn, token usage, and running instances every 15 minutes.
Maintains history in JSON Lines and SQLite DB, and generates an interactive HTML dashboard.
"""

from __future__ import annotations

import argparse
import datetime
import html
import http.server
import json
import os
import socketserver
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TRACKER_DIR = Path(__file__).resolve().parent
DATA_DIR = TRACKER_DIR / "data"
JSONL_FILE = DATA_DIR / "usage_history.jsonl"
DB_FILE = DATA_DIR / "usage_history.db"
LATEST_JSON_FILE = DATA_DIR / "latest.json"
HTML_FILE = TRACKER_DIR / "index.html"
AGL_BIN = Path.home() / ".local" / "bin" / "agl"
SUMMARIES_DB = Path.home() / ".gemini" / "antigravity-cli" / "conversation_summaries.db"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def init_db() -> sqlite3.Connection:
    ensure_dirs()
    con = sqlite3.connect(DB_FILE)
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA busy_timeout = 5000;")
    con.execute("PRAGMA synchronous = NORMAL;")
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            email TEXT,
            plan TEXT,
            model TEXT,
            project_id TEXT,
            gemini_5h_pct REAL,
            gemini_5h_reset TEXT,
            gemini_weekly_pct REAL,
            gemini_weekly_reset TEXT,
            claude_5h_pct REAL,
            claude_5h_reset TEXT,
            claude_weekly_pct REAL,
            claude_weekly_reset TEXT,
            today_tokens INTEGER,
            today_input_tokens INTEGER,
            today_output_tokens INTEGER,
            today_requests INTEGER,
            tokens_delta INTEGER DEFAULT 0,
            requests_delta INTEGER DEFAULT 0,
            gemini_5h_delta REAL DEFAULT 0.0,
            gemini_weekly_delta REAL DEFAULT 0.0,
            interval_seconds INTEGER DEFAULT 0,
            active_instances_count INTEGER DEFAULT 0,
            lifetime_tokens INTEGER DEFAULT 0,
            raw_json TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_epoch ON snapshots(epoch)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp)")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS running_instances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            pid INTEGER NOT NULL,
            elapsed_seconds INTEGER,
            cwd TEXT,
            conversation_id TEXT,
            session_name TEXT,
            args TEXT,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots(id) ON DELETE CASCADE
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_instances_snapshot_id ON running_instances(snapshot_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_instances_session ON running_instances(session_name)")

    # Analytical views for rapid insight
    cur.execute(
        """
        CREATE VIEW IF NOT EXISTS v_hourly_burn AS
        SELECT
            strftime('%Y-%m-%d %H:00', timestamp) AS hour_window,
            count(*) AS snapshot_count,
            sum(tokens_delta) AS total_tokens_burned,
            sum(requests_delta) AS total_requests,
            round(min(gemini_5h_pct), 2) AS min_5h_quota_pct,
            round(max(gemini_5h_pct), 2) AS max_5h_quota_pct,
            round(avg(gemini_5h_pct), 2) AS avg_5h_quota_pct,
            round(avg(active_instances_count), 1) AS avg_active_instances
        FROM snapshots
        GROUP BY hour_window
        ORDER BY hour_window DESC
        """
    )

    cur.execute(
        """
        CREATE VIEW IF NOT EXISTS v_instance_activity AS
        SELECT
            session_name,
            cwd,
            count(DISTINCT snapshot_id) AS snapshots_seen,
            min(elapsed_seconds) AS min_elapsed_seconds,
            max(elapsed_seconds) AS max_elapsed_seconds
        FROM running_instances
        GROUP BY session_name, cwd
        ORDER BY snapshots_seen DESC, max_elapsed_seconds DESC
        """
    )

    con.commit()
    return con


def get_environment() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HOME", str(Path.home()))
    path_dirs = ["/usr/local/bin", "/usr/bin", "/bin", str(Path.home() / ".local" / "bin")]
    curr_path = env.get("PATH", "")
    for p in path_dirs:
        if p not in curr_path:
            curr_path = f"{p}:{curr_path}" if curr_path else p
    env["PATH"] = curr_path

    if "DBUS_SESSION_BUS_ADDRESS" not in env:
        uid = os.getuid()
        bus_file = Path(f"/run/user/{uid}/bus")
        if bus_file.exists():
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_file}"
    return env


def fetch_agl_data() -> Dict[str, Any]:
    env = get_environment()
    agl_path = str(AGL_BIN) if AGL_BIN.exists() else "agl"
    try:
        proc = subprocess.run(
            [agl_path, "--json"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=30,
        )
        return json.loads(proc.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"agl command failed (code {e.returncode}): {e.stderr.strip()}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data from agl: {e}") from e


def fetch_active_conversations(limit: int = 15) -> List[Dict[str, Any]]:
    if not SUMMARIES_DB.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{SUMMARIES_DB}?mode=ro", uri=True)
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT conversation_id, title, step_count, last_modified_time, not_fully_idle, status, workspace_uris
            FROM conversation_summaries
            ORDER BY last_modified_time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        con.close()
        return [
            {
                "conversation_id": r[0] or "",
                "title": r[1] or "Untitled",
                "step_count": int(r[2] or 0),
                "last_modified": r[3] or "",
                "is_active": bool(r[4]),
                "status": r[5] or "",
                "workspace": r[6] or "",
            }
            for r in rows
        ]
    except Exception:
        return []


def get_previous_snapshot(con: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    cur = con.cursor()
    row = cur.execute(
        """
        SELECT id, epoch, today_tokens, today_requests, gemini_5h_pct, gemini_weekly_pct
        FROM snapshots
        ORDER BY epoch DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "epoch": row[1],
        "today_tokens": row[2],
        "today_requests": row[3],
        "gemini_5h_pct": row[4],
        "gemini_weekly_pct": row[5],
    }


def parse_quota_buckets(quotas_data: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "gemini_5h_pct": 100.0,
        "gemini_5h_reset": None,
        "gemini_weekly_pct": 100.0,
        "gemini_weekly_reset": None,
        "claude_5h_pct": 100.0,
        "claude_5h_reset": None,
        "claude_weekly_pct": 100.0,
        "claude_weekly_reset": None,
    }
    for group in quotas_data.get("groups", []):
        group_name = group.get("displayName", "")
        for bucket in group.get("buckets", []):
            bid = bucket.get("bucketId", "")
            window = bucket.get("window", "")
            frac = bucket.get("remainingFraction", 1.0)
            pct = round(frac * 100, 2)
            reset_time = bucket.get("resetTime")

            if "Gemini" in group_name or "gemini" in bid:
                if window == "5h" or "5h" in bid:
                    result["gemini_5h_pct"] = pct
                    result["gemini_5h_reset"] = reset_time
                elif window == "weekly" or "weekly" in bid:
                    result["gemini_weekly_pct"] = pct
                    result["gemini_weekly_reset"] = reset_time
            elif "Claude" in group_name or "3p" in bid:
                if window == "5h" or "5h" in bid:
                    result["claude_5h_pct"] = pct
                    result["claude_5h_reset"] = reset_time
                elif window == "weekly" or "weekly" in bid:
                    result["claude_weekly_pct"] = pct
                    result["claude_weekly_reset"] = reset_time
    return result


def record_snapshot() -> Dict[str, Any]:
    ensure_dirs()
    con = init_db()

    agl_data = fetch_agl_data()
    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
    epoch_now = int(now.timestamp())
    iso_now = now.isoformat()

    account = agl_data.get("account", {})
    email = account.get("email", "unknown")
    plan = account.get("plan", "Antigravity")
    model = account.get("model", "Gemini 3.8 Flash (High)")
    project_id = account.get("projectId", "default-cli-project")

    quotas = parse_quota_buckets(agl_data.get("quotas", {}))

    usage = agl_data.get("usage", {})
    windows = usage.get("windows", {})
    today_w = windows.get("today", {})
    today_tokens = int(today_w.get("total_tokens", 0))
    today_input = int(today_w.get("input_tokens", 0))
    today_output = int(today_w.get("output_tokens", 0))
    today_requests = int(today_w.get("requests", 0))

    summary = usage.get("summary", {})
    lifetime_tokens = int(summary.get("lifetimeTokens", 0))

    instances = agl_data.get("instances", [])
    active_instances_count = len(instances)

    # Compute deltas against previous snapshot
    prev = get_previous_snapshot(con)
    tokens_delta = 0
    requests_delta = 0
    gemini_5h_delta = 0.0
    gemini_weekly_delta = 0.0
    interval_seconds = 0

    if prev:
        interval_seconds = max(0, epoch_now - prev["epoch"])
        # If today_tokens < prev["today_tokens"], midnight day reset occurred
        if today_tokens >= prev["today_tokens"]:
            tokens_delta = today_tokens - prev["today_tokens"]
            requests_delta = today_requests - prev["today_requests"]
        else:
            tokens_delta = today_tokens
            requests_delta = today_requests

        gemini_5h_delta = round(quotas["gemini_5h_pct"] - prev["gemini_5h_pct"], 2)
        gemini_weekly_delta = round(quotas["gemini_weekly_pct"] - prev["gemini_weekly_pct"], 2)

    top_conversations = fetch_active_conversations(12)

    record = {
        "timestamp": iso_now,
        "epoch": epoch_now,
        "account": {
            "email": email,
            "plan": plan,
            "model": model,
            "projectId": project_id,
        },
        "quotas": quotas,
        "today_usage": {
            "total_tokens": today_tokens,
            "input_tokens": today_input,
            "output_tokens": today_output,
            "requests": today_requests,
        },
        "lifetime_tokens": lifetime_tokens,
        "deltas": {
            "tokens_delta": tokens_delta,
            "requests_delta": requests_delta,
            "gemini_5h_delta": gemini_5h_delta,
            "gemini_weekly_delta": gemini_weekly_delta,
            "interval_seconds": interval_seconds,
        },
        "instances": instances,
        "top_conversations": top_conversations,
        "raw_agl_usage": usage,
    }

    # Insert into SQLite
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO snapshots (
            timestamp, epoch, email, plan, model, project_id,
            gemini_5h_pct, gemini_5h_reset, gemini_weekly_pct, gemini_weekly_reset,
            claude_5h_pct, claude_5h_reset, claude_weekly_pct, claude_weekly_reset,
            today_tokens, today_input_tokens, today_output_tokens, today_requests,
            tokens_delta, requests_delta, gemini_5h_delta, gemini_weekly_delta,
            interval_seconds, active_instances_count, lifetime_tokens, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            iso_now,
            epoch_now,
            email,
            plan,
            model,
            project_id,
            quotas["gemini_5h_pct"],
            quotas["gemini_5h_reset"],
            quotas["gemini_weekly_pct"],
            quotas["gemini_weekly_reset"],
            quotas["claude_5h_pct"],
            quotas["claude_5h_reset"],
            quotas["claude_weekly_pct"],
            quotas["claude_weekly_reset"],
            today_tokens,
            today_input,
            today_output,
            today_requests,
            tokens_delta,
            requests_delta,
            gemini_5h_delta,
            gemini_weekly_delta,
            interval_seconds,
            active_instances_count,
            lifetime_tokens,
            json.dumps(record),
        ),
    )
    snapshot_id = cur.lastrowid

    for inst in instances:
        cur.execute(
            """
            INSERT INTO running_instances (
                snapshot_id, pid, elapsed_seconds, cwd, conversation_id, session_name, args
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                inst.get("pid", 0),
                inst.get("elapsedSeconds", 0),
                inst.get("cwd", ""),
                inst.get("conversationId") or "",
                inst.get("sessionName") or "",
                inst.get("args") or "",
            ),
        )

    con.commit()
    con.close()

    # Append to JSON Lines
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Write latest JSON cache
    with open(LATEST_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # Regenerate static HTML dashboard
    generate_html_dashboard(record)

    return record


def get_recent_history(limit: int = 96) -> List[Dict[str, Any]]:
    # 96 snapshots * 15 minutes = 24 hours
    if not DB_FILE.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT timestamp, epoch, gemini_5h_pct, gemini_weekly_pct, today_tokens,
                   today_input_tokens, today_output_tokens, tokens_delta, requests_delta,
                   gemini_5h_delta, active_instances_count, interval_seconds, gemini_5h_reset
            FROM snapshots
            ORDER BY epoch DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        con.close()
        rows.reverse()
        return [
            {
                "timestamp": r[0],
                "epoch": r[1],
                "gemini_5h_pct": r[2],
                "gemini_weekly_pct": r[3],
                "today_tokens": r[4],
                "today_input_tokens": r[5],
                "today_output_tokens": r[6],
                "tokens_delta": r[7],
                "requests_delta": r[8],
                "gemini_5h_delta": r[9],
                "active_instances_count": r[10],
                "interval_seconds": r[11],
                "gemini_5h_reset": r[12],
            }
            for r in rows
        ]
    except Exception:
        return []


def format_number(val: Optional[int | float]) -> str:
    if val is None:
        return "—"
    num = float(val)
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if abs(num) >= 1_000:
        return f"{num / 1_000:.1f}K"
    return f"{int(num):,}"


def format_duration_seconds(seconds: int) -> str:
    if seconds <= 0:
        return "now"
    days, sec = divmod(seconds, 86400)
    hours, sec = divmod(sec, 3600)
    minutes = sec // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and len(parts) < 2:
        parts.append(f"{minutes}m")
    return " ".join(parts) or "<1m"


def format_reset_time(iso_ts: Optional[str]) -> str:
    if not iso_ts:
        return "Unknown"
    try:
        dt = datetime.datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        diff = int((dt - now).total_seconds())
        local_str = dt.astimezone().strftime("%d %b %H:%M")
        if diff > 0:
            return f"{local_str} (in {format_duration_seconds(diff)})"
        return f"{local_str} (refreshing)"
    except Exception:
        return str(iso_ts)


def get_db_metrics() -> Dict[str, Any]:
    if not DB_FILE.exists():
        return {"size_kb": 0, "snapshots_count": 0, "instances_count": 0, "journal_mode": "WAL"}
    try:
        con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
        cur = con.cursor()
        sc = cur.execute("SELECT count(*) FROM snapshots").fetchone()[0]
        ic = cur.execute("SELECT count(*) FROM running_instances").fetchone()[0]
        jm = cur.execute("PRAGMA journal_mode").fetchone()[0]
        con.close()
        return {
            "size_kb": round(DB_FILE.stat().st_size / 1024, 1),
            "snapshots_count": sc,
            "instances_count": ic,
            "journal_mode": str(jm).upper(),
        }
    except Exception:
        return {"size_kb": 0, "snapshots_count": 0, "instances_count": 0, "journal_mode": "WAL"}


def get_hourly_summary(limit: int = 6) -> List[Dict[str, Any]]:
    if not DB_FILE.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT hour_window, snapshot_count, total_tokens_burned, total_requests,
                   min_5h_quota_pct, max_5h_quota_pct, avg_5h_quota_pct, avg_active_instances
            FROM v_hourly_burn
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        con.close()
        return [
            {
                "hour": r[0],
                "snapshots": r[1],
                "tokens": r[2] or 0,
                "requests": r[3] or 0,
                "min_5h": r[4] or 0.0,
                "max_5h": r[5] or 0.0,
                "avg_5h": r[6] or 0.0,
                "avg_instances": r[7] or 0.0,
            }
            for r in rows
        ]
    except Exception:
        return []


def generate_html_dashboard(latest_record: Optional[Dict[str, Any]] = None) -> None:
    if not latest_record:
        if LATEST_JSON_FILE.exists():
            with open(LATEST_JSON_FILE, "r", encoding="utf-8") as f:
                latest_record = json.load(f)
        else:
            return

    history = get_recent_history(96)
    db_metrics = get_db_metrics()
    hourly_summary = get_hourly_summary(6)
    full_payload = {
        "latest": latest_record,
        "history": history,
        "db_metrics": db_metrics,
        "hourly_summary": hourly_summary,
        "generated_at": datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    json_embedded = json.dumps(full_payload).replace("</script>", "<\\/script>")

    account = latest_record.get("account", {})
    email = html.escape(str(account.get("email", "unknown")))
    plan = html.escape(str(account.get("plan", "Antigravity")))
    model = html.escape(str(account.get("model", "Gemini 3.8 Flash (High)")))

    quotas = latest_record.get("quotas", {})
    gemini_5h_pct = quotas.get("gemini_5h_pct", 100.0)
    gemini_weekly_pct = quotas.get("gemini_weekly_pct", 100.0)
    claude_5h_pct = quotas.get("claude_5h_pct", 100.0)
    claude_weekly_pct = quotas.get("claude_weekly_pct", 100.0)

    gemini_5h_reset = format_reset_time(quotas.get("gemini_5h_reset"))
    gemini_weekly_reset = format_reset_time(quotas.get("gemini_weekly_reset"))

    today = latest_record.get("today_usage", {})
    today_total = today.get("total_tokens", 0)
    today_input = today.get("input_tokens", 0)
    today_output = today.get("output_tokens", 0)
    today_requests = today.get("requests", 0)

    deltas = latest_record.get("deltas", {})
    tokens_delta = deltas.get("tokens_delta", 0)
    requests_delta = deltas.get("requests_delta", 0)
    gemini_5h_delta = deltas.get("gemini_5h_delta", 0.0)

    instances = latest_record.get("instances", [])
    top_convs = latest_record.get("top_conversations", [])

    # Calculate 1-hour burn rate from history if available
    recent_1h_tokens = 0
    if len(history) >= 2:
        cutoff = int(time.time()) - 3600
        recs_in_1h = [h for h in history if h.get("epoch", 0) >= cutoff]
        recent_1h_tokens = sum(h.get("tokens_delta", 0) for h in recs_in_1h)

    # HTML template with pure CSS and zero external dependencies
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Antigravity Quota & Token Burn Monitor</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap');

    :root {{
      --bg-main: #090d16;
      --bg-card: #121824;
      --bg-card-hover: #172030;
      --border: #202b3f;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --accent-blue: #38bdf8;
      --accent-green: #22c55e;
      --accent-yellow: #eab308;
      --accent-red: #ef4444;
      --accent-purple: #a855f7;
      --font-mono: 'JetBrains Mono', 'JetBrainsMono Nerd Font', 'JetBrainsMono NF', monospace;
    }}
    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: var(--font-mono) !important;
    }}
    body {{
      background: var(--bg-main);
      color: var(--text-main);
      font-family: var(--font-mono) !important;
      line-height: 1.5;
      padding: 24px;
      -webkit-font-smoothing: antialiased;
    }}
    .container {{
      max-width: 1300px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
      flex-wrap: wrap;
      gap: 12px;
    }}
    .header-title {{
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .header-title h1 {{
      font-size: 20px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
      letter-spacing: -0.5px;
    }}
    .header-title .tag {{
      font-size: 11px;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent-blue);
      padding: 2px 8px;
      border-radius: 9999px;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }}
    .header-meta {{
      font-size: 12.5px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .header-meta strong {{
      color: var(--text-main);
    }}
    .refresh-badge {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 14px;
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 12px;
      white-space: nowrap;
    }}
    .live-dot {{
      display: inline-block;
      width: 8px;
      height: 8px;
      background: var(--accent-green);
      border-radius: 50%;
      margin-right: 6px;
      box-shadow: 0 0 8px var(--accent-green);
    }}

    /* Top Grid Metrics (Single Line) */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .metric-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      transition: border-color 0.15s ease;
      min-width: 0;
    }}
    .metric-card:hover {{
      border-color: #334155;
    }}

    /* Velocity Stats Grid */
    .velocity-stats-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .vstat-box {{
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 8px;
      padding: 12px 14px;
      min-width: 0;
    }}
    .vstat-label {{
      font-size: 11px;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.5px;
      margin-bottom: 5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .vstat-value {{
      font-size: 19px;
      font-weight: 800;
      color: var(--text-main);
      margin-bottom: 4px;
      display: flex;
      align-items: baseline;
      gap: 6px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .vstat-sub {{
      font-size: 11px;
      color: var(--text-muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .metric-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .metric-label {{
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .metric-badge {{
      font-size: 11px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 6px;
      font-family: var(--font-mono);
    }}
    .badge-green {{ background: rgba(34, 197, 94, 0.15); color: var(--accent-green); border: 1px solid rgba(34, 197, 94, 0.3); }}
    .badge-yellow {{ background: rgba(234, 179, 8, 0.15); color: var(--accent-yellow); border: 1px solid rgba(234, 179, 8, 0.3); }}
    .badge-red {{ background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.3); }}
    .badge-blue {{ background: rgba(56, 189, 248, 0.15); color: var(--accent-blue); border: 1px solid rgba(56, 189, 248, 0.3); }}

    .metric-value {{
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -1px;
      margin-bottom: 8px;
      font-family: var(--font-mono);
      display: flex;
      align-items: baseline;
      gap: 6px;
    }}
    .metric-value .unit {{
      font-size: 14px;
      font-weight: 500;
      color: var(--text-muted);
    }}
    .progress-bar-bg {{
      height: 8px;
      background: #1e293b;
      border-radius: 4px;
      overflow: hidden;
      margin: 10px 0;
    }}
    .progress-bar-fill {{
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s ease;
    }}
    .metric-footer {{
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      margin-top: 6px;
    }}
    .metric-footer strong {{
      color: var(--text-main);
    }}

    /* Charts Section */
    .chart-section {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
      margin-bottom: 24px;
    }}
    .section-title {{
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .chart-container {{
      width: 100%;
      height: 220px;
      position: relative;
    }}
    svg.timeline-svg {{
      width: 100%;
      height: 100%;
      overflow: visible;
    }}

    /* Tables */
    .table-container {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
      margin-bottom: 24px;
      overflow-x: auto;
      min-width: 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      color: var(--text-muted);
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      white-space: nowrap;
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-family: var(--font-mono);
      font-size: 12.5px;
      white-space: nowrap;
    }}
    tr:hover td {{
      background: var(--bg-card-hover);
    }}
    .td-title {{
      font-family: var(--font-mono) !important;
      font-weight: 600;
      color: var(--text-main);
    }}
    .status-pill {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
    }}

    /* Split layout */
    .split-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .split-grid > * {{
      min-width: 0;
    }}
    @media (max-width: 900px) {{
      .split-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="header-title">
        <h1>Antigravity Quota Tracker <span class="tag">AI Ultra (20x)</span></h1>
        <div class="header-meta">
          <span>Account: <strong>{email}</strong></span>
          <span>•</span>
          <span>Plan: <strong>{plan}</strong></span>
          <span>•</span>
          <span>Model: <strong>{model}</strong></span>
        </div>
      </div>
      <div class="refresh-badge">
        <span class="live-dot"></span>
        <strong>15m Cron Active</strong>
        <span style="color: var(--border);">|</span>
        <span style="color: var(--text-muted);">Updated: <span id="last-update">{full_payload['generated_at']}</span></span>
      </div>
    </header>

    <!-- KPI Metric Cards (Single Line) -->
    <div class="metrics-grid">
      <!-- 5-Hour Quota -->
      <div class="metric-card">
        <div class="metric-header">
          <span class="metric-label">Gemini 5-Hour Limit</span>
          <span class="metric-badge {'badge-green' if gemini_5h_pct >= 50 else ('badge-yellow' if gemini_5h_pct >= 20 else 'badge-red')}">
            {gemini_5h_pct:.1f}% Left
          </span>
        </div>
        <div class="metric-value">
          {gemini_5h_pct:.1f}<span class="unit">%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: {min(100.0, max(0.0, gemini_5h_pct))}%; background: {'var(--accent-green)' if gemini_5h_pct >= 50 else ('var(--accent-yellow)' if gemini_5h_pct >= 20 else 'var(--accent-red)')};"></div>
        </div>
        <div class="metric-footer">
          <span>Resets: <strong>{gemini_5h_reset}</strong></span>
          <span>15m Change: <strong>{'+' if gemini_5h_delta > 0 else ''}{gemini_5h_delta:.1f}%</strong></span>
        </div>
      </div>

      <!-- Weekly Quota -->
      <div class="metric-card">
        <div class="metric-header">
          <span class="metric-label">Gemini 7-Day Limit</span>
          <span class="metric-badge badge-green">
            {gemini_weekly_pct:.1f}% Left
          </span>
        </div>
        <div class="metric-value">
          {gemini_weekly_pct:.1f}<span class="unit">%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: {min(100.0, max(0.0, gemini_weekly_pct))}%; background: var(--accent-green);"></div>
        </div>
        <div class="metric-footer">
          <span>Resets: <strong>{gemini_weekly_reset}</strong></span>
          <span>Weekly Cap Tier: <strong>Ultra (20x)</strong></span>
        </div>
      </div>

      <!-- Token Consumption Today -->
      <div class="metric-card">
        <div class="metric-header">
          <span class="metric-label">Tokens Consumed Today</span>
          <span class="metric-badge badge-blue">{today_requests:,} Req</span>
        </div>
        <div class="metric-value">
          {format_number(today_total)}
        </div>
        <div class="metric-footer">
          <span>In: <strong>{format_number(today_input)}</strong> | Out: <strong>{format_number(today_output)}</strong></span>
          <span>Last 15m: <strong>+{format_number(tokens_delta)}</strong></span>
        </div>
        <div class="metric-footer" style="margin-top: 4px;">
          <span>1h Consumption: <strong>~{format_number(recent_1h_tokens or tokens_delta * 4)}/hr</strong></span>
          <span>Lifetime: <strong>{format_number(latest_record.get('lifetime_tokens'))}</strong></span>
        </div>
      </div>

      <!-- Active Running Instances -->
      <div class="metric-card">
        <div class="metric-header">
          <span class="metric-label">Active CLI Instances</span>
          <span class="metric-badge badge-yellow">{len(instances)} Processes</span>
        </div>
        <div class="metric-value">
          {len(instances)}<span class="unit">instances</span>
        </div>
        <div class="metric-footer">
          <span>Claude/GPT 5h: <strong>{claude_5h_pct:.0f}% left</strong></span>
          <span>Claude/GPT 7d: <strong>{claude_weekly_pct:.0f}% left</strong></span>
        </div>
        <div class="metric-footer" style="margin-top: 4px;">
          <span>Primary Workspace: <strong>~/astralane-quant</strong></span>
        </div>
      </div>
    </div>

    <!-- Burn Down Velocity & 5-Hour Average Forecast Section -->
    <div class="table-container" style="border: 1px solid rgba(56, 189, 248, 0.35); background: linear-gradient(180deg, rgba(18, 24, 36, 0.95) 0%, rgba(10, 14, 23, 0.95) 100%); margin-bottom: 24px;">
      <div class="section-title">
        <span style="display: flex; align-items: center; gap: 8px;">
          <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" style="color: var(--accent-blue);"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path></svg>
          Burn Down Velocity & 5-Hour Limit Forecast
        </span>
        <span id="forecast-pill" class="status-pill badge-green">CALCULATING...</span>
      </div>

      <!-- Quick Velocity & Token Metrics Bar -->
      <div class="velocity-stats-grid">
        <div class="vstat-box">
          <div class="vstat-label">Total Tokens Used Today</div>
          <div class="vstat-value">{format_number(today_total)}</div>
          <div class="vstat-sub">Prompt: <strong>{format_number(today_input)}</strong> ({today_input / max(1, today_total) * 100:.1f}%) | Gen: <strong>{format_number(today_output)}</strong> ({today_output / max(1, today_total) * 100:.1f}%)</div>
        </div>

        <div class="vstat-box">
          <div class="vstat-label">Available Tokens in 5h Window</div>
          <div class="vstat-value" style="color: var(--accent-blue);">~{format_number(gemini_5h_pct * 850000)} <span style="font-size: 13px; font-weight: normal; color: var(--text-muted);">({gemini_5h_pct:.1f}% left)</span></div>
          <div class="vstat-sub">Est. 5h Window Capacity: <strong>~85M</strong> tokens (Google AI Ultra)</div>
        </div>

        <div class="vstat-box">
          <div class="vstat-label">Current Burn Velocity</div>
          <div class="vstat-value" style="color: var(--accent-purple);">+{format_number(tokens_delta)} <span style="font-size: 13px; font-weight: normal; color: var(--text-muted);">/ 15m</span></div>
          <div class="vstat-sub">Velocity: <strong>~{format_number(recent_1h_tokens or tokens_delta * 4)}/hr</strong> &nbsp;|&nbsp; <strong>~{format_number((tokens_delta or 0) / 15)}/min</strong></div>
        </div>

        <div class="vstat-box">
          <div class="vstat-label">5-Hour Quota Runway / Forecast</div>
          <div class="vstat-value" id="runway-val" style="color: var(--accent-green);">UNDERSHOOT (SAFE)</div>
          <div class="vstat-sub" id="runway-sub">Resets: <strong>{gemini_5h_reset}</strong></div>
        </div>
      </div>

      <!-- Detailed Burn Down Table -->
      <div style="overflow-x: auto; margin-top: 10px;">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Total Used</th>
              <th>Input (Prompt)</th>
              <th>Output (Gen)</th>
              <th>15m Burn</th>
              <th>Burn Velocity</th>
              <th>5h Quota</th>
              <th>5h Delta</th>
              <th>5h Forecast</th>
              <th>Instances</th>
            </tr>
          </thead>
          <tbody id="burndown-rows">
            <!-- Rendered dynamically by script from embedded data -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Timeline Charts Section -->
    <div class="chart-section">
      <div class="section-title">
        <span>Quota & Token Burn Timeline (Past 24 Hours)</span>
        <span style="font-size: 12px; color: var(--text-muted); font-weight: normal;">Recorded in 15-minute intervals</span>
      </div>
      <div class="chart-container" id="timeline-chart">
        <svg id="chart-svg" class="timeline-svg" viewBox="0 0 1000 200" preserveAspectRatio="none">
          <!-- Rendered dynamically by script -->
        </svg>
      </div>
      <div style="display: flex; gap: 20px; font-size: 12px; color: var(--text-muted); margin-top: 12px; justify-content: flex-end;">
        <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 12px; height: 3px; background: #38bdf8; border-radius: 2px;"></span> Gemini 5-Hour Quota (%)</span>
        <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 12px; height: 3px; background: #22c55e; border-radius: 2px;"></span> Gemini 7-Day Quota (%)</span>
        <span style="display: flex; align-items: center; gap: 6px;"><span style="width: 10px; height: 10px; background: rgba(168, 85, 247, 0.4); border: 1px solid #a855f7; border-radius: 2px;"></span> Tokens Burned in 15m (Delta)</span>
      </div>
    </div>

    <!-- Split View: Active Processes & Top Token Sessions -->
    <div class="split-grid">
      <!-- Active Processes Table -->
      <div class="table-container">
        <div class="section-title">
          <span>Currently Running agy CLI Processes ({len(instances)})</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>PID</th>
              <th>Directory</th>
              <th>Session / Title</th>
              <th>Uptime</th>
            </tr>
          </thead>
          <tbody>
            {"".join(f'''
            <tr>
              <td>{inst.get("pid")}</td>
              <td>{html.escape(str(inst.get("cwd", "")).replace("/home/mizzlr", "~"))}</td>
              <td class="td-title">{html.escape(inst.get("sessionName") or "Default")}</td>
              <td>{format_duration_seconds(inst.get("elapsedSeconds", 0))}</td>
            </tr>
            ''' for inst in instances) if instances else '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No active CLI processes found</td></tr>'}
          </tbody>
        </table>
      </div>

      <!-- Top Active Sessions from DB -->
      <div class="table-container">
        <div class="section-title">
          <span>Recent High-Step Conversations</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Conversation</th>
              <th>Total Steps</th>
              <th>Status</th>
              <th>Last Active</th>
            </tr>
          </thead>
          <tbody>
            {"".join(f'''
            <tr>
              <td class="td-title">{html.escape(c.get("title", "Untitled"))}</td>
              <td style="color: var(--accent-blue); font-weight: 700;">{c.get("step_count", 0):,}</td>
              <td><span class="status-pill {'badge-green' if c.get('is_active') else 'badge-blue'}">{'ACTIVE' if c.get('is_active') else 'IDLE'}</span></td>
              <td style="color: var(--text-muted); font-size: 11px;">{c.get("last_modified", "")[:19].replace("T", " ")}</td>
            </tr>
            ''' for c in top_convs[:8]) if top_convs else '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No conversation records</td></tr>'}
          </tbody>
        </table>
    </div>

    <!-- Hourly Rollup Analytics (Full Width SQLite View) -->
    <div class="table-container">
      <div class="section-title">
        <span style="display: flex; align-items: center; gap: 8px;">
          <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" style="color: var(--accent-purple);"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          Hourly Rollup Analytics (SQLite View)
        </span>
        <span class="status-pill badge-purple">Aggregated Hourly</span>
      </div>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Hour Window</th>
              <th>Tokens Burned</th>
              <th>Requests</th>
              <th>Avg 5h Quota</th>
              <th>Avg Instances</th>
            </tr>
          </thead>
          <tbody>
            {"".join(f'''
            <tr>
              <td><strong>{h.get("hour")}</strong></td>
              <td style="color: var(--accent-purple); font-weight: 700;">+{format_number(h.get("tokens", 0))}</td>
              <td>{format_number(h.get("requests", 0))}</td>
              <td><strong>{h.get("avg_5h", 0.0):.1f}%</strong></td>
              <td><span class="status-pill badge-yellow">{h.get("avg_instances", 0.0):.1f}</span></td>
            </tr>
            ''' for h in hourly_summary) if hourly_summary else '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Collecting hourly data...</td></tr>'}
          </tbody>
        </table>
      </div>
    </div>

    <!-- SQLite Database Architecture Panel -->
    <div class="table-container" style="padding: 18px 22px;">
      <div class="section-title" style="margin-bottom: 12px;">
        <span style="display: flex; align-items: center; gap: 8px;">
          <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" style="color: var(--accent-green);"><path d="M4 7v10c0 2 1.5 3 3.5 3h9c2 0 3.5-1 3.5-3V7c0-2-1.5-3-3.5-3h-9C5.5 4 4 5 4 7z"></path><path d="M4 12c0 2 1.5 3 3.5 3h9c2 0 3.5-1 3.5-3"></path></svg>
          SQLite Storage & View Architecture
        </span>
        <span class="status-pill badge-green">WAL Active</span>
      </div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px;">
        <div class="vstat-box">
          <div class="vstat-label">Database Path</div>
          <div style="font-size: 13px; font-weight: 600; color: var(--text-main); font-family: var(--font-mono); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{html.escape(str(DB_FILE))}">{html.escape(str(DB_FILE).replace("/home/mizzlr", "~"))}</div>
          <div class="vstat-sub">{db_metrics["size_kb"]} KB &nbsp;|&nbsp; {db_metrics["journal_mode"]} Mode</div>
        </div>
        <div class="vstat-box">
          <div class="vstat-label">Persisted Records</div>
          <div class="vstat-value" style="color: var(--accent-purple);">{format_number(db_metrics["snapshots_count"])} <span style="font-size: 13px; font-weight: normal; color: var(--text-muted);">snapshots</span></div>
          <div class="vstat-sub">Logged states: <strong>{format_number(db_metrics["instances_count"])}</strong> rows</div>
        </div>
        <div class="vstat-box">
          <div class="vstat-label">Analytical Views</div>
          <div style="font-size: 13px; font-weight: 600; color: var(--accent-blue); font-family: var(--font-mono); margin-top: 4px;">v_hourly_burn</div>
          <div class="vstat-sub">v_instance_activity</div>
        </div>
        <div class="vstat-box">
          <div class="vstat-label">CLI Query Access</div>
          <div style="font-size: 11px; color: #cbd5e1; font-family: var(--font-mono); margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title='python3 tracker.py sql "SELECT * FROM v_hourly_burn"'><code>python3 tracker.py sql "..."</code></div>
          <div class="vstat-sub">Fast atomic SQLite reads</div>
        </div>
      </div>
    </div>

  <!-- Embedded JSON Data for Instant Offline / file:// Loading -->
  <script id="tracker-data" type="application/json">
{json_embedded}
  </script>

  <script>
    (function() {{
      const rawData = document.getElementById('tracker-data').textContent;
      let data = {{}};
      try {{
        data = JSON.parse(rawData);
      }} catch (e) {{
        console.error("Failed to parse embedded tracker data", e);
        return;
      }}

      const history = data.history || [];
      const burndownRows = document.getElementById('burndown-rows');
      if (burndownRows && history.length > 0) {{
        const reversed = [...history].reverse();

        // Calculate latest forecast for the top banner
        const latest = reversed[0];
        const secLatest = latest.interval_seconds || 900;
        const latestPctDropHr = (latest.gemini_5h_delta < 0 && secLatest > 0) ? Math.abs(latest.gemini_5h_delta) * (3600 / secLatest) : 0;
        const runwayHours = latestPctDropHr > 0 ? (latest.gemini_5h_pct / latestPctDropHr) : 999;
        
        const forecastPill = document.getElementById('forecast-pill');
        const runwayVal = document.getElementById('runway-val');
        const runwaySub = document.getElementById('runway-sub');

        if (runwayVal) {{
          if (latestPctDropHr > 20) {{
            runwayVal.textContent = 'OVERSHOOT RISK';
            runwayVal.style.color = 'var(--accent-red)';
            if (forecastPill) {{
              forecastPill.className = 'status-pill badge-red';
              forecastPill.textContent = 'OVERSHOOT RISK (' + runwayHours.toFixed(1) + 'h left)';
            }}
            if (runwaySub) runwaySub.innerHTML = 'Burn <strong>' + latestPctDropHr.toFixed(1) + '%/hr</strong> &nbsp;|&nbsp; 0% in <strong>' + runwayHours.toFixed(1) + 'h</strong> (< 5h reset)';
          }} else if (latestPctDropHr > 0) {{
            runwayVal.textContent = 'UNDERSHOOT (SAFE)';
            runwayVal.style.color = 'var(--accent-green)';
            if (forecastPill) {{
              forecastPill.className = 'status-pill badge-green';
              forecastPill.textContent = 'UNDERSHOOT (SAFE ' + runwayHours.toFixed(1) + 'h)';
            }}
            if (runwaySub) runwaySub.innerHTML = 'Burn <strong>' + latestPctDropHr.toFixed(1) + '%/hr</strong> &nbsp;|&nbsp; 0% in <strong>' + runwayHours.toFixed(1) + 'h</strong> (> 5h window)';
          }} else {{
            runwayVal.textContent = 'RECOVERING / STEADY';
            runwayVal.style.color = 'var(--accent-blue)';
            if (forecastPill) {{
              forecastPill.className = 'status-pill badge-blue';
              forecastPill.textContent = 'RECOVERING / STEADY';
            }}
            if (runwaySub) runwaySub.innerHTML = 'Burn rate is <strong>0%/hr</strong> or quota replenishing';
          }}
        }}

        burndownRows.innerHTML = reversed.map(h => {{
          const timeStr = h.timestamp ? h.timestamp.substring(11, 19) : '—';
          const totT = h.today_tokens != null ? formatNumber(h.today_tokens) : '—';
          const inT = h.today_input_tokens != null ? formatNumber(h.today_input_tokens) : '—';
          const outT = h.today_output_tokens != null ? formatNumber(h.today_output_tokens) : '—';
          const inPct = (h.today_tokens && h.today_input_tokens) ? ((h.today_input_tokens / h.today_tokens) * 100).toFixed(1) + '%' : '—';
          const outPct = (h.today_tokens && h.today_output_tokens) ? ((h.today_output_tokens / h.today_tokens) * 100).toFixed(1) + '%' : '—';

          const tDelta = h.tokens_delta != null ? (h.tokens_delta > 0 ? '+' : '') + formatNumber(h.tokens_delta) : '—';
          const sec = h.interval_seconds || 900;
          const hourlyTokens = (h.tokens_delta != null && sec > 0) ? formatNumber((h.tokens_delta / sec) * 3600) + '/hr' : '—';
          const q5 = h.gemini_5h_pct != null ? h.gemini_5h_pct.toFixed(2) + '%' : '—';
          const d5 = h.gemini_5h_delta != null ? (h.gemini_5h_delta > 0 ? '+' : '') + h.gemini_5h_delta.toFixed(2) + '%' : '—';
          
          let d5Color = 'var(--text-muted)';
          if (h.gemini_5h_delta < 0) d5Color = 'var(--accent-red)';
          else if (h.gemini_5h_delta > 0) d5Color = 'var(--accent-green)';

          const pctDropHr = (h.gemini_5h_delta < 0 && sec > 0) ? Math.abs(h.gemini_5h_delta) * (3600 / sec) : 0;
          let forecastTag = '';
          if (pctDropHr > 20) {{
            const hrs = (h.gemini_5h_pct / pctDropHr).toFixed(1);
            forecastTag = `<span class="status-pill badge-red">OVERSHOOT (${{hrs}}h left)</span>`;
          }} else if (pctDropHr > 0) {{
            const hrs = (h.gemini_5h_pct / pctDropHr).toFixed(1);
            forecastTag = `<span class="status-pill badge-green">UNDERSHOOT (${{hrs}}h)</span>`;
          }} else {{
            forecastTag = `<span class="status-pill badge-blue">STEADY / RECOVERY</span>`;
          }}

          const inst = h.active_instances_count != null ? h.active_instances_count : '—';

          return `
            <tr>
              <td><strong>${{timeStr}}</strong></td>
              <td style="color: var(--text-main); font-weight: 700;" title="${{h.today_tokens ? h.today_tokens.toLocaleString('en-US') + ' tokens' : ''}}">${{totT}}</td>
              <td style="color: var(--accent-blue);" title="${{h.today_input_tokens ? h.today_input_tokens.toLocaleString('en-US') + ' tokens' : ''}}">${{inT}} <span style="font-size: 11px; color: var(--text-muted);">(${{inPct}})</span></td>
              <td style="color: var(--accent-yellow);" title="${{h.today_output_tokens ? h.today_output_tokens.toLocaleString('en-US') + ' tokens' : ''}}">${{outT}} <span style="font-size: 11px; color: var(--text-muted);">(${{outPct}})</span></td>
              <td style="color: var(--accent-purple); font-weight: 700;" title="${{h.tokens_delta ? (h.tokens_delta > 0 ? '+' : '') + h.tokens_delta.toLocaleString('en-US') + ' tokens' : ''}}">${{tDelta}}</td>
              <td>${{hourlyTokens}}</td>
              <td><strong>${{q5}}</strong></td>
              <td style="color: ${{d5Color}}; font-weight: 600;">${{d5}}</td>
              <td>${{forecastTag}}</td>
              <td><span class="status-pill badge-yellow">${{inst}}</span></td>
            </tr>
          `;
        }}).join('');
      }}

      // Render SVG timeline chart
      const svg = document.getElementById('chart-svg');
      if (svg && history.length >= 1) {{
        renderChart(svg, history);
      }}

      function formatNumber(val) {{
        if (val == null) return '—';
        const n = Math.abs(val);
        if (n >= 1e9) return (val / 1e9).toFixed(2) + 'B';
        if (n >= 1e6) return (val / 1e6).toFixed(2) + 'M';
        if (n >= 1e3) return (val / 1e3).toFixed(1) + 'K';
        return val.toLocaleString();
      }}

      function renderChart(svg, points) {{
        const width = 1000;
        const height = 200;
        const padX = 40;
        const padY = 25;
        const chartW = width - padX * 2;
        const chartH = height - padY * 2;

        let content = '';

        // Grid lines (y = 100%, 75%, 50%, 25%, 0%)
        [100, 75, 50, 25, 0].forEach(level => {{
          const y = padY + chartH * (1 - level / 100);
          content += `<line x1="${{padX}}" y1="${{y}}" x2="${{width - padX}}" y2="${{y}}" stroke="#1e293b" stroke-dasharray="4 4" stroke-width="1" />`;
          content += `<text x="${{padX - 8}}" y="${{y + 4}}" fill="#64748b" font-size="10" font-family="'JetBrains Mono', monospace" text-anchor="end">${{level}}%</text>`;
        }});

        if (points.length < 2) {{
          // Single point or empty
          const p = points[0];
          const y5 = padY + chartH * (1 - (p.gemini_5h_pct || 100) / 100);
          content += `<circle cx="${{width / 2}}" cy="${{y5}}" r="6" fill="#38bdf8" />`;
          content += `<text x="${{width / 2}}" y="${{y5 - 12}}" fill="#38bdf8" font-size="12" font-weight="bold" text-anchor="middle">${{p.gemini_5h_pct}}%</text>`;
          svg.innerHTML = content;
          return;
        }}

        // Calculate max token delta for bar heights
        const maxTokens = Math.max(...points.map(p => p.tokens_delta || 0), 100000);

        // Draw Token Delta Bars in background
        const barW = Math.max(4, Math.min(24, (chartW / points.length) * 0.5));
        points.forEach((p, idx) => {{
          const x = padX + (idx / (points.length - 1)) * chartW;
          const tokenFrac = Math.min(1, (p.tokens_delta || 0) / maxTokens);
          const barH = tokenFrac * (chartH * 0.7);
          const barY = padY + chartH - barH;
          content += `<rect x="${{x - barW / 2}}" y="${{barY}}" width="${{barW}}" height="${{barH}}" fill="rgba(168, 85, 247, 0.25)" rx="2" />`;
        }});

        // Build path for gemini_5h_pct
        const pts5h = points.map((p, idx) => {{
          const x = padX + (idx / (points.length - 1)) * chartW;
          const y = padY + chartH * (1 - (p.gemini_5h_pct != null ? p.gemini_5h_pct : 100) / 100);
          return [x, y];
        }});
        const d5h = pts5h.map((pt, i) => (i === 0 ? 'M' : 'L') + `${{pt[0]}},${{pt[1]}}`).join(' ');
        content += `<path d="${{d5h}}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linejoin="round" />`;

        // Draw dots for 5h
        pts5h.forEach((pt, i) => {{
          content += `<circle cx="${{pt[0]}}" cy="${{pt[1]}}" r="3.5" fill="#38bdf8" />`;
        }});

        // Build path for gemini_weekly_pct
        const pts7d = points.map((p, idx) => {{
          const x = padX + (idx / (points.length - 1)) * chartW;
          const y = padY + chartH * (1 - (p.gemini_weekly_pct != null ? p.gemini_weekly_pct : 100) / 100);
          return [x, y];
        }});
        const d7d = pts7d.map((pt, i) => (i === 0 ? 'M' : 'L') + `${{pt[0]}},${{pt[1]}}`).join(' ');
        content += `<path d="${{d7d}}" fill="none" stroke="#22c55e" stroke-width="1.8" stroke-dasharray="5 3" />`;

        svg.innerHTML = content;
      }}

      // Live auto-refresh when accessed via HTTP server
      if (window.location.protocol.startsWith('http')) {{
        setInterval(() => {{
          fetch('/api/latest')
            .then(res => res.json())
            .then(latest => {{
              if (latest && latest.timestamp !== (data.latest && data.latest.timestamp)) {{
                window.location.reload();
              }}
            }})
            .catch(() => {{}});
        }}, 30000);
      }}
    }})();
  </script>
</body>
</html>
"""
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)


def print_status() -> None:
    if not LATEST_JSON_FILE.exists():
        print("No recorded snapshots yet. Run: python3 tracker.py record")
        return
    with open(LATEST_JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    acc = data.get("account", {})
    quotas = data.get("quotas", {})
    usage = data.get("today_usage", {})
    deltas = data.get("deltas", {})
    instances = data.get("instances", [])

    print("=========================================================")
    print(" ANTIGRAVITY QUOTA & TOKEN BURN MONITOR")
    print("=========================================================")
    print(f" Account:       {acc.get('email')} [{acc.get('plan')}]")
    print(f" Active Model:  {acc.get('model')}")
    print(f" Timestamp:     {data.get('timestamp')}")
    print("---------------------------------------------------------")
    print(f" Gemini 5-Hour Limit:   {quotas.get('gemini_5h_pct', 0.0):>5.1f}% left   (Resets: {format_reset_time(quotas.get('gemini_5h_reset'))})")
    print(f" Gemini 7-Day Limit:    {quotas.get('gemini_weekly_pct', 0.0):>5.1f}% left   (Resets: {format_reset_time(quotas.get('gemini_weekly_reset'))})")
    print(f" Claude / GPT 5-Hour:   {quotas.get('claude_5h_pct', 100.0):>5.1f}% left")
    print("---------------------------------------------------------")
    print(f" Tokens Today:          {format_number(usage.get('total_tokens', 0)):>8}  ({usage.get('requests', 0):,} requests)")
    print(f" 15m Token Burn:       +{format_number(deltas.get('tokens_delta', 0))}  (5h Delta: {deltas.get('gemini_5h_delta', 0.0):+.2f}%)")
    print(f" Active CLI Processes:  {len(instances)}")
    print("=========================================================")


def run_sql_query(query: str) -> None:
    if not DB_FILE.exists():
        print(f"Database {DB_FILE} does not exist yet. Run: python3 tracker.py record")
        return
    con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
    cur = con.cursor()
    try:
        cur.execute(query)
        rows = cur.fetchall()
        headers = [desc[0] for desc in cur.description] if cur.description else []
        if not headers:
            print("Query executed with no result set.")
            return

        col_widths = [len(h) for h in headers]
        str_rows = []
        for r in rows:
            sr = [str(val) if val is not None else "" for val in r]
            str_rows.append(sr)
            for i, val in enumerate(sr):
                col_widths[i] = max(col_widths[i], len(val))

        header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "  ".join("-" * col_widths[i] for i in range(len(headers)))
        print(header_line)
        print(sep_line)
        for sr in str_rows:
            print("  ".join(sr[i].ljust(col_widths[i]) for i in range(len(sr))))
        print(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''})")
    except Exception as e:
        print(f"SQL Error: {e}", file=sys.stderr)
    finally:
        con.close()


def print_db_info() -> None:
    if not DB_FILE.exists():
        print(f"Database {DB_FILE} does not exist yet. Run: python3 tracker.py record")
        return
    con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
    cur = con.cursor()
    db_size = DB_FILE.stat().st_size
    journal_mode = con.execute("PRAGMA journal_mode;").fetchone()[0]

    tables = cur.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()

    print("=========================================================")
    print(" SQLITE DATABASE METRICS")
    print("=========================================================")
    print(f" File:          {DB_FILE}")
    print(f" Size:          {db_size / 1024:.1f} KB")
    print(f" Journal Mode:  {journal_mode.upper()}")
    print("---------------------------------------------------------")
    print(f" {'TYPE':<8} {'NAME':<24} {'ROW COUNT':>12}")
    print(f" {'-'*8} {'-'*24} {'-'*12}")
    for name, obj_type in tables:
        try:
            count = cur.execute(f"SELECT count(*) FROM \"{name}\"").fetchone()[0]
            print(f" {obj_type.upper():<8} {name:<24} {count:>12,}")
        except Exception:
            print(f" {obj_type.upper():<8} {name:<24} {'—':>12}")
    print("=========================================================")
    con.close()


def export_csv(table_name: str, out_path: Optional[str] = None) -> None:
    import csv
    if not DB_FILE.exists():
        print(f"Database {DB_FILE} does not exist yet.")
        return
    target = Path(out_path) if out_path else DATA_DIR / f"{table_name}.csv"
    con = sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)
    cur = con.cursor()
    try:
        cur.execute(f"SELECT * FROM \"{table_name}\"")
        rows = cur.fetchall()
        headers = [desc[0] for desc in cur.description]
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"Exported {len(rows)} rows from '{table_name}' to {target}")
    except Exception as e:
        print(f"Export Error: {e}", file=sys.stderr)
    finally:
        con.close()


class TrackerHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(TRACKER_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/latest":
            if LATEST_JSON_FILE.exists():
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with open(LATEST_JSON_FILE, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "No latest data")
            return
        if self.path == "/api/history":
            history = get_recent_history(96)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(history).encode("utf-8"))
            return
        super().do_GET()


def find_available_port(start_port: int, max_tries: int = 50) -> int:
    import socket
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("", port))
                return port
            except OSError:
                continue
    return start_port


def serve_dashboard(port: int = 8989) -> None:
    socketserver.TCPServer.allow_reuse_address = True
    actual_port = find_available_port(port)
    if actual_port != port:
        print(f"Notice: Requested port {port} is occupied. Using available port {actual_port}.")
    with socketserver.TCPServer(("", actual_port), TrackerHTTPHandler) as httpd:
        print(f"Serving AG Tracker dashboard at http://localhost:{actual_port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Antigravity Quota & Token Burn Tracker")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    subparsers.add_parser("record", help="Fetch live snapshot, compute deltas, write to JSONL and SQLite, update HTML")
    subparsers.add_parser("status", help="Print recent status summary in terminal")
    subparsers.add_parser("db-info", help="Print SQLite database tables, views, and row counts")
    sql_parser = subparsers.add_parser("sql", help="Run arbitrary SQL query against the SQLite database")
    sql_parser.add_argument("query", type=str, help="SQL query to execute")

    csv_parser = subparsers.add_parser("export-csv", help="Export a database table or view to CSV")
    csv_parser.add_argument("--table", type=str, default="snapshots", help="Table or view name (default: snapshots)")
    csv_parser.add_argument("--output", type=str, default=None, help="Output CSV path")

    subparsers.add_parser("generate", help="Regenerate index.html from stored data")
    serve_parser = subparsers.add_parser("serve", help="Run a local HTTP server to view the dashboard")
    serve_parser.add_argument("--port", type=int, default=8989, help="Port to serve on (default: 8989)")

    args = parser.parse_args()
    cmd = args.subcommand or "status"

    if cmd == "record":
        rec = record_snapshot()
        print(f"Recorded snapshot at {rec['timestamp']}: 5h quota {rec['quotas']['gemini_5h_pct']}% | Today: {format_number(rec['today_usage']['total_tokens'])} tokens")
    elif cmd == "status":
        print_status()
    elif cmd == "db-info":
        print_db_info()
    elif cmd == "sql":
        run_sql_query(args.query)
    elif cmd == "export-csv":
        export_csv(args.table, args.output)
    elif cmd == "generate":
        generate_html_dashboard()
        print(f"Generated dashboard at {HTML_FILE}")
    elif cmd == "serve":
        serve_dashboard(args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

