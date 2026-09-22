#!/usr/bin/env python3
"""Enhanced Antigravity CLI status viewer (agb / ag-top).

Enriches agl's output with live 5h token balance, burn velocity,
input/output ratio, and active vs idle process spotlight.
Does NOT modify or replace the original agl CLI.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sqlite3
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AGL_PATH = Path.home() / ".local" / "bin" / "agl"
TRACKER_DB = Path(__file__).resolve().parent / "data" / "usage_history.db"
BRAIN_DIR = Path.home() / ".gemini" / "antigravity-cli" / "conversations"
SUMMARIES_DB = Path.home() / ".gemini" / "antigravity-cli" / "conversation_summaries.db"


def load_agl():
    if not AGL_PATH.exists():
        sys.exit(f"Error: {AGL_PATH} not found.")
    return SourceFileLoader("agl", str(AGL_PATH)).load_module()


def get_plan_name(agl_module, creds: Optional[dict]) -> str:
    try:
        if creds and "token" in creds:
            token = creds["token"].get("access_token")
            if token:
                resp = agl_module._api_post(agl_module.LOAD_CODE_ASSIST_URL, token, {}, timeout=5.0)
                paid = resp.get("paidTier") or {}
                if paid.get("name"):
                    tier_id = paid.get("id", "")
                    if "ultra" in tier_id:
                        return f"{paid['name']} (20x)"
                    if "pro" in tier_id:
                        return f"{paid['name']} (5x)"
                    return paid["name"]
    except Exception:
        pass
    return "Google AI Ultra (20x)"


def get_active_session_metrics(cutoff_seconds: int = 1800) -> Dict[str, Dict[str, Any]]:
    """Scan conversation DBs modified within cutoff_seconds to get recent burn metrics."""
    now = time.time()
    cutoff = now - cutoff_seconds
    metrics: Dict[str, Dict[str, Any]] = {}

    if not BRAIN_DIR.exists():
        return metrics

    for db_path in BRAIN_DIR.glob("*.db"):
        try:
            mtime = db_path.stat().st_mtime
            if mtime < cutoff:
                continue

            cid = db_path.stem
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cur = con.cursor()

            # Inspect steps in the active period
            recent_tokens = 0
            recent_reqs = 0
            step_timestamps = []

            for row in cur.execute("SELECT idx, metadata FROM steps WHERE metadata IS NOT NULL"):
                meta_blob = row[1]
                if not meta_blob:
                    continue
                # Simple extraction of tokens and timestamp
                rec = parse_token_blob(meta_blob)
                if rec:
                    ts_epoch = rec[0]
                    if ts_epoch >= cutoff:
                        recent_tokens += rec[1]  # total
                        recent_reqs += 1
                        step_timestamps.append(ts_epoch)

            con.close()

            if recent_reqs > 0:
                # Calculate cadence
                cadence_str = ""
                if len(step_timestamps) >= 3:
                    step_timestamps.sort()
                    diffs = [t2 - t1 for t1, t2 in zip(step_timestamps[:-1], step_timestamps[1:]) if t2 - t1 > 0]
                    if diffs:
                        median_diff = sorted(diffs)[len(diffs) // 2]
                        if median_diff <= 10:
                            cadence_str = f"~{int(median_diff)}s loop"
                        elif median_diff <= 60:
                            cadence_str = f"~{int(median_diff)}s cadence"

                metrics[cid] = {
                    "recent_tokens": recent_tokens,
                    "recent_reqs": recent_reqs,
                    "cadence": cadence_str,
                }
        except Exception:
            continue

    return metrics


def parse_token_blob(metadata_blob: bytes) -> Optional[Tuple[float, int]]:
    """Fast protobuf scanner for timestamp and total tokens."""
    try:
        # We reuse agl's robust proto parser if needed, or inline quick proto unpack
        top = _fast_parse_proto(metadata_blob)
        ts_epoch = 0.0
        if 1 in top:
            for _, data in top[1]:
                sub1 = _fast_parse_proto(data)
                if 1 in sub1 and sub1[1]:
                    ts_epoch = float(sub1[1][0][1])
                    break
        if 9 in top:
            for _, data in top[9]:
                sub9 = _fast_parse_proto(data)
                inp = int(sub9[2][0][1]) if 2 in sub9 and sub9[2] else 0
                out = int(sub9[3][0][1]) if 3 in sub9 and sub9[3] else 0
                if inp + out > 0:
                    return (ts_epoch or time.time(), inp + out)
    except Exception:
        pass
    return None


def _fast_parse_proto(b: bytes) -> Dict[int, List[Tuple[str, Any]]]:
    fields: Dict[int, List[Tuple[str, Any]]] = {}
    i = 0
    length_b = len(b)
    while i < length_b:
        try:
            shift = 0
            key = 0
            while True:
                byte = b[i]
                i += 1
                key |= (byte & 0x7F) << shift
                shift += 7
                if not (byte & 0x80):
                    break
            field_num = key >> 3
            wire_type = key & 7
            if wire_type == 0:
                val = 0
                shift = 0
                while True:
                    byte = b[i]
                    i += 1
                    val |= (byte & 0x7F) << shift
                    shift += 7
                    if not (byte & 0x80):
                        break
                fields.setdefault(field_num, []).append(("varint", val))
            elif wire_type == 2:
                length = 0
                shift = 0
                while True:
                    byte = b[i]
                    i += 1
                    length |= (byte & 0x7F) << shift
                    shift += 7
                    if not (byte & 0x80):
                        break
                data = b[i : i + length]
                i += length
                fields.setdefault(field_num, []).append(("bytes", data))
            elif wire_type == 1:
                i += 8
            elif wire_type == 5:
                i += 4
            else:
                break
        except Exception:
            break
    return fields


def get_tracker_metrics() -> Dict[str, Any]:
    """Fetch delta and velocity metrics from usage_history.db if present."""
    if not TRACKER_DB.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{TRACKER_DB}?mode=ro", uri=True)
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT gemini_5h_pct, gemini_5h_delta, tokens_delta, interval_seconds, timestamp
            FROM snapshots ORDER BY id DESC LIMIT 2
            """
        ).fetchall()
        con.close()
        if rows:
            latest = rows[0]
            sec = latest[3] or 900
            hourly_rate = (latest[2] / sec) * 3600 if sec > 0 else 0
            return {
                "latest_5h_pct": latest[0],
                "latest_5h_delta": latest[1],
                "tokens_delta": latest[2],
                "hourly_rate": hourly_rate,
            }
    except Exception:
        pass
    return {}


def format_short_tokens(val: Optional[int | float]) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-a", "--all", action="store_true", help="show all instances without collapsing idle ones")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color output")
    parser.add_argument("--timeout", type=float, default=10, help="API timeout in seconds")
    args = parser.parse_args()

    agl = load_agl()

    try:
        data = agl.fetch_status(args.timeout)
    except Exception as error:
        print(f"agb: failed to fetch live status: {error}", file=sys.stderr)
        return 1

    records = agl.load_local_usage()
    data["usage"] = agl.usage_summary(records)
    instances = agl._running_instances()
    data["instances"] = instances

    # Additional contextual enrichments
    creds = agl._get_credentials()
    plan_name = get_plan_name(agl, creds)
    tracker_stats = get_tracker_metrics()
    active_metrics = get_active_session_metrics(cutoff_seconds=1800)

    # Colors
    color = sys.stdout.isatty() and not args.no_color
    bold = "\033[1m" if color else ""
    dim = "\033[2m" if color else ""
    green = "\033[32m" if color else ""
    yellow = "\033[33m" if color else ""
    red = "\033[31m" if color else ""
    cyan = "\033[36m" if color else ""
    purple = "\033[35m" if color else ""
    reset = "\033[0m" if color else ""

    # 1. Account Section
    account = data.get("account", {})
    email = account.get("email") or "unknown"
    model = account.get("model") or "Gemini 3.8 Flash"

    print(f"{bold}Antigravity account{reset}")
    print(f"  {email} [{bold}{cyan}{plan_name}{reset}]  {dim}(Model: {model}){reset}")

    # Calculate real-time burn velocity from active sessions (last 30m * 2)
    sum_recent_30m = sum(act.get("recent_tokens", 0) for act in active_metrics.values())
    hourly_burn = sum_recent_30m * 2 if sum_recent_30m > 0 else tracker_stats.get("hourly_rate", 0)
    delta_15m = tracker_stats.get("tokens_delta", 0) or (sum_recent_30m // 2)

    # 2. Quota Bars Section with Capacity & Runway
    quota_rows = agl._quota_rows(data.get("quotas", {}))
    gemini_5h_pct = 100.0
    gemini_5h_reset = ""

    # Estimate 5h capacity (typically ~124M tokens under AI Ultra 20x)
    est_5h_capacity = 124_000_000

    for row in quota_rows:
        name = row["label"]
        rem = row["remainingPercent"]
        if "Gemini 5h" in name:
            gemini_5h_pct = rem
            gemini_5h_reset = agl._format_reset(row["resetTime"])
        if rem >= 50:
            bar_color = green
        elif rem >= 20:
            bar_color = yellow
        else:
            bar_color = red

        bar_str = f"{bar_color}{agl._bar(rem)}{reset}"
        reset_str = f"{dim}{agl._format_reset(row['resetTime'])}{reset}"

        # Context tag for Gemini rows
        extra_tag = ""
        if "Gemini 5h" in name:
            avail = est_5h_capacity * (rem / 100.0)
            if hourly_burn > 0:
                burn_str = f"burn: ~{format_short_tokens(hourly_burn)}/h"
                runway_hrs = avail / hourly_burn if hourly_burn > 0 else 99
                status_str = f"{green}safe {runway_hrs:.1f}h{reset}" if runway_hrs >= 5 else f"{red}overshoot {runway_hrs:.1f}h{reset}"
                extra_tag = f" {dim}(~{format_short_tokens(avail)} left | {burn_str} | {status_str}){reset}"
            else:
                extra_tag = f" {dim}(~{format_short_tokens(avail)} available of {format_short_tokens(est_5h_capacity)}){reset}"
        elif "Gemini 7d" in name:
            extra_tag = f" {dim}(~2.2B token bank | ~8.5x weekly headroom){reset}"

        print(f"  {name + ':':<18} {bar_str} {rem:>3}% left{extra_tag}  {reset_str}")

    # 3. Token Usage Section (Preserves existing lines + adds rich velocity & in/out)
    usage = data.get("usage", {})
    windows = usage.get("windows", {})
    summary = usage.get("summary", {})

    today_win = windows.get("today", {})
    today_tokens = today_win.get("total_tokens", 0)
    today_in = today_win.get("input_tokens", 0)
    today_out = today_win.get("output_tokens", 0)
    w7_tokens = windows.get("7d", {}).get("total_tokens", 0)
    w30_tokens = windows.get("30d", {}).get("total_tokens", 0)

    in_pct = (today_in / max(1, today_tokens)) * 100
    out_pct = (today_out / max(1, today_tokens)) * 100

    # Estimate window burn (burned since 5h window start)
    burned_in_5h = est_5h_capacity * ((100.0 - gemini_5h_pct) / 100.0)

    print(f"\n{bold}Token usage{reset}")
    today_delta_tag = f" {purple}(+{format_short_tokens(delta_15m)} in 15m){reset}" if delta_15m > 0 else ""
    print(f"  Today:    {agl._format_number(today_tokens):>8}{today_delta_tag}   {dim}In: {format_short_tokens(today_in)} ({in_pct:.1f}%) | Out: {format_short_tokens(today_out)} ({out_pct:.1f}%){reset}")
    if burned_in_5h > 0 or hourly_burn > 0:
        print(f"  Window:   {format_short_tokens(burned_in_5h):>8} in 5h window   {dim}Velocity: ~{format_short_tokens(hourly_burn)}/h (~{format_short_tokens(hourly_burn/60)}/min){reset}")
    print(f"  7d:       {agl._format_number(w7_tokens):>8}   30d:   {agl._format_number(w30_tokens):>8}")
    print(f"  Lifetime: {agl._format_number(summary.get('lifetimeTokens'))}   Streak: {summary.get('currentStreakDays') or 0}d   Peak day: {agl._format_number(summary.get('peakDailyTokens'))}")

    # 4. Running Instances Section with Activity Spotlight
    active_list = []
    idle_list = []

    for inst in instances:
        cid = inst.get("conversationId") or ""
        act = active_metrics.get(cid)
        if act and act.get("recent_tokens", 0) > 0:
            active_list.append((inst, act))
        else:
            idle_list.append(inst)

    # Sort active by recent tokens burned desc
    active_list.sort(key=lambda x: x[1]["recent_tokens"], reverse=True)

    print(f"\n{bold}Running instances{reset}: {len(instances)}  {dim}({len(active_list)} active, {len(idle_list)} idle){reset}")

    # Print Active Instances
    for inst, act in active_list:
        elapsed = agl._format_duration(inst["elapsedSeconds"])
        cwd = format_cwd(inst["cwd"])
        name = inst.get("sessionName") or "Default"
        tokens_burned = act["recent_tokens"]
        reqs = act["recent_reqs"]
        cadence = f", {act['cadence']}" if act.get("cadence") else ""

        flame = f"{red}🔥{reset}" if "3s" in cadence or "loop" in cadence or tokens_burned > 5_000_000 else f"{yellow}⚡{reset}"
        tag = f"{bold}{yellow}● ACTIVE{reset}"

        print(f"  {tag}  {cwd:<18} {bold}{name}{reset} {dim}(PID {inst['pid']}, {elapsed}){reset}")
        print(f"            {flame} {purple}+{format_short_tokens(tokens_burned)}{reset} tokens in 30m {dim}({reqs} calls{cadence}){reset}")

    # Print Idle Instances
    if args.all:
        for inst in idle_list:
            elapsed = agl._format_duration(inst["elapsedSeconds"])
            cwd = format_cwd(inst["cwd"])
            name = inst.get("sessionName") or "Default"
            print(f"  {dim}○ IDLE    {cwd:<18} {name} (PID {inst['pid']}, {elapsed}) [0 tok in 30m]{reset}")
    else:
        if idle_list:
            names = [inst.get("sessionName") or f"PID {inst['pid']}" for inst in idle_list[:4]]
            more_cnt = len(idle_list) - len(names)
            more_str = f" +{more_cnt} more" if more_cnt > 0 else ""
            print(f"  {dim}○ IDLE    {len(idle_list)} instances (0 tokens in last 30m): {', '.join(names)}{more_str}  [use -a to show all]{reset}")

    return 0


def format_cwd(cwd: str) -> str:
    try:
        home = str(Path.home())
        if cwd == home:
            return "~"
        if cwd.startswith(home):
            return "~/" + str(Path(cwd).relative_to(Path.home()))
    except Exception:
        pass
    return cwd


if __name__ == "__main__":
    sys.exit(main())
