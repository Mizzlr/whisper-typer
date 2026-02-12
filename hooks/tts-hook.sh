#!/bin/bash
# tts-hook.sh — Dynamic TTS for Claude Code events
# Triggered by: SessionStart, Stop, PermissionRequest, Notification, UserPromptSubmit
# Reads event JSON from stdin, sends to code_speaker HTTP API

TTS_API="http://localhost:8767"

# Quick connectivity check (non-blocking)
curl -s --connect-timeout 0.5 "$TTS_API/status" >/dev/null 2>&1 || exit 0

INPUT=$(cat)

# Debug: log ALL events
echo "$(date '+%Y-%m-%d %H:%M:%S') RAW_EVENT: $INPUT" >> /tmp/tts-hook-debug.log

# Parse event type
EVENT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('hook_event_name', ''))
except:
    pass
" 2>/dev/null)

echo "$(date '+%Y-%m-%d %H:%M:%S') PARSED_EVENT: '$EVENT'" >> /tmp/tts-hook-debug.log

case "$EVENT" in
  SessionStart)
    # Only announce on fresh startup, not on resume (prevents double-speak)
    SOURCE=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('source',''))" 2>/dev/null)
    if [ "$SOURCE" = "resume" ]; then
      exit 0
    fi

    # Check if code_speaker API is running and announce readiness
    STATUS=$(curl -s --connect-timeout 1 "$TTS_API/status" 2>/dev/null)
    if [ -n "$STATUS" ]; then
      MODEL_LOADED=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('model_loaded',False))" 2>/dev/null)
      if [ "$MODEL_LOADED" = "True" ] || [ "$MODEL_LOADED" = "true" ]; then
        curl -s -X POST "$TTS_API/speak" \
          -H "Content-Type: application/json" \
          -d '{"text":"Claude Code is ready.","summarize":false,"event_type":"session_start","start_reminder":false}' \
          >/dev/null 2>&1 &
      fi
    fi
    ;;

  Stop)
    # Pipe Python JSON directly to curl — never store in bash variable
    # This avoids shell expansion mangling of Unicode/backticks/special chars
    echo "$INPUT" | python3 -c "
import sys, json

event = json.load(sys.stdin)
transcript_path = event.get('transcript_path', '')

if not transcript_path:
    sys.exit(0)

try:
    with open(transcript_path) as f:
        lines = f.readlines()
except FileNotFoundError:
    sys.exit(0)

for line in reversed(lines):
    try:
        entry = json.loads(line)
        if entry.get('type') == 'assistant':
            content = entry.get('message', {}).get('content', [])
            for block in content:
                if block.get('type') == 'text':
                    text = block['text'][:2000]
                    if text.strip():
                        payload = json.dumps({
                            'text': text,
                            'summarize': True,
                            'event_type': 'stop',
                            'start_reminder': True,
                        })
                        # Write JSON to temp file (avoids any shell issues)
                        import tempfile, subprocess, os
                        fd, path = tempfile.mkstemp(suffix='.json')
                        os.write(fd, payload.encode('utf-8'))
                        os.close(fd)
                        subprocess.Popen([
                            'curl', '-s', '-X', 'POST',
                            '${TTS_API}/speak',
                            '-H', 'Content-Type: application/json',
                            '--data-binary', '@' + path,
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        # Clean up after brief delay (let curl read the file)
                        import threading
                        threading.Timer(5.0, os.unlink, args=[path]).start()
                        sys.exit(0)
    except (json.JSONDecodeError, KeyError):
        continue
" 2>/tmp/tts-hook-stop-err.log &
    ;;

  PermissionRequest)
    TOOL=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_name', 'unknown tool'))
except:
    print('unknown tool')
" 2>/dev/null)

    curl -s -X POST "$TTS_API/speak" \
      -H "Content-Type: application/json" \
      -d "$(python3 -c "import json; print(json.dumps({'text': 'Claude needs permission to use $TOOL.', 'summarize': False, 'event_type': 'permission', 'start_reminder': True}))")" \
      >/dev/null 2>&1 &
    ;;

  Notification)
    NTYPE=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('notification_type', ''))
except:
    pass
" 2>/dev/null)

    case "$NTYPE" in
      idle_prompt)
        curl -s -X POST "$TTS_API/speak" \
          -H "Content-Type: application/json" \
          -d '{"text":"Claude is waiting for your input.","summarize":false,"event_type":"notification","start_reminder":true}' \
          >/dev/null 2>&1 &
        ;;
      permission_prompt)
        curl -s -X POST "$TTS_API/speak" \
          -H "Content-Type: application/json" \
          -d '{"text":"Permission needed.","summarize":false,"event_type":"permission","start_reminder":true}' \
          >/dev/null 2>&1 &
        ;;
    esac
    ;;

  UserPromptSubmit)
    # User responded — cancel all reminders and current speech
    curl -s -X POST "$TTS_API/cancel-reminder" >/dev/null 2>&1 &
    ;;
esac

exit 0
