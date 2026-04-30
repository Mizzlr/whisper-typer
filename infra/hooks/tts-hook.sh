#!/bin/bash
# tts-hook.sh — Dynamic TTS for Claude Code events
# Triggered by: SessionStart, Stop, PermissionRequest, Notification, UserPromptSubmit
# Reads event JSON from stdin, sends to code_speaker HTTP API

TTS_API="http://localhost:8767"

# Quick connectivity check — max 1s total, exit cleanly if TTS API is down
curl -s --connect-timeout 0.3 --max-time 1 "$TTS_API/status" >/dev/null 2>&1 || exit 0

INPUT=$(cat)

# Debug: log ALL events
echo "$(date '+%Y-%m-%d %H:%M:%S') RAW_EVENT: $INPUT" >> /tmp/tts-hook-debug.log

# Parse event type (jq is ~50x faster than spawning python3)
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // empty' 2>/dev/null)

echo "$(date '+%Y-%m-%d %H:%M:%S') PARSED_EVENT: '$EVENT'" >> /tmp/tts-hook-debug.log

case "$EVENT" in
  SessionStart)
    # Only announce on fresh startup — skip resume and compaction restarts
    SOURCE=$(echo "$INPUT" | jq -r '.source // empty' 2>/dev/null)
    if [ "$SOURCE" = "resume" ] || [ "$SOURCE" = "compact" ]; then
      exit 0
    fi

    # Check if code_speaker API is running and announce readiness
    STATUS=$(curl -s --connect-timeout 0.3 --max-time 1 "$TTS_API/status" 2>/dev/null)
    if [ -n "$STATUS" ]; then
      MODEL_LOADED=$(echo "$STATUS" | jq -r '.model_loaded // false' 2>/dev/null)
      if [ "$MODEL_LOADED" = "true" ]; then
        curl -s --max-time 3 -X POST "$TTS_API/speak" \
          -H "Content-Type: application/json" \
          -d '{"text":"Claude Code is ready.","summarize":false,"event_type":"session_start","start_reminder":false}' \
          >/dev/null 2>&1 &
      fi
    fi
    ;;

  Stop)
    # Extract last assistant text from transcript and send to TTS
    # Python still needed here for transcript file reading, but pipes directly to curl
    echo "$INPUT" | python3 -c "
import sys, json
event = json.load(sys.stdin)
tp = event.get('transcript_path', '')
if not tp:
    sys.exit(0)
try:
    with open(tp) as f:
        lines = f.readlines()
except FileNotFoundError:
    sys.exit(0)
for line in reversed(lines):
    try:
        entry = json.loads(line)
        if entry.get('type') == 'assistant':
            for block in entry.get('message', {}).get('content', []):
                if block.get('type') == 'text' and block['text'].strip():
                    print(json.dumps({
                        'text': block['text'][:2000],
                        'summarize': True,
                        'event_type': 'stop',
                        'start_reminder': True,
                    }))
                    sys.exit(0)
    except (json.JSONDecodeError, KeyError):
        continue
" 2>/dev/null | curl -s --max-time 3 -X POST "$TTS_API/speak" \
    -H "Content-Type: application/json" \
    -d @- >/dev/null 2>&1 &
    ;;

  PermissionRequest)
    TOOL=$(echo "$INPUT" | jq -r '.tool_name // "unknown tool"' 2>/dev/null)

    curl -s --max-time 3 -X POST "$TTS_API/speak" \
      -H "Content-Type: application/json" \
      -d "$(jq -n --arg tool "$TOOL" '{text: ("Claude needs permission to use " + $tool + "."), summarize: false, event_type: "permission", start_reminder: true}')" \
      >/dev/null 2>&1 &
    ;;

  Notification)
    NTYPE=$(echo "$INPUT" | jq -r '.notification_type // empty' 2>/dev/null)

    case "$NTYPE" in
      idle_prompt)
        curl -s --max-time 3 -X POST "$TTS_API/speak" \
          -H "Content-Type: application/json" \
          -d '{"text":"Claude is waiting for your input.","summarize":false,"event_type":"notification","start_reminder":true}' \
          >/dev/null 2>&1 &
        ;;
      permission_prompt)
        curl -s --max-time 3 -X POST "$TTS_API/speak" \
          -H "Content-Type: application/json" \
          -d '{"text":"Permission needed.","summarize":false,"event_type":"permission","start_reminder":true}' \
          >/dev/null 2>&1 &
        ;;
    esac
    ;;

  UserPromptSubmit)
    # User responded — cancel all reminders and current speech
    curl -s --max-time 3 -X POST "$TTS_API/cancel-reminder" >/dev/null 2>&1 &
    ;;
esac

exit 0
