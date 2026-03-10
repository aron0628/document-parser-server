#!/bin/bash
set -e

MODE="${1:-$RUN_MODE}"
PORT="${PORT:-9997}"

case "$MODE" in
    api)
        echo "Starting Document Parser API server on port $PORT..."
        exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: entrypoint.sh [api]"
        echo "Or set RUN_MODE=api environment variable"
        exit 1
        ;;
esac
