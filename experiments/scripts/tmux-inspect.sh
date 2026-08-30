#!/usr/bin/env bash
# Open the haiku.rag inspector TUI on one database, in its own tmux session.
#
#   scripts/tmux-inspect.sh alpha_db          # a synthetic corpus
#   scripts/tmux-inspect.sh cv22b             # a real airpubs corpus
#   scripts/tmux-inspect.sh /path/to/x.lancedb
#
# `inspect` takes --db PATH only -- it has no --db-name -- so a configured name
# is resolved to its path here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin"
TARGET="${1:?usage: tmux-inspect.sh <db-name|path>}"

case "$TARGET" in
  alpha_db|beta_db|gamma_db) DB="$ROOT/synth/$TARGET.lancedb"; CFG="$ROOT/synth/config-synth.yaml" ;;
  cv22b|ac130j)              DB="$ROOT/data/$TARGET.lancedb";  CFG="$ROOT/config-federated.yaml" ;;
  *)                         DB="$TARGET";                     CFG="$ROOT/config-federated.yaml" ;;
esac

[ -d "$DB" ] || { echo "no such database: $DB" >&2; exit 1; }
SESSION="hrag-$(basename "$DB" .lancedb)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "attaching to existing session $SESSION"
else
  tmux new-session -d -s "$SESSION" -x "$(tput cols)" -y "$(tput lines)" \
    -e "HAIKU_RAG_CONFIG_PATH=$CFG" -e "PYTHONPATH=$ROOT/src" \
    "$PY/haiku-rag inspect --db '$DB'"
  # A second window with a shell already pointed at the same database, for
  # running queries alongside the TUI.
  tmux new-window -t "$SESSION" -n shell \
    -e "HAIKU_RAG_CONFIG_PATH=$CFG" -e "PYTHONPATH=$ROOT/src" -c "$ROOT"
  tmux select-window -t "$SESSION:0"
fi
tmux attach -t "$SESSION"
