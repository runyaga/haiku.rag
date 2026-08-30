#!/usr/bin/env bash
# A four-pane working session over the synthetic corpora.
#
#   window 0 "inspect"  inspector TUI on alpha_db
#   window 1 "chat"     chat TUI across all three (needs a chat model configured)
#   window 2 "shell"    python with fusionlab importable and the config set
#   window 3 "watch"    the gate, re-run on every save
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin"
CFG="$ROOT/synth/config-synth.yaml"
SESSION="hrag-lab"
ENVS=(-e "HAIKU_RAG_CONFIG_PATH=$CFG" -e "PYTHONPATH=$ROOT/src")

tmux has-session -t "$SESSION" 2>/dev/null && exec tmux attach -t "$SESSION"

tmux new-session -d -s "$SESSION" -n inspect -c "$ROOT" "${ENVS[@]}" \
  "$PY/haiku-rag inspect --db '$ROOT/synth/alpha_db.lancedb'"
tmux new-window -t "$SESSION" -n chat -c "$ROOT" "${ENVS[@]}" \
  "$PY/haiku-rag chat || (echo; echo 'chat needs a chat model in the config'; exec $SHELL)"
tmux new-window -t "$SESSION" -n shell -c "$ROOT" "${ENVS[@]}"
tmux send-keys -t "$SESSION:shell" \
  "$PY/python -q -ic 'from fusionlab import corpus as c, filters as f; from fusionlab.fusion import federated_search; docs=c.build_corpus(); print(f\"{len(docs)} docs, manifest {c.manifest_hash(docs)[:16]}…\")'" C-m

# Re-run the gate whenever a source file changes. watchfiles ships with the
# haiku.rag install, so there is nothing extra to install.
tmux new-window -t "$SESSION" -n watch -c "$ROOT" "${ENVS[@]}" \
  "$PY/watchfiles 'make check' src tests || (echo 'watchfiles unavailable; run: make check'; exec $SHELL)"

tmux select-window -t "$SESSION:inspect"
tmux attach -t "$SESSION"
