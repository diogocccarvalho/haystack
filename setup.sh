#!/usr/bin/env bash
# Haystack (notes-router) setup: model + config + embeddings server.
# Flags: --start  download+configure+launch the embeddings server (background)
#        --systemd  also install a user systemd unit for the server (Linux)
set -euo pipefail

MODEL_URL="https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/bge-small-en-v1.5-q8_0.gguf"
MODEL_NAME="bge-small-en-v1.5-q8_0.gguf"
MODEL_DIR="${HOME}/.local/share/haystack/models"
LLAMA_SERVE="${LLAMA_SERVER:-}"   # path to llama-server; auto-detected if empty
PORT="${PORT:-8082}"
CONFIG_DIR="${HOME}/.config/haystack"
START=0; SYSTEMD=0
for a in "$@"; do case "$a" in --start) START=1;; --systemd) SYSTEMD=1; START=1;; esac; done

mkdir -p "$MODEL_DIR" "$CONFIG_DIR" "${HOME}/.cache/haystack"

if [[ ! -f "$MODEL_DIR/$MODEL_NAME" ]]; then
  echo ">> downloading embedding model (~35 MB)"
  curl -fL --retry 3 -o "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
fi

if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
  echo ">> installing config template"
  sed -e "s|path = \"~/notes\"|path = \"${VAULT:-$HOME/notes}\"|" \
      -e "s|~/.cache/notes-router/index.sqlite|$HOME/.cache/haystack/index.sqlite|" \
    "$(dirname "$0")/config.toml" > "$CONFIG_DIR/config.toml"
  echo "   edit $CONFIG_DIR/config.toml — set vault.path and excludes"
fi

if [[ ! -f "$CONFIG_DIR/qa.json" ]]; then
  echo '{ "pairs": [] }' > "$CONFIG_DIR/qa.json"
fi

if [[ -z "$LLAMA_SERVE" ]]; then
  for c in llama-server "$HOME/src/llama.cpp/build/bin/llama-server" /usr/local/bin/llama-server; do
    if command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]]; then LLAMA_SERVE="$c"; break; fi
  done
fi

started=no
if [[ $START -eq 1 ]]; then
  if [[ -z "$LLAMA_SERVE" ]]; then
    echo "!! llama-server not found (set LLAMA_SERVER=/path/to/llama-server)." >&2
    echo "   Build: git clone https://github.com/ggml-org/llama.cpp && cmake -B build && cmake --build build -j --target llama-server" >&2
    echo "   Or point config [embeddings] url at any OpenAI-compatible endpoint (Ollama, LM Studio)." >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo ">> embeddings server already up on :$PORT"
    started=already
  else
    echo ">> starting embeddings server: $LLAMA_SERVE"
    nohup "$LLAMA_SERVE" -m "$MODEL_DIR/$MODEL_NAME" \
      --embeddings --pooling cls -c 512 -ub 512 -t 4 --port "$PORT" --host 127.0.0.1 \
      > "$HOME/.cache/haystack/embd-server.log" 2>&1 &
    for _ in $(seq 1 30); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { started=yes; break; }
      sleep 1
    done
    [[ "$started" == yes ]] || { echo "!! server did not come up; log: $HOME/.cache/haystack/embd-server.log" >&2; exit 1; }
  fi
fi

if [[ $SYSTEMD -eq 1 ]]; then
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/haystack-embd.service" <<UNIT
[Unit]
Description=Haystack embeddings server (bge-small, :$PORT)
After=network.target

[Service]
ExecStart=$LLAMA_SERVE -m $MODEL_DIR/$MODEL_NAME --embeddings --pooling cls -c 512 -ub 512 -t 4 --port $PORT --host 127.0.0.1
Restart=on-failure

[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload
  systemctl --user enable --now haystack-embd.service
  echo ">> systemd unit installed (haystack-embd.service)"
fi

cat <<EOF

Setup done.
  config:  $CONFIG_DIR/config.toml   (edit vault.path + excludes)
EOF
if [[ $started == yes ]]; then
  echo "  server:  running on :$PORT"
elif [[ $started == already ]]; then
  echo "  server:  already running on :$PORT"
else
  echo "  server:  NOT started — rerun with --start, or run:"
  echo "    $LLAMA_SERVE -m $MODEL_DIR/$MODEL_NAME --embeddings --pooling cls -c 512 -ub 512 --port $PORT --host 127.0.0.1"
fi
cat <<EOF
  index:   NR_CONFIG=$CONFIG_DIR/config.toml python3 $(dirname "$0")/notes_router.py reindex
  query:   NR_CONFIG=$CONFIG_DIR/config.toml python3 $(dirname "$0")/notes_router.py query "anything"
EOF
