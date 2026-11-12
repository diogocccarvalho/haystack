#!/usr/bin/env bash
# notes-router setup: model + config + (optional) systemd service
set -euo pipefail

MODEL_URL="https://huggingface.co/ggml-org/bge-small-en-v1.5-GGUF/resolve/main/bge-small-en-v1.5-Q8_0.gguf"
MODEL_DIR="${HOME}/.local/share/notes-router/models"
LLAMA_SERVE="${LLAMA_SERVER:-}"   # path to llama-server binary; empty = use docker later
PORT=8082
CONFIG_DIR="${HOME}/.config/notes-router"

mkdir -p "$MODEL_DIR" "$CONFIG_DIR" "${HOME}/.cache/notes-router"

if [[ ! -f "$MODEL_DIR/bge-small-en-v1.5-Q8_0.gguf" ]]; then
  echo ">> downloading embedding model (~35 MB)"
  curl -fL --retry 3 -o "$MODEL_DIR/bge-small-en-v1.5-Q8_0.gguf" "$MODEL_URL"
fi

if [[ ! -f "$CONFIG_DIR/config.toml" ]]; then
  echo ">> installing config template"
  sed "s|path = \"~/notes\"|path = \"${VAULT:-$HOME/notes}\"|" \
    "$(dirname "$0")/config.toml" > "$CONFIG_DIR/config.toml"
  echo "   edit $CONFIG_DIR/config.toml — set vault.path and excludes"
fi

if [[ ! -f "$CONFIG_DIR/qa.json" ]]; then
  echo '{ "pairs": [] }' > "$CONFIG_DIR/qa.json"
fi

cat <<EOF

Setup done. Remaining manual steps:
1. Start an embeddings server (any OpenAI-compatible endpoint works):
   llama-server -m $MODEL_DIR/bge-small-en-v1.5-Q8_0.gguf \\
     --embeddings --pooling cls -c 512 -ub 512 --port $PORT --host 127.0.0.1
   (see llama.cpp for building it; Ollama/LM Studio also work — update config [embeddings].url)
2. Edit $CONFIG_DIR/config.toml (vault path, excludes).
3. First index:  NR_CONFIG=$CONFIG_DIR/config.toml python3 $(dirname "$0")/notes_router.py reindex
4. Try:          ... query "anything"
EOF
