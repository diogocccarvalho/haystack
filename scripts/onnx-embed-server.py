#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/embeddings server for ONNX embedding models.

Serves any ONNX encoder (BERT-style: token input, CLS-pooled, L2-normalized)
via onnxruntime + tokenizers. Purpose: run multilingual embedders (e.g.
granite-embedding-97m-multilingual-r2 int8) that have no llama.cpp/Ollama
package, so notes-router can A/B them with zero code changes (config only).

Usage:
  python3 onnx-embed-server.py --model model_int8.onnx --tokenizer tokenizer.json --port 8086
"""
import argparse

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--port", type=int, default=8086)
ap.add_argument("--max-tokens", type=int, default=512)
args = ap.parse_args()

tok = Tokenizer.from_file(args.tokenizer)
tok.enable_truncation(max_length=args.max_tokens)
# max_tokens must be a power of two for flash-attn-style kernels; round up
if args.max_tokens > 512:
    tok.enable_truncation(max_length=512)
sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

app = FastAPI()


class EmbReq(BaseModel):
    input: object  # str or list[str]
    model: str = ""


def embed(texts):
    enc = [tok.encode(t) for t in texts]
    maxlen = max(len(e.ids) for e in enc)
    ids = np.zeros((len(enc), maxlen), dtype=np.int64)
    mask = np.zeros((len(enc), maxlen), dtype=np.int64)
    for i, e in enumerate(enc):
        n = len(e.ids)
        ids[i, :n] = e.ids
        mask[i, :n] = e.attention_mask
    feed = {"input_ids": ids, "attention_mask": mask}
    missing = {i.name for i in sess.get_inputs()} - set(feed)
    if "token_type_ids" in missing:
        feed["token_type_ids"] = np.zeros_like(ids)
    out = sess.run(None, feed)[0]  # (batch, seq, hidden) last hidden state
    # CLS pooling + L2 normalize (granite CLS-pooling convention)
    cls = out[:, 0, :]
    cls = cls / (np.linalg.norm(cls, axis=1, keepdims=True) + 1e-12)
    return cls


@app.post("/v1/embeddings")
def embeddings(req: EmbReq):
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    vecs = embed(texts)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v.tolist()}
            for i, v in enumerate(vecs)
        ],
        "model": req.model or "onnx",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
