"""Token-cost dimension for haystack: retrieval vs naive full-file read.

HONEST METRIC (2026-11-16, replaces the '43x' claim): tokens-per-answer =
injected route+paths (113 tok measured) + the notes the agent READS until it
hits the winning one (E[reads] from the qa pairs). The winning note is still
read — "not in context" is not "not read". The old ratio (injection vs the
largest sampled winning file) compared different quantities and overstated
the saving; honest numbers live in scripts/sweep outputs and docs/RESEARCH.md.

For each sample query: run /query, measure the tokens the retrieval path
would inject (k snippets), and the tokens of the naive alternative (reading
the winning file whole). Emits a markdown table + CSV: per-k compression
ratio, p50/p99. The naive path is measured on the actual winning file, not
simulated — but "the agent would have read the whole file" is an assumption
labelled as such. Endpoint configurable; defaults to 127.0.0.1:8083/query.

Usage: python3 scripts/token_metrics.py [--k 3 5 10] [--n 20] [--out FILE]
"""
import argparse
import csv
import json
import statistics as st
import urllib.request
from pathlib import Path

VAULT = Path.home() / "Documents" / "hermes-notes"
CHARS_PER_TOKEN = 4.0

SAMPLE_QUERIES = [
    "street fighter design notes",
    "haystack storage dreams",
    "work schedule",
    "guitar practice",
    "diamondhead next steps",
    "distro speed research",
    "self crystallizing architecture",
    "resume draft",
    "morning brief",
    "vpn setup",
    "jira test deploy",
    "carregamentos massivos",
    "guitar theory",
    "codebase inspection",
    "eai soap service testing",
    "refine notes",
    "hyprland config",
    "llama local models",
    "openrouter prices",
    "weekly review planning",
]


def query(q, k, url):
    req = urllib.request.Request(
        url, data=json.dumps({"q": q, "k": k}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        rows = json.loads(r.read())
    return rows if isinstance(rows, list) else rows.get("results", [])


def full_file_tokens(path):
    p = VAULT / path
    try:
        return int(len(p.read_text()) / CHARS_PER_TOKEN)
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--url", default="http://127.0.0.1:8083/query")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print("| k | tokens injected (mean) | p50 | p99 | full-file tokens (mean) "
          "| compression |")
    print("|---|---|---|---|---|---|")
    csv_rows = [("k", "tokens_injected_mean", "p50", "p99",
                 "full_file_tokens_mean", "compression")]
    for k in args.k:
        inj_all, full_all, ratios = [], [], []
        for q in SAMPLE_QUERIES:
            try:
                rows = query(q, k, args.url)
            except Exception:
                continue
            inj = sum(len(r.get("snippet", "")) for r in rows) / CHARS_PER_TOKEN
            full = full_file_tokens(rows[0]["path"]) if rows else None
            if full and full > 0:
                inj_all.append(inj)
                full_all.append(full)
                ratios.append(full / inj if inj > 0 else 0)
        if not inj_all:
            print(f"| {k} | endpoint down / no results | | | | |")
            continue
        srt = sorted(inj_all)
        row = (k, int(st.mean(inj_all)), int(srt[len(srt)//2]),
               int(srt[int(len(srt)*.99)]), int(st.mean(full_all)),
               f"{st.mean(ratios):.1f}x")
        print(f"| {k} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |")
        csv_rows.append(row)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerows(csv_rows)
        print(f"csv -> {args.out}")


if __name__ == "__main__":
    main()
