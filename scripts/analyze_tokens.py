#!/usr/bin/env python3
"""Deep token-level analysis of why distro queries fail."""
import re
from pathlib import Path

STOP = set("the a an of about my i me that this thing stuff should where do is are for on in to with and or".split())

def get_tokens(text):
    out = []
    for w in re.findall(r"\w{4,}", text.lower()):
        if w not in STOP:
            out.append(w)
    return out

# Query 1
q1 = "dude whats the fastest setup for this laptop again"
q1_all = set(re.findall(r"\w{4,}", q1.lower()))
q1_active = q1_all - STOP
print(f"Q1 ALL tokens (incl stop/filler): {q1_all}")
print(f"Q1 ACTIVE tokens (post-STOP filter): {q1_active}")

# Read both targets
tgt1_path = Path.home() / "Documents/hermes-notes/work/research/distro-alt-NEXT-STEPS.md"
tgt2_path = Path.home() / "Documents/hermes-notes/work/research/distro-alt-v2-cli-desktop.md"

queries_and_targets = [
    ("Query 1", q1, tgt1_path),
    ("Query 2", "that distro comparison thing, the v2 deep dive", tgt2_path),
]

synonym_map = {
    'fastest': ['fast', 'speedy', 'quickest', 'rapid', 'speed', 'quick'],
    'setup': ['configuration', 'config', 'tune', 'tuning', 'arrangement', 'structure'],
    'laptop': ['notebook', 'portable', 'device'], 
    'comparison': ['compare', 'versus', 'contrast', 'versus'],
    'deep': ['thorough', 'extensive', 'comprehensive'],
}

for query_name, query_text, target_path in queries_and_targets:
    print(f"\n{'='*60}")
    print(f"{query_name}: '{query_text}'")
    print(f"Target: {target_path.name}")
    print(f"{'='*60}")
    
    raw_target = target_path.read_text().lower()
    
    # Extract title from frontmatter
    title_match = re.search(r'^title:\s*"?([^"\n]+)', raw_target[:100], re.M)
    title = title_match.group(1).strip('"').lower() if title_match else ""
    
    # Extract body (after title block)
    body_parts = re.split(r'^## ', raw_target[50:], flags=re.M)
    body = "\n".join(body_parts)
    
    # Get all tokens from different parts
    title_toks = set(re.findall(r"\w{4,}", title))
    body_toks = set(re.findall(r"\w{4,}", body))
    all_tgt_toks = title_toks | body_toks
    
    query_tokens = set(re.findall(r"\w{4,}", query_text.lower())) - STOP
    print(f"Title: {title}")
    print(f"Query tokens (active, post-STOP): {sorted(query_tokens)}")
    
    # Show which query words match WHERE
    print("\nToken overlap analysis:")
    for tw in sorted(query_tokens):
        in_title = tw in title_toks
        in_body = tw in body_toks
        in_all = tw in all_tgt_toks
        print(f"  '{tw}': title={in_title}, body={in_body}, any_chunk={in_all}")
    
    # Filename tokens
    doc_rel = str(target_path.relative_to(Path.home() / "Documents/hermes-notes")).lower()
    fname_toks = set(re.findall(r"\w+", doc_rel))
    print(f"\nFilename tokens: {fname_toks}")
    for tw in sorted(query_tokens):
        in_filename = tw in fname_toks
        print(f"  '{tw}' in filename? {in_filename}")
    
    # Vocabulary mismatch check
    print("\nVocabulary MISMATCH ANALYSIS:")
    for tw in sorted(query_tokens):
        found_directly = tw in all_tgt_toks
        synonym_hit = False
        for orig, synonyms in synonym_map.items():
            if tw == orig or tw in synonyms:
                synonym_hit = True
                if found_directly:
                    print(f"  DIRECT+SYNONYM MATCH: '{tw}' found directly in content (matches '{orig}')")
                else:
                    print(f"  SYNONYM NEEDED: '{tw}' not in content; would need synonym expansion for '{orig}'")
                break
        if not synonym_hit and not found_directly:
            print(f"  MISSING TOKEN: '{tw}' has ZERO evidence in target content!")
        elif not synonym_hit and found_directly:
            print(f"  SELF-SUFFICIENT: '{tw}' matches directly")
