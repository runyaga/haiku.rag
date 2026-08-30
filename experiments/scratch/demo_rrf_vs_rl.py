"""RRF and RL, each with per-source parametric filters, across two databases."""
import asyncio, sys
sys.path.insert(0, "/Users/runyaga/dev/haiku-rag-fusion/experiments")
from collections import Counter
from fusion import federated_search
from haiku.rag.client import HaikuRAG

SRC = ["cv22b", "ac130j"]
Q = "engine fire emergency procedure"

# A DIFFERENT parametric filter per database -- impossible through the library API.
# Genuinely selective, and DIFFERENT per database:
#   cv22b  -> 18 of 354 (HOSTAC shipboard pubs), plus a DATE RANGE
#   ac130j -> 42 of 180 (checklists)
PER_SOURCE = {
    "cv22b":  "uri LIKE '%hostac%' AND created_at >= '2020-01-01'",
    "ac130j": "uri LIKE '%checklist%'",
}

def show(title, hits):
    alloc = Counter(h.source for h in hits)
    print(f"\n{title}\n  slots={dict(alloc)}")
    print(f"  {'#':<2} {'source':<8} {'db-rank':>7} {'raw':>10} {'fused':>9}  text")
    for i, h in enumerate(hits):
        print(f"  {i:<2} {h.source:<8} {h.rank:>7} {h.raw:>10.4f} {h.score:>9.4f}  {h.content[:40].strip()!r}")

async def main():
    async with HaikuRAG(read_only=True) as rag:
        for mode in ("rrf", "rl"):
            hits = await federated_search(rag, Q, sources=SRC, mode=mode, limit=6)
            show(f"=== {mode.upper()}  |  no filter  |  2 databases", hits)
        print("\n" + "-" * 78)
        for mode in ("rrf", "rl"):
            hits = await federated_search(rag, Q, sources=SRC, filters=PER_SOURCE, mode=mode, limit=6)
            show(f"=== {mode.upper()}  |  PER-SOURCE parametric filters  |  2 databases", hits)

asyncio.run(main())
