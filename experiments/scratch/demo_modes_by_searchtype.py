"""RL depends on raw scores being real. Hybrid scores are already RRF constants."""
import asyncio, sys
sys.path.insert(0, "/Users/runyaga/dev/haiku-rag-fusion/experiments")
from collections import Counter
from fusion import federated_search
from haiku.rag.client import HaikuRAG

SRC = ["cv22b", "ac130j"]
Q = "engine fire emergency procedure"
F = {"cv22b": "uri LIKE '%hostac%'", "ac130j": "uri LIKE '%checklist%'"}

async def main():
    async with HaikuRAG(read_only=True) as rag:
        for st in ("hybrid", "vector", "fts"):
            print(f"\n### search_type={st}")
            for mode in ("rrf", "rl"):
                h = await federated_search(rag, Q, sources=SRC, filters=F,
                                           mode=mode, search_type=st, limit=6)
                raws = [f"{x.raw:.3f}" for x in h]
                spread = max(x.raw for x in h) - min(x.raw for x in h)
                print(f"  {mode.upper():<4} slots={dict(Counter(x.source for x in h))}  "
                      f"raw-spread={spread:.4f}  raws={raws}")

asyncio.run(main())
