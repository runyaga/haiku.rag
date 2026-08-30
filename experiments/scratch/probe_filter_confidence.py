"""Filtering happens before fusion. Can either mode tell a good filtered set
from a bad one, or does it confidently rank the best of a bad set?"""
import asyncio, sys
sys.path.insert(0, "/Users/runyaga/dev/haiku-rag-fusion/experiments")
from fusion import federated_search
from haiku.rag.client import HaikuRAG

SRC = ["cv22b", "ac130j"]
CASES = [
    ("on-topic,  no filter",   "engine fire emergency procedure", None),
    ("on-topic,  filtered",    "engine fire emergency procedure",
     {"cv22b": "uri LIKE '%hostac%'", "ac130j": "uri LIKE '%checklist%'"}),
    ("OFF-TOPIC, no filter",   "chocolate cake baking recipe sugar flour", None),
    ("OFF-TOPIC, filtered",    "chocolate cake baking recipe sugar flour",
     {"cv22b": "uri LIKE '%hostac%'", "ac130j": "uri LIKE '%checklist%'"}),
]

async def main():
    async with HaikuRAG(read_only=True) as rag:
        for st in ("fts", "vector"):
            print(f"\n### search_type={st}")
            print(f"  {'case':<24} {'mode':<5} {'n':>2} {'top fused':>10} {'top raw':>9}")
            for label, q, f in CASES:
                for mode in ("rrf", "rl"):
                    h = await federated_search(rag, q, sources=SRC, filters=f,
                                               mode=mode, search_type=st, limit=6)
                    if not h:
                        print(f"  {label:<24} {mode.upper():<5} {'0':>2}  (empty)")
                        continue
                    print(f"  {label:<24} {mode.upper():<5} {len(h):>2} "
                          f"{h[0].score:>10.4f} {h[0].raw:>9.4f}")

asyncio.run(main())
