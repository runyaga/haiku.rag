"""Slot allocation + duplicate collapse across several queries, one fusion arm."""
import asyncio, sys
from haiku.rag.client import HaikuRAG

QUERIES = [
    "engine fire emergency procedure",
    "hydraulic system failure checklist",
    "minimum crew requirements for night vision goggle operations",
    "fuel dump procedure",
    "aircrew flight equipment inspection interval",
    "emergency egress from the aircraft",
]

async def main():
    arm = sys.argv[1]
    async with HaikuRAG(read_only=True) as rag:
        print(f"{'query':<52} {'slots':<22} uniq/6")
        tot = {}
        uniq_sum = 0
        for q in QUERIES:
            r = await rag.search(q, limit=6, include_images=False)
            alloc = {}
            for x in r:
                alloc[x.source] = alloc.get(x.source, 0) + 1
                tot[x.source] = tot.get(x.source, 0) + 1
            uniq = len({x.content.strip() for x in r})
            uniq_sum += uniq
            print(f"{q[:50]:<52} {str(alloc):<22} {uniq}/6")
        print(f"\n[{arm}] totals={tot}  mean distinct texts per 6 slots={uniq_sum/len(QUERIES):.2f}")

asyncio.run(main())
