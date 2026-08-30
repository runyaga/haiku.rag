"""Show cross-database RRF slot allocation on the two airpubs corpora."""
import asyncio, os, sys
from haiku.rag.client import HaikuRAG

async def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "engine fire emergency procedure"
    async with HaikuRAG(read_only=True) as rag:
        for name in ("full set", "cv22b only", "ac130j only"):
            srcs = None if name == "full set" else [name.split()[0]]
            r = await rag.search(q, limit=6, sources=srcs, include_images=False)
            alloc = {}
            for x in r:
                alloc[x.source] = alloc.get(x.source, 0) + 1
            print(f"\n{name}: slots={alloc}")
            for i, x in enumerate(r):
                print(f"  {i}  {str(x.source):<7} {x.score:.6f}  {x.content[:52].strip()!r}")

asyncio.run(main())
