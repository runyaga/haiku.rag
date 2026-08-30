"""What can actually be filtered on, and what a divergent filter does."""
import asyncio, json
from collections import Counter
from haiku.rag.client import HaikuRAG

async def main():
    async with HaikuRAG(read_only=True) as rag:
        for name in ("cv22b", "ac130j"):
            c = (await rag.clients_for([name]))[0]
            docs = await c.list_documents()
            keys = Counter()
            for d in docs:
                for k in (d.metadata or {}):
                    keys[k] += 1
            print(f"{name}: {len(docs)} docs, metadata keys = {dict(keys)}")

        q = "engine fire emergency procedure"
        tests = [
            ("no filter", None),
            ("content_type (both have it)", "metadata LIKE '%\"content_type\": \"application/pdf\"%'"),
            ("parent_uri (cv22b only?)", "metadata LIKE '%\"parent_uri\"%'"),
            ("bogus_key (neither has it)", "metadata LIKE '%\"airframe\"%'"),
            ("uri path segment", "uri LIKE '%stan-eval%'"),
        ]
        print()
        for label, f in tests:
            try:
                r = await rag.search(q, limit=6, filter=f, include_images=False)
                alloc = Counter(x.source for x in r)
                print(f"  {label:<30} -> {len(r)} results  {dict(alloc)}")
            except Exception as e:
                print(f"  {label:<30} -> {type(e).__name__}: {str(e)[:60]}")

asyncio.run(main())
