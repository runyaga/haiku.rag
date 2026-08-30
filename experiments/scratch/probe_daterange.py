"""What SQL does haiku.rag's `filter` actually accept? Date-range specifically."""
import asyncio
from haiku.rag.client import HaikuRAG

TESTS = [
    ("plain LIKE (known good)",        "metadata LIKE '%\"content_type\"%'"),
    ("real column >= (created_at)",    "created_at >= '2020-01-01'"),
    ("real column range",              "created_at >= '2020-01-01' AND created_at < '2030-01-01'"),
    ("uri lexicographic range",        "uri >= 'file:///a' AND uri < 'file:///z'"),
    ("substr()",                       "substr(metadata, 1, 4) = '{\"co'"),
    ("strpos()",                       "strpos(metadata, '\"md5\"') > 0"),
    ("substr+strpos extract",          "substr(metadata, strpos(metadata, '\"content_type\": \"') + 18, 3) = 'app'"),
    ("regexp_match()",                 "regexp_match(metadata, '\"md5\": \"[0-9a-f]+\"') IS NOT NULL"),
    ("regexp_like()",                  "regexp_like(metadata, '\"md5\"')"),
    ("cast to date",                   "CAST(created_at AS DATE) >= DATE '2020-01-01'"),
    ("length()",                       "length(metadata) > 10"),
    ("numeric cast of extracted str",  "CAST(substr(metadata, strpos(metadata, '\"source_revision\": \"') + 20, 4) AS BIGINT) > 1000"),
]

async def main():
    async with HaikuRAG(read_only=True) as rag:
        c = (await rag.clients_for(["cv22b"]))[0]
        total = await c.count_documents()
        print(f"cv22b total docs = {total}\n")
        for label, f in TESTS:
            try:
                n = await c.count_documents(filter=f)
                print(f"  OK    {label:<32} -> {n:>4} docs")
            except Exception as e:
                msg = str(e).replace("\n", " ")[:78]
                print(f"  FAIL  {label:<32} -> {type(e).__name__}: {msg}")

asyncio.run(main())
