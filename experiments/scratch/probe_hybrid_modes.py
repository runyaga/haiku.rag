"""RRF vs LinearCombination hybrid fusion, with and without a parametric filter."""
import asyncio, lancedb
from lancedb.rerankers import RRFReranker, LinearCombinationReranker
from haiku.rag.client import HaikuRAG

DB = "/Users/runyaga/dev/haiku-rag-fusion/experiments/data/cv22b.lancedb"
Q = "engine fire emergency procedure"

async def main():
    async with HaikuRAG(read_only=True) as rag:
        c = (await rag.clients_for(["cv22b"]))[0]
        vec = await c.embedder.embed_query(Q)

    db = await lancedb.connect_async(DB)
    meta = await db.open_table("document_meta")
    chunks = await db.open_table("chunks")

    rows = await (meta.query().select(["id"])
                  .where("uri LIKE '%hostac%'").to_pandas())
    ids = ", ".join(f"'{d}'" for d in rows["id"])
    filt = f"document_id IN ({ids})"
    print(f"filter matches {len(rows)} of 354 docs\n")

    for mode, rr in (("RRF", RRFReranker()),
                     ("LinearCombination", LinearCombinationReranker(weight=0.7))):
        for label, where in (("no filter", None), ("+ parametric filter", filt)):
            q = (chunks.query().nearest_to(vec).column("vector")
                 .nearest_to_text(Q, columns="content_fts").rerank(rr))
            if where:
                q = q.where(where)
            try:
                res = await q.limit(4).to_list()
                sc = [round(r.get("_relevance_score", float("nan")), 4) for r in res]
                print(f"  {mode:<18} {label:<22} {len(res)} rows  scores={sc}")
            except Exception as e:
                print(f"  {mode:<18} {label:<22} {type(e).__name__}: {str(e)[:70]}")
    print()

asyncio.run(main())
