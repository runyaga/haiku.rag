"""FULL capability probe of haiku.rag's `filter`.

Every expression is tested as a TRUE/FALSE pair. An expression that returns the
same count for both is being IGNORED, not evaluated -- the dangerous case.
Known subsets (parent_uri=67, hostac=18, source_revision=287) anchor discrimination.
"""
import asyncio
from haiku.rag.client import HaikuRAG

TOTAL = 354
# (label, expr_expected_ALL_or_subset, expr_expected_NONE)
PAIRS = [
    ("LIKE",                "metadata LIKE '%\"md5\"%'",                        "metadata LIKE '%ZZQQNOPE%'"),
    ("NOT LIKE",            "metadata NOT LIKE '%ZZQQNOPE%'",                   "metadata NOT LIKE '%\"md5\"%'"),
    ("= equality",          "title IS NOT NULL OR title IS NULL",               "uri = 'nope://x'"),
    ("<> inequality",       "uri <> 'nope://x'",                                "uri <> uri"),
    ("IN list",             "1 = 1",                                            "uri IN ('a://1','b://2')"),
    ("IS NULL",             "uri IS NOT NULL",                                  "uri IS NULL"),
    ("AND",                 "metadata LIKE '%md5%' AND uri IS NOT NULL",        "metadata LIKE '%md5%' AND uri IS NULL"),
    ("OR",                  "metadata LIKE '%ZZQQ%' OR uri IS NOT NULL",        "metadata LIKE '%ZZQQ%' OR uri IS NULL"),
    ("NOT ()",              "NOT (uri IS NULL)",                                "NOT (uri IS NOT NULL)"),
    ("str >= (created_at)", "created_at >= '2000-01-01'",                       "created_at >= '2099-01-01'"),
    ("str range BETWEEN",   "created_at BETWEEN '2000-01-01' AND '2099-01-01'", "created_at BETWEEN '2098-01-01' AND '2099-01-01'"),
    ("CAST AS DATE",        "CAST(created_at AS DATE) >= DATE '2000-01-01'",    "CAST(created_at AS DATE) >= DATE '2099-01-01'"),
    ("CAST AS TIMESTAMP",   "CAST(created_at AS TIMESTAMP) >= TIMESTAMP '2000-01-01 00:00:00'", "CAST(created_at AS TIMESTAMP) >= TIMESTAMP '2099-01-01 00:00:00'"),
    ("uri lexicographic",   "uri >= 'a' AND uri < 'z'",                         "uri >= 'zzzz'"),
    ("regexp_like",         "regexp_like(metadata, '\"md5\"')",                 "regexp_like(metadata, 'ZZQQNOPE')"),
    ("regexp anchored arr", "regexp_like(metadata, '\"md5\": \"[0-9a-f]{32}\"')", "regexp_like(metadata, '\"md5\": \"[G-Z]{32}\"')"),
    ("strpos",              "strpos(metadata, '\"md5\"') > 0",                  "strpos(metadata, 'ZZQQNOPE') > 0"),
    ("length",              "length(metadata) > 10",                            "length(metadata) > 1000000"),
    ("starts_with",         "starts_with(uri, 'file://')",                      "starts_with(uri, 'zzz://')"),
    ("contains()",          "contains(metadata, '\"md5\"')",                    "contains(metadata, 'ZZQQNOPE')"),
    ("lower()",             "lower(uri) LIKE '%file%'",                         "lower(uri) LIKE '%ZZQQ%'"),
    ("substr",              "substr(metadata, 1, 1) = '{'",                     "substr(metadata, 1, 1) = 'Q'"),
    ("array_has on tags",   "1 = 1",                                            "array_has(metadata, 'x')"),
]
KNOWN = [
    ("parent_uri subset",   "metadata LIKE '%\"parent_uri\"%'", 67),
    ("hostac uri subset",   "uri LIKE '%hostac%'",              18),
    ("source_revision",     "metadata LIKE '%\"source_revision\"%'", 287),
    ("regexp == LIKE",      "regexp_like(metadata, '\"parent_uri\"')", 67),
]

async def main():
    async with HaikuRAG(read_only=True) as rag:
        c = (await rag.clients_for(["cv22b"]))[0]
        async def n(f):
            try:
                return await c.count_documents(filter=f)
            except Exception as e:
                return f"ERR:{type(e).__name__}"
        print(f"cv22b = {TOTAL} docs.  Verdict rules: T must be {TOTAL} (or subset), F must be 0.\n")
        print(f"  {'expression':<22} {'TRUE':>10} {'FALSE':>10}   verdict")
        for label, t, f in PAIRS:
            a, b = await n(t), await n(f)
            if isinstance(a, str) or isinstance(b, str):
                v = "UNSUPPORTED"
            elif a == b:
                v = "*** IGNORED (no discrimination) ***"
            elif b == 0 and a > 0:
                v = "WORKS"
            else:
                v = f"partial (T={a} F={b})"
            print(f"  {label:<22} {str(a):>10} {str(b):>10}   {v}")
        print("\n  known-subset checks (must match exactly):")
        for label, f, exp in KNOWN:
            got = await n(f)
            print(f"    {label:<22} got={got!s:<8} expected={exp:<6} {'OK' if got == exp else '*** MISMATCH ***'}")

asyncio.run(main())
