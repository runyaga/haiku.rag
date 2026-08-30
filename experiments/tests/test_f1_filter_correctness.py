"""Suite F1 -- the cases the generated matrix structurally cannot cover.

Most of what was here has been deleted, not lost: `test_f1_surface_matrix.py`
generates 47 cells from `surface.supported_cells()` and asserts exact set match
on each, subsuming the per-attribute and per-operation tests that used to live
in this file.

Four kinds of case survive, because the matrix cannot express them:

  * a filter matching NOTHING -- the matrix's non-empty-oracle guard rejects it
  * a filter that ERRORS -- the matrix asserts equality, never an exception
  * a DATA-DERIVED boundary -- the matrix's builders hardcode their literals, so
    a window computed from the corpus (min(published) + 30 days) is a different
    case from the fixed 2024 window
  * the OVER-MATCHING control -- a naive LIKE returning a strict superset is the
    trap's other half, and the matrix only ever asserts exact equality
"""

from datetime import timedelta

import pytest

from fusionlab import corpus as c
from fusionlab import filters as f

pytestmark = pytest.mark.integration

DB = "alpha_db"


@pytest.fixture(scope="module")
def docs() -> list[c.SynthDoc]:
    return [d for d in c.build_corpus() if d.db == DB]


@pytest.fixture
async def uris():
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(read_only=True) as rag:
        client = (await rag.clients_for([DB]))[0]

        async def query(predicate: str | None) -> set[str]:
            found = await client.list_documents(limit=10_000, filter=predicate)
            return {d.uri for d in found if d.uri}

        yield query


class TestZeroMatch:
    """The matrix requires a non-empty oracle, so it can never assert this."""

    async def test_a_filter_matching_nothing_returns_nothing_without_error(self, uris):
        assert await uris(f.meta_eq("owner", "nobody-here")) == set()

    async def test_an_impossible_range_returns_nothing(self, uris):
        from datetime import date

        assert await uris(f.uri_date_range(date(2024, 1, 1), date(2024, 1, 1))) == set()


class TestErrors:
    """The matrix asserts equality; it never asserts that something raises."""

    async def test_an_unknown_column_raises_rather_than_returning_empty(self, uris):
        with pytest.raises(Exception, match=r"[Nn]o field named|Invalid"):
            await uris("nonexistent_column = 'x'")

    async def test_substr_the_only_extraction_primitive_is_rejected(self, uris):
        """Why RANGE is unsupported for every metadata attribute."""
        with pytest.raises(Exception, match=r"not supported|Invalid"):
            await uris("substr(metadata, 1, 4) >= '2024'")


class TestDataDerivedBoundary:
    """The matrix hardcodes its date window; this one is computed from the data,
    so it lands on the corpus's actual earliest edge rather than inside it."""

    async def test_a_narrow_window_at_the_corpus_edge_matches_exactly(self, uris, docs):
        start = min(d.published for d in docs)
        end = start + timedelta(days=30)
        expected = c.expected(docs, lambda d: start <= d.published < end)
        assert expected, "the window must not be empty"
        assert len(expected) < len(docs)
        assert await uris(f.uri_date_range(start, end)) == expected


class TestOverMatching:
    """The matrix asserts exact equality, so it cannot express 'strictly more'."""

    async def test_a_naive_like_over_matches_the_anchored_regex(self, uris, docs):
        truth = c.expected(docs, lambda d: "alpha" in d.tags)
        naive = await uris("metadata LIKE '%alpha%'")
        assert naive > truth, "the collision trap did not fire; the corpus is wrong"
        assert naive - truth, "no false positives means no trap"
