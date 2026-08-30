"""Suite F1b -- the 47 cells through SEARCH, not just list_documents.

GAP THIS CLOSES. `test_f1_surface_matrix.py` proves each predicate is exact
against `document_meta` via `list_documents`. But `search()` is a DIFFERENT
code path: it resolves the filter to document ids and rewrites it as a
chunk-level `document_id IN (...)` clause (store/repositories/chunk.py:232-247).

So a predicate could be exactly right in `list_documents` and wrong through
`search`, and nothing before this file would have caught it.

HYPOTHESIS: for every supported cell, the documents behind the chunks that
`search` returns are a SUBSET of the oracle set -- never a document the filter
excludes. Subset rather than equality because `search` also ranks and truncates
to `limit`, so it legitimately returns fewer.
"""

import pytest

from fusionlab import corpus as c
from fusionlab.observability import cell_span, configure
from fusionlab.surface import build, supported_cells

pytestmark = pytest.mark.integration

DB = "alpha_db"
CELLS = supported_cells()
QUERY = "quartz feldspar mica basalt granite schist"

configure()


@pytest.fixture(scope="module")
def docs() -> list[c.SynthDoc]:
    return [d for d in c.build_corpus() if d.db == DB]


@pytest.fixture
async def client():
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(read_only=True) as rag:
        yield (await rag.clients_for([DB]))[0]


class TestSearchHonoursEveryCell:
    @pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.id)
    @pytest.mark.parametrize("search_type", ["fts", "vector", "hybrid"])
    async def test_search_never_returns_an_excluded_document(
        self, client, docs, cell, search_type
    ):
        predicate, oracle = build(cell, docs)
        allowed = c.expected(docs, oracle)
        assert allowed, f"{cell.id}: empty oracle"
        assert len(allowed) < len(docs), f"{cell.id}: oracle matches everything"

        with cell_span(
            "f1b.search_cell",
            cell=cell.id,
            attribute=cell.attribute.name,
            operation=cell.op.value,
            storage=cell.attribute.storage.value,
            search_type=search_type,
            oracle_size=len(allowed),
        ) as span:
            results = await client.search(
                QUERY, limit=25, search_type=search_type, filter=predicate
            )
            returned = {r.document_uri for r in results if r.document_uri}
            leaked = returned - allowed
            if span is not None:
                span.set_attribute("returned", len(returned))
                span.set_attribute("leaked", len(leaked))

        assert not leaked, (
            f"{cell.id}/{search_type}: search returned {len(leaked)} document(s) "
            f"the filter excludes -- the chunk-level IN(...) translation does not "
            f"agree with the document-level predicate. e.g. {sorted(leaked)[:2]}"
        )


class TestTheTranslationIsNotVacuous:
    """A filter that excluded nothing would make every assertion above trivial."""

    async def test_a_narrow_filter_actually_reduces_what_search_returns(
        self, client, docs
    ):
        narrow = "uri LIKE '%doc-000%'"
        wide = await client.search(QUERY, limit=25, search_type="fts")
        cut = await client.search(QUERY, limit=25, search_type="fts", filter=narrow)
        assert wide, "the unfiltered arm must return something"
        assert len(cut) < len(wide), (
            "the filter did not reduce the result set, so the subset assertions "
            "above are satisfied trivially"
        )
