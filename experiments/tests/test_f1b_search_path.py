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

from typing import ClassVar

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
    async def test_search_never_returns_an_excluded_document(self, client, docs, cell):
        """Subset, across all three search types at once.

        A SUBSET assertion is satisfied by an empty result -- measured, 11 of 47
        fts cells returned nothing and passed vacuously before this was fixed.
        Requiring non-empty per search type would be wrong: a date-equality
        filter combined with a fixed query legitimately matches nothing under
        BM25. So the requirement is that the cell produces results in AT LEAST
        ONE search type, which fails loudly if the filter breaks entirely.
        """
        predicate, oracle = build(cell, docs)
        allowed = c.expected(docs, oracle)
        assert allowed, f"{cell.id}: empty oracle"
        assert len(allowed) < len(docs), f"{cell.id}: oracle matches everything"

        per_type: dict[str, set[str]] = {}
        for search_type in ("fts", "vector", "hybrid"):
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
                per_type[search_type] = returned
                leaked = returned - allowed
                if span is not None:
                    span.set_attribute("returned", len(returned))
                    span.set_attribute("leaked", len(leaked))
                    span.set_attribute("empty", not returned)

            assert not leaked, (
                f"{cell.id}/{search_type}: search returned {len(leaked)} "
                "document(s) the filter excludes -- the chunk-level IN(...) "
                f"translation disagrees with the predicate. e.g. "
                f"{sorted(leaked)[:2]}"
            )

        assert any(per_type.values()), (
            f"{cell.id}: every search type returned NOTHING. The subset "
            "assertions above are all vacuously true, so this cell proved "
            "nothing about the filter."
        )


class TestEmptyResultsDoNotGrowSilently:
    """A tripwire on the vacuous cells.

    MEASURED at 45b3ef66: fts 11 of 47, vector 0, hybrid 0. Those 11 are
    legitimate -- the predicate and the fixed query simply do not intersect. But
    if that count RISES, filters are silently matching less than they should and
    every subset assertion in this file gets weaker without failing.
    """

    EXPECTED_EMPTY: ClassVar[dict[str, int]] = {"fts": 11, "vector": 0, "hybrid": 0}

    @pytest.mark.parametrize("search_type", ["fts", "vector", "hybrid"])
    async def test_the_number_of_empty_cells_is_unchanged(
        self, client, docs, search_type
    ):
        empty = []
        for cell in CELLS:
            predicate, _ = build(cell, docs)
            results = await client.search(
                QUERY, limit=25, search_type=search_type, filter=predicate
            )
            if not results:
                empty.append(cell.id)
        assert len(empty) == self.EXPECTED_EMPTY[search_type], (
            f"{search_type}: {len(empty)} cells return nothing, expected "
            f"{self.EXPECTED_EMPTY[search_type]}. Cells: {sorted(empty)}"
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
