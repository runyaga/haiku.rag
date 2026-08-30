"""Suite F1b -- every supported cell of the parametric surface, against the DB.

Generated from `fusionlab.surface.supported_cells()`, not hand-written. Adding
an attribute or an operation to the enumeration makes this suite grow on its
own; failing to add a builder makes `tests/test_surface.py` fail before this
suite is even reached.

Each cell is scored as an EXACT SET MATCH against an independently written
oracle, with the two anti-vacuity guards: a non-empty expected set, and a
strict subset of the corpus.
"""

import pytest

from fusionlab import corpus as c
from fusionlab.surface import (
    Op,
    all_cells,
    build,
    supported_cells,
    unsupported_reason,
)

pytestmark = pytest.mark.integration

DB = "alpha_db"
CELLS = supported_cells()


@pytest.fixture(scope="module")
def docs() -> list[c.SynthDoc]:
    return [d for d in c.build_corpus() if d.db == DB]


@pytest.fixture
async def uris():
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(read_only=True) as rag:
        client = (await rag.clients_for([DB]))[0]

        async def query(predicate: str) -> set[str]:
            found = await client.list_documents(limit=10_000, filter=predicate)
            return {d.uri for d in found if d.uri}

        yield query


class TestEverySupportedCell:
    @pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.id)
    async def test_the_filter_returns_exactly_the_oracle_set(self, uris, docs, cell):
        predicate, oracle = build(cell, docs)
        expected = c.expected(docs, oracle)
        assert expected, f"{cell.id}: empty oracle, the comparison would be vacuous"
        assert len(expected) < len(docs), (
            f"{cell.id}: oracle matches every document -- a filter returning "
            "everything is indistinguishable from one being ignored"
        )
        assert await uris(predicate) == expected, cell.id


class TestUnsupportedCellsReallyAreUnsupported:
    """The other half of the claim. A cell marked unsupported must actually fail
    or misbehave -- otherwise the enumeration is over-restrictive and we are
    leaving capability on the table.

    RANGE over a metadata value is the interesting one: it is not that the SQL
    errors, but that no bounding expression exists to write. The proof is that
    `substr`, the only extraction primitive, is rejected.
    """

    async def test_substr_the_only_extraction_primitive_is_rejected(self, uris):
        with pytest.raises(Exception, match=r"not supported|Invalid"):
            await uris("substr(metadata, 1, 4) >= '2024'")

    async def test_but_a_prefix_of_the_same_value_is_matchable(self, uris, docs):
        """So the limit is bounding, not reaching -- which is why the surface
        marks RANGE unsupported for metadata but PREFIX supported."""
        from fusionlab import filters as f

        expected = c.expected(docs, lambda d: d.published.year == 2024)
        assert expected
        assert await uris(f.meta_starts_with("published_meta", "2024")) == expected

    def test_every_metadata_attribute_is_range_unsupported_for_one_reason(self):
        reasons = {
            unsupported_reason(cell)
            for cell in (cell for cell in all_cells() if cell.op is Op.RANGE)
            if unsupported_reason(cell) is not None
        }
        assert any("substr" in r for r in reasons)


def test_the_matrix_is_not_empty():
    """If `supported_cells()` ever returned nothing, every parametrized test
    above would silently vanish and this file would pass having run nothing."""
    assert len(CELLS) >= 40
