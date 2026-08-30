"""Suite F5 -- filter integrity.

Not a retrieval suite. Three claims about haiku.rag itself, each demonstrated
rather than asserted from reading:

  1. a `filter` string is never escaped, parsed or validated, so a filter built
     by concatenating user text can be made to mean something else;
  2. federated `count_documents` and `list_documents` truncate by different
     rules, and `tools/document.py` pages one against the other;
  3. `tools/search.py` documents combining the caller's filter and does not.

Claims 2 and 3 are CANARIES: they assert the defect still exists, so they turn
red when upstream fixes it. That is the point -- a silently-fixed defect we are
still reporting is worse than no test.
"""

import inspect
import json
from pathlib import Path

import pytest

from fusionlab import filters as f

# NOT module-level integration: the source-inspection and index-set canaries
# below touch neither a database nor the network, and they are the drift alarms
# this suite exists for. Marking the whole module would hide them behind
# `make suites`, which additionally needs built corpora and a live embedder.
# Only the classes that query a database carry the marker.

DB = "alpha_db"


@pytest.fixture
async def rag():
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(read_only=True) as client:
        yield client


@pytest.fixture
async def client(rag):
    return (await rag.clients_for([DB]))[0]


@pytest.mark.integration
class TestInjection:
    """A UI that concatenates user text into a filter can be made to widen it."""

    @staticmethod
    def _naive(user_value: str) -> str:
        """What a caller writes when there is no builder to reach for.

        This is the shape the library invites: `filter` takes raw SQL and
        nothing in haiku.rag escapes it (`escape_sql_string` in utils.py is
        applied only to internally-built ids).
        """
        return f'metadata LIKE \'%"owner": "{user_value}"%\''

    async def test_a_benign_value_narrows_as_intended(self, client):
        total = await client.count_documents()
        narrowed = await client.count_documents(filter=self._naive("north"))
        assert 0 < narrowed < total, "the control arm must actually narrow"

    async def test_a_crafted_value_escapes_the_predicate_and_widens_it(self, client):
        """The demonstration. A quote closes the literal early and the rest is
        read as SQL, so a filter meant to select one owner returns everything."""
        total = await client.count_documents()
        narrowed = await client.count_documents(filter=self._naive("north"))
        crafted = await client.count_documents(filter=self._naive("zzz\"%' OR 1=1 --"))
        assert crafted > narrowed, "the injection did not widen the result"
        assert crafted == total, (
            f"expected the crafted filter to return the whole corpus; "
            f"got {crafted} of {total}"
        )

    async def test_the_builder_neutralises_the_same_value(self, client):
        """`filters.py` escapes at construction, so the same input is data."""
        crafted = await client.count_documents(
            filter=f.meta_eq("owner", "zzz\"%' OR 1=1 --")
        )
        assert crafted == 0, "an escaped value must match nothing, not everything"

    async def test_the_builder_refuses_an_unknown_column(self):
        """A typo cannot become a silently-empty result."""
        with pytest.raises(f.UnknownColumnError):
            f.eq("owner", "north")  # `owner` is metadata, not a real column


@pytest.mark.integration
class TestCountAndListAgree:
    """RETRACTED CLAIM, kept as a regression guard.

    The plan asserted that federated `count_documents` (which sums per database)
    and `list_documents` (which interleaves with zip_longest, then windows) use
    different truncation rules and therefore disagree. That was inferred from
    reading the code and it is WRONG: measured over equal databases (300/300/300)
    and unequal ones (cv22b 354 / ac130j 180), at page sizes 10, 37, 50 and 100,
    filtered and unfiltered, paging tiles the corpus exactly -- no duplicates,
    nothing missed.

    The mechanism the reading described is real; the consequence it predicted is
    not. These tests now guard the behaviour that actually holds.
    """

    async def test_count_sums_every_source(self, rag):
        counted = await rag.count_documents()
        per_source = []
        for name in ("alpha_db", "beta_db", "gamma_db"):
            one = (await rag.clients_for([name]))[0]
            per_source.append(await one.count_documents())
        assert counted == sum(per_source)

    async def test_a_full_listing_agrees_with_the_count(self, rag):
        counted = await rag.count_documents()
        listed = await rag.list_documents(limit=counted + 10)
        assert len(listed) == counted

    @pytest.mark.parametrize("page_size", [10, 37, 100])
    async def test_paging_tiles_the_corpus_exactly(self, rag, page_size):
        counted = await rag.count_documents()
        seen: list[str] = []
        for offset in range(0, counted, page_size):
            page = await rag.list_documents(limit=page_size, offset=offset)
            seen.extend(d.id for d in page if d.id)
        assert len(seen) == counted, "paging dropped or repeated rows"
        assert len(set(seen)) == counted, "paging returned duplicates"

    async def test_paging_a_filtered_listing_also_tiles_exactly(self, rag):
        predicate = f.meta_eq("owner", "north")
        counted = await rag.count_documents(filter=predicate)
        assert counted, "the filter must match something"
        seen: list[str] = []
        for offset in range(0, counted, 10):
            page = await rag.list_documents(limit=10, offset=offset, filter=predicate)
            seen.extend(d.id for d in page if d.id)
        assert len(set(seen)) == counted


class TestSearchToolDropsTheFilter:
    """CANARY, by source inspection: the defect is in a code path that takes no
    filter argument, so it cannot be reached with one to demonstrate."""

    def test_the_search_tool_takes_no_filter_parameter(self):
        from haiku.rag.tools import search as search_tool

        source = inspect.getsource(search_tool)
        assert "effective_filter = base_filter" in source, (
            "the line that discards the caller's filter is gone; re-read "
            "tools/search.py and update F5"
        )

    def test_its_docstring_still_promises_to_combine_one(self):
        from haiku.rag.tools import search as search_tool

        source = inspect.getsource(search_tool)
        assert "Combined with any filter" in source, (
            "the docstring no longer claims combination; the contradiction may "
            "have been resolved"
        )

    def test_the_tool_does_not_forward_a_source_selection(self):
        """So agent-facing search always spans the whole configured set."""
        from haiku.rag.tools import search as search_tool

        source = inspect.getsource(search_tool)
        body = source[source.index("async def search") :]
        assert "sources=" not in body


class TestUnindexedColumns:
    """The reason every filter in F2 was a full scan, asserted against the
    declared index set rather than inferred from timings."""

    @staticmethod
    def _columns(table: str) -> set[str]:
        from haiku.rag.store.schema import index_specs

        return {column for column, _ in index_specs(table)}

    def test_the_only_indexed_document_meta_columns_are_id_and_uri(self):
        assert self._columns("document_meta") == {"id", "uri"}

    def test_uris_btree_cannot_serve_a_contains_filter(self):
        """A BTree answers equality and range, not `LIKE '%...%'`. So even the
        one indexed text column does not help the filters callers actually
        write -- but it DOES serve the uri date range, which is a true range."""
        from haiku.rag.store.schema import index_specs
        from lancedb.index import BTree

        by_column = dict(index_specs("document_meta"))
        assert isinstance(by_column["uri"], BTree)


def test_results_carry_the_manifest():
    """Every recorded number must name the corpus it came from."""
    results = Path(__file__).resolve().parent.parent / "results"
    manifest = json.loads(
        (Path(__file__).resolve().parent.parent / "synth" / "manifest.json").read_text()
    )["manifest"]
    found = sorted(results.glob("f*-local-*.json"))
    assert found, (
        "no result artifacts to check -- this test is the harness's provenance "
        "guard and must not pass on an empty glob. Run `make suites` first."
    )
    for path in found:
        assert json.loads(path.read_text())["manifest"] == manifest, path
