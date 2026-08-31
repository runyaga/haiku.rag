"""Suite F3b -- every meaningful cell of the federated surface.

GAP THIS CLOSES. `federation.py` enumerated 108 cells (62 meaningful) but
nothing ran them: `to_kwargs` appeared only in the enumeration's own unit test.
This iterates all 62 against the three real databases.

The cross is: filter shape {none, shared, per-source, partial}
            x search type {fts, vector, hybrid}
            x fusion      {rrf, rl-normalised, rl-raw}
            x scope       {1, 2, 3 databases}

HYPOTHESES under test, each asserted per cell:

  H1  Every document behind every returned chunk satisfies that source's own
      filter. A per-source filter must not leak across sources -- this is the
      capability haiku.rag does not have, so nothing upstream tests it.
  H2  A source whose filter admits nothing contributes nothing, and the query
      still succeeds. (PARTIAL and lopsided shapes.)
  H3  RRF's fused score is always 1/(K+1) at rank 0, whatever the cell.
  H4  No cell returns more than `limit`.
"""

import pytest

from fusionlab import corpus as c
from fusionlab import filters as f
from fusionlab.federation import FilterShape, meaningful_cells, to_kwargs
from fusionlab.fusion import RRF_K, federated_search
from fusionlab.observability import cell_span, configure

pytestmark = pytest.mark.integration

NAMES = list(c.DATABASES)
CELLS = meaningful_cells()
QUERY = "quartz feldspar mica basalt granite"
LIMIT = 6

# Admits a known, strict subset of every database: level 4 or 5.
PREDICATE = f.any_of(f.meta_eq("level", 4), f.meta_eq("level", 5))

configure()


@pytest.fixture(scope="module")
def allowed() -> dict[str, set[str]]:
    """Per database, the uris the predicate admits. The oracle for H1."""
    docs = c.build_corpus()
    return {
        name: {d.uri for d in docs if d.db == name and d.level in (4, 5)}
        for name in NAMES
    }


@pytest.fixture(scope="module")
def everything() -> dict[str, set[str]]:
    docs = c.build_corpus()
    return {name: {d.uri for d in docs if d.db == name} for name in NAMES}


@pytest.fixture
async def rag():
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(read_only=True) as client:
        yield client


def _uri_of(hit, by_content: dict[str, str]) -> str | None:
    return by_content.get(hit.content)


@pytest.fixture(scope="module")
def uri_by_content() -> dict[str, str]:
    """Chunk text is the document's text verbatim (one chunk per document)."""
    return {d.text: d.uri for d in c.build_corpus()}


class TestTheOracleIsStrict:
    def test_the_predicate_admits_a_strict_subset_of_every_database(
        self, allowed, everything
    ):
        for name in NAMES:
            assert allowed[name], name
            assert len(allowed[name]) < len(everything[name]), name


class TestEveryMeaningfulCell:
    @pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.id)
    async def test_the_cell_holds_its_hypotheses(
        self, rag, cell, allowed, uri_by_content
    ):
        kwargs = to_kwargs(cell, NAMES, PREDICATE)
        with cell_span(
            "f3b.federated_cell",
            cell=cell.id,
            filter_shape=cell.shape.value,
            search_type=cell.search.value,
            fusion=cell.fusion.value,
            sources=cell.sources,
        ) as span:
            hits = await federated_search(rag, QUERY, limit=LIMIT, **kwargs)
            if span is not None:
                span.set_attribute("hits", len(hits))
                span.set_attribute("sources_hit", len({h.source for h in hits}))

        # An empty result makes every assertion below vacuous: H1's loop does
        # not execute, H3 is guarded, H4 is trivially true. The predicate admits
        # 120-130 documents in every database, so an empty result means the cell
        # is broken, not merely unlucky.
        assert hits, (
            f"{cell.id}: returned nothing. The predicate admits documents in "
            "every source, so every assertion below would pass vacuously."
        )

        # H4 -- never more than asked for.
        assert len(hits) <= LIMIT, cell.id

        # H1 -- no hit may come from a document its own source's filter excludes.
        filters = kwargs["filters"]
        for hit in hits:
            uri = _uri_of(hit, uri_by_content)
            assert uri is not None, f"{cell.id}: unrecognised chunk content"
            filtered = filters is not None and (
                filters is PREDICATE or hit.source in filters
            )
            if filtered:
                assert uri in allowed[hit.source], (
                    f"{cell.id}: {hit.source} returned {uri}, which its own "
                    "filter excludes -- a per-source filter leaked"
                )

        # H3 -- RRF's head score is a constant of the fusion rule.
        if hits and cell.fusion.value == "rrf":
            assert hits[0].score == pytest.approx(1 / (RRF_K + 1)), cell.id


class TestH2SourcesMatchingNothingContributeNothing:
    """H2 was declared in this file's docstring and never asserted. The suite's
    PREDICATE admits 120-130 documents per database, so no cell in the matrix
    exercises a filter that admits nothing."""

    @pytest.mark.parametrize("search_type", ["fts", "vector", "hybrid"])
    async def test_one_empty_source_contributes_nothing_and_does_not_error(
        self, rag, search_type
    ):
        empty_for_beta = {
            "alpha_db": PREDICATE,
            "beta_db": f.meta_eq("owner", "nobody-here"),
            "gamma_db": PREDICATE,
        }
        hits = await federated_search(
            rag,
            QUERY,
            sources=NAMES,
            filters=empty_for_beta,
            search_type=search_type,
            limit=LIMIT,
        )
        assert hits, "the two matching sources must still answer"
        assert "beta_db" not in {h.source for h in hits}

    @pytest.mark.parametrize("search_type", ["fts", "vector", "hybrid"])
    async def test_every_source_empty_returns_empty_without_raising(
        self, rag, search_type
    ):
        nothing = dict.fromkeys(NAMES, f.meta_eq("owner", "nobody-here"))
        hits = await federated_search(
            rag,
            QUERY,
            sources=NAMES,
            filters=nothing,
            search_type=search_type,
            limit=LIMIT,
        )
        assert hits == []


class TestPartialShapeLeavesSourcesOpen:
    """The shape haiku.rag cannot express at all: some sources filtered, the
    rest deliberately unfiltered in the SAME query."""

    async def test_an_unfiltered_source_may_return_excluded_documents(
        self, rag, allowed, uri_by_content
    ):
        cell = next(
            c_
            for c_ in CELLS
            if c_.shape is FilterShape.PARTIAL
            and c_.sources == 3
            and c_.search.value == "fts"
        )
        kwargs = to_kwargs(cell, NAMES, PREDICATE)
        hits = await federated_search(rag, QUERY, limit=30, **kwargs)
        unfiltered = {h.source for h in hits} - set(kwargs["filters"])
        assert unfiltered, (
            "no unfiltered source contributed, so this arm cannot distinguish "
            "partial from shared"
        )
        for h in hits:
            assert _uri_of(h, uri_by_content) is not None, (
                f"unrecognised chunk content from {h.source}: the content->uri "
                "map has broken (chunking changed?), and `None not in allowed` "
                "would silently satisfy the assertion below instead of failing"
            )
        outside = [
            h
            for h in hits
            if h.source in unfiltered
            and _uri_of(h, uri_by_content) not in allowed[h.source]
        ]
        assert outside, (
            "every hit from the UNFILTERED sources happened to satisfy the "
            "filter anyway; the arm proves nothing about partial shape"
        )
