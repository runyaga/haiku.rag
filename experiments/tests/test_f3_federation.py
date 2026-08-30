"""Suite F3 -- filter x federation.

Named F3, not S3: "S3" is object storage, which is a different axis entirely
(suite F6) and must not be confused with this one.

The three questions the library cannot answer, because it takes one filter for
every source and hardcodes the fusion mode:

  * does a filter behave the same federated as it does alone?
  * what does a filter matching in one database and nothing in another produce?
  * how do the fusion modes allocate slots when the filters are lopsided?

Selectivity is controlled exactly through the uri's zero-padded document number:
`doc-0000` matches 1 document, `doc-000` matches 10, `doc-00` matches 100.
"""

import pytest

from fusionlab import corpus as c
from fusionlab import filters as f
from fusionlab.fusion import federated_search

pytestmark = pytest.mark.integration

ALL = list(c.DATABASES)
QUERY = "quartz feldspar mica basalt"  # the mineralogy vocabulary
WIDE, MID, NARROW = "doc-00", "doc-000", "doc-0000"  # 100 : 10 : 1


@pytest.fixture(scope="module")
def docs() -> list[c.SynthDoc]:
    return c.build_corpus()


@pytest.fixture
async def rag():
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(read_only=True) as client:
        yield client


async def counts(rag, predicate: str) -> dict[str, int]:
    """How many documents each database matches, straight from the store."""
    clients = await rag.clients_for(ALL)
    return {
        name: await client.count_documents(filter=predicate)
        for name, client in zip(ALL, clients, strict=True)
    }


def slots(hits) -> dict[str, int]:
    counter: dict[str, int] = {}
    for hit in hits:
        counter[hit.source] = counter.get(hit.source, 0) + 1
    return counter


class TestSelectivityIsControlled:
    """The instrument first: if the filters are not the sizes claimed, every
    allocation result below is measuring something else."""

    @pytest.mark.parametrize(
        ("fragment", "expected"), [(WIDE, 100), (MID, 10), (NARROW, 1)]
    )
    async def test_each_fragment_matches_the_intended_count(
        self, rag, fragment, expected
    ):
        assert set((await counts(rag, f.contains("uri", fragment))).values()) == {
            expected
        }


class TestFilterReachesEverySource:
    async def test_a_shared_filter_narrows_all_three_databases(self, rag, docs):
        got = await counts(rag, f.contains("uri", MID))
        assert got == dict.fromkeys(ALL, 10)

    async def test_a_per_source_filter_narrows_each_differently(self, rag):
        per_source = {
            "alpha_db": f.contains("uri", NARROW),
            "beta_db": f.contains("uri", MID),
            "gamma_db": f.contains("uri", WIDE),
        }
        hits = await federated_search(
            rag, QUERY, sources=ALL, filters=per_source, search_type="fts", limit=30
        )
        # Each source can contribute at most what its own filter admits.
        assert slots(hits).get("alpha_db", 0) <= 1


class TestOneSourceMatchesNothing:
    """A filter valid everywhere but matching in only one database."""

    @pytest.fixture
    def lopsided(self) -> dict[str, str]:
        return {
            "alpha_db": f.contains("uri", WIDE),
            "beta_db": f.meta_eq("owner", "nobody-here"),
            "gamma_db": f.meta_eq("owner", "nobody-here"),
        }

    async def test_it_returns_results_rather_than_erroring(self, rag, lopsided):
        hits = await federated_search(
            rag, QUERY, sources=ALL, filters=lopsided, search_type="fts", limit=6
        )
        assert hits

    async def test_every_slot_goes_to_the_matching_source(self, rag, lopsided):
        """Silently, with nothing distinguishing this from a genuine result."""
        hits = await federated_search(
            rag, QUERY, sources=ALL, filters=lopsided, search_type="fts", limit=6
        )
        assert set(slots(hits)) == {"alpha_db"}

    async def test_all_sources_matching_nothing_returns_empty_not_an_error(self, rag):
        nothing = dict.fromkeys(ALL, f.meta_eq("owner", "nobody-here"))
        hits = await federated_search(
            rag, QUERY, sources=ALL, filters=nothing, search_type="fts", limit=6
        )
        assert hits == []


class TestRRFAllocationIsRigid:
    """RRF reads rank and discards score, so allocation cannot respond to how
    good -- or how selective -- any source is."""

    @pytest.mark.parametrize(
        ("label", "filters"),
        [
            ("1:1:1", {n: f.contains("uri", WIDE) for n in ALL}),
            (
                "1:10:100",
                {
                    "alpha_db": f.contains("uri", MID),
                    "beta_db": f.contains("uri", WIDE),
                    "gamma_db": None,
                },
            ),
        ],
    )
    async def test_every_contributing_source_gets_exactly_one_slot(
        self, rag, label, filters
    ):
        """The split is even across sources that RETURN something.

        Selectivity and relevance are independent: a filter admitting one
        document says nothing about whether that document matches the query, so
        the claim is about contributors, not about the configured set.
        """
        hits = await federated_search(
            rag, QUERY, sources=ALL, filters=filters, search_type="fts", limit=3
        )
        assert hits, "no source contributed; the arm proves nothing"
        assert set(slots(hits).values()) == {1}, label

    async def test_the_top_score_is_the_same_constant_regardless(self, rag):
        """RRF's top score carries no information about the filtered set.

        Both filters are topic-aligned so both certainly admit matches -- the
        point is the SCORE, not whether anything was found.
        """
        wide = {n: f.contains("uri", WIDE) for n in ALL}
        narrow = {n: f.contains("title", "mineralogy") for n in ALL}
        a = await federated_search(
            rag, QUERY, sources=ALL, filters=wide, search_type="fts", limit=3
        )
        b = await federated_search(
            rag, QUERY, sources=ALL, filters=narrow, search_type="fts", limit=3
        )
        assert a, "the wide arm must return something"
        assert b, "the narrow arm must return something"
        assert a[0].score == pytest.approx(1 / 61)
        assert b[0].score == pytest.approx(1 / 61)


class TestRLAllocationRespondsToScores:
    async def test_normalised_rl_can_give_one_source_more_than_its_share(self, rag):
        hits = await federated_search(
            rag,
            QUERY,
            sources=ALL,
            filters={n: f.contains("uri", WIDE) for n in ALL},
            search_type="fts",
            mode="rl",
            limit=6,
        )
        assert max(slots(hits).values()) > 6 // len(ALL)

    async def test_unnormalised_rl_ranks_on_the_raw_score(self, rag):
        """Correct for vector, where every database shares one embedder and the
        same 1/(distance+1) transform, so raws are already comparable."""
        hits = await federated_search(
            rag,
            QUERY,
            sources=ALL,
            search_type="vector",
            mode="rl",
            normalize=False,
            limit=6,
        )
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
        assert all(h.score == h.raw for h in hits)


class TestOneSourceVersusMany:
    async def test_a_single_source_selection_returns_only_that_source(self, rag):
        hits = await federated_search(
            rag, QUERY, sources=["beta_db"], search_type="fts", limit=6
        )
        assert set(slots(hits)) == {"beta_db"}

    async def test_a_filter_admits_the_same_documents_alone_as_federated(self, rag):
        """The filter itself must not change meaning when other databases join."""
        predicate = f.contains("uri", MID)
        alone = await federated_search(
            rag,
            QUERY,
            sources=["alpha_db"],
            filters=predicate,
            search_type="fts",
            limit=50,
        )
        together = await federated_search(
            rag, QUERY, sources=ALL, filters=predicate, search_type="fts", limit=150
        )
        from_alpha = {h.chunk_id for h in together if h.source == "alpha_db"}
        assert {h.chunk_id for h in alone} <= from_alpha

    async def test_an_unknown_source_is_rejected(self, rag):
        with pytest.raises(Exception, match=r"nope|[Uu]nknown"):
            await federated_search(rag, QUERY, sources=["nope"], search_type="fts")


class TestFetchDepth:
    async def test_rrf_output_is_identical_at_every_depth(self, rag):
        """Rank-only fusion cannot promote anything ranked worse than `limit`."""
        runs = [
            await federated_search(
                rag,
                QUERY,
                sources=ALL,
                search_type="fts",
                mode="rrf",
                limit=6,
                fetch_multiplier=m,
            )
            for m in (1, 3, 10)
        ]
        assert all(
            [h.chunk_id for h in run] == [h.chunk_id for h in runs[0]] for run in runs
        )

    async def test_rl_output_changes_with_depth(self, rag):
        """Depth anchors the normalisation floor nearer the true background."""
        shallow = await federated_search(
            rag,
            QUERY,
            sources=ALL,
            search_type="fts",
            mode="rl",
            limit=6,
            fetch_multiplier=1,
        )
        deep = await federated_search(
            rag,
            QUERY,
            sources=ALL,
            search_type="fts",
            mode="rl",
            limit=6,
            fetch_multiplier=10,
        )
        assert [h.chunk_id for h in shallow] != [h.chunk_id for h in deep]
