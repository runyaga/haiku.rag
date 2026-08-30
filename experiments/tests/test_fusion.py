"""RRF and RL, and the fan-out around them. No network: the client is a stub."""

import pytest

from conftest import StubClient, chunks
from fusionlab.fusion import (
    RRF_K,
    Hit,
    federated_search,
    fuse_rl,
    fuse_rrf,
    resolve_fetch,
    resolve_filters,
)


def _call(client: StubClient, source: str, index: int = 0) -> dict:
    """What `source`'s repository was actually asked on call `index`."""
    return client._owners[source].chunk_repository.calls[index]


class TestRRF:
    def test_score_is_the_reciprocal_rank_and_ignores_raw(self):
        hits = fuse_rrf({"a": chunks("a", [99.0, 0.001])})
        assert hits[0].score == pytest.approx(1 / (RRF_K + 1))
        assert hits[1].score == pytest.approx(1 / (RRF_K + 2))
        assert hits[0].raw == 99.0

    def test_disjoint_sources_interleave_round_robin_in_source_order(self):
        """Nothing appears twice, so nothing is ever summed. Ties keep insertion
        order, which makes the configured source order a ranking parameter."""
        hits = fuse_rrf(
            {"a": chunks("a", [0.01, 0.01]), "b": chunks("b", [99.0, 99.0])}
        )
        assert [h.source for h in hits] == ["a", "b", "a", "b"]

    def test_a_weak_source_listed_first_still_takes_rank_zero(self):
        hits = fuse_rrf({"b": chunks("b", [99.0]), "a": chunks("a", [0.01])})
        assert hits[0].source == "b"

    def test_no_sources_gives_no_hits(self):
        assert fuse_rrf({}) == []

    def test_an_empty_source_contributes_nothing(self):
        assert [h.source for h in fuse_rrf({"a": [], "b": chunks("b", [1.0])})] == ["b"]

    def test_k_is_configurable(self):
        assert fuse_rrf({"a": chunks("a", [1.0])}, k=0)[0].score == pytest.approx(1.0)


class TestRL:
    def test_min_max_puts_the_best_at_one_and_the_worst_at_zero(self):
        hits = fuse_rl({"a": chunks("a", [0.9, 0.5, 0.1])})
        assert [h.score for h in hits] == pytest.approx([1.0, 0.5, 0.0])

    def test_a_flat_span_scores_zero_rather_than_being_promoted(self):
        """A source whose scores are identical carries no ordering. Anchoring it
        at 1.0 would let a uniformly-bad source outrank a genuinely good one."""
        hits = fuse_rl({"a": chunks("a", [0.4, 0.4, 0.4])})
        assert [h.score for h in hits] == [0.0, 0.0, 0.0]

    def test_without_normalisation_the_raw_score_ranks_directly(self):
        """Correct for `vector`: every database applies the same
        1/(distance+1) transform and shares an embedder, so raws are already
        comparable and normalising only destroys that."""
        hits = fuse_rl(
            {"a": chunks("a", [0.90]), "b": chunks("b", [0.40])}, normalize=False
        )
        assert [h.source for h in hits] == ["a", "b"]
        assert hits[0].score == 0.90

    def test_normalisation_lets_a_weak_source_tie_a_strong_one(self):
        hits = fuse_rl({"a": chunks("a", [0.9, 0.1]), "b": chunks("b", [0.4, 0.39])})
        assert hits[0].score == hits[1].score == 1.0

    def test_a_weight_scales_a_source(self):
        hits = fuse_rl(
            {"a": chunks("a", [1.0, 0.0]), "b": chunks("b", [1.0, 0.0])},
            weights={"b": 0.5},
        )
        assert hits[0].source == "a"

    def test_an_empty_source_is_skipped(self):
        assert [h.source for h in fuse_rl({"a": [], "b": chunks("b", [1.0, 0.0])})] == [
            "b",
            "b",
        ]

    def test_no_sources_gives_no_hits(self):
        assert fuse_rl({}) == []


class TestResolveFetch:
    def test_rrf_defaults_to_no_over_fetch(self):
        """Measured: RRF output is identical at multipliers 1, 3, 5 and 10."""
        assert resolve_fetch("rrf", 6, None) == 6

    def test_rl_defaults_to_over_fetching(self):
        """Depth anchors the normalisation floor nearer the true background."""
        assert resolve_fetch("rl", 6, None) == 30

    def test_an_explicit_multiplier_overrides_the_default(self):
        assert resolve_fetch("rrf", 6, 4) == 24

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_multiplier_below_one_is_refused(self, bad):
        with pytest.raises(ValueError, match="must be >= 1"):
            resolve_fetch("rl", 6, bad)


class TestResolveFilters:
    def test_none_applies_to_every_source(self):
        assert resolve_filters(None, ["a", "b"]) == {"a": None, "b": None}

    def test_a_string_applies_to_every_source(self):
        assert resolve_filters("x = 1", ["a", "b"]) == {"a": "x = 1", "b": "x = 1"}

    def test_a_mapping_gives_each_source_its_own(self):
        assert resolve_filters({"a": "x = 1"}, ["a", "b"]) == {"a": "x = 1", "b": None}


class TestFederatedSearch:
    async def test_no_sources_returns_empty_without_touching_the_client(self):
        """The library returns [] for an empty selection; match it rather than
        raising from clients[0]."""
        client = StubClient({})
        hits = await federated_search(client, "q", sources=[], search_type="vector")
        assert hits == []
        assert client.requested == []

    async def test_fts_never_embeds(self, two_sources):
        await federated_search(two_sources, "q", sources=["alpha"], search_type="fts")
        assert two_sources._owners["alpha"].embedder.queries == []

    async def test_vector_embeds_once_for_the_whole_set(self, two_sources):
        await federated_search(
            two_sources, "q", sources=["alpha", "beta"], search_type="vector"
        )
        assert two_sources._owners["alpha"].embedder.queries == ["q"]
        assert two_sources._owners["beta"].embedder.queries == []

    async def test_each_source_receives_its_own_filter(self, two_sources):
        await federated_search(
            two_sources,
            "q",
            sources=["alpha", "beta"],
            filters={"alpha": "a = 1", "beta": "b = 2"},
            search_type="fts",
        )
        assert _call(two_sources, "alpha")["filter"] == "a = 1"
        assert _call(two_sources, "beta")["filter"] == "b = 2"

    async def test_the_result_is_cut_to_limit(self, two_sources):
        hits = await federated_search(
            two_sources, "q", sources=["alpha", "beta"], search_type="fts", limit=3
        )
        assert len(hits) == 3

    async def test_rl_and_rrf_order_the_same_candidates_differently(
        self, divergent_shapes
    ):
        """RRF interleaves by rank; RL respects each source's own distribution.

        Uses `divergent_shapes` rather than `two_sources`: under min-max both
        sources' tops become 1.0, so the modes can only differ in the TAIL, and
        the tail must be separated by more than float noise for the assertion
        to mean anything. Here it is separated by 0.475.
        """
        common = {"sources": ["alpha", "beta"], "search_type": "fts", "limit": 4}
        rrf = await federated_search(divergent_shapes, "q", mode="rrf", **common)
        rl = await federated_search(divergent_shapes, "q", mode="rl", **common)
        assert [h.source for h in rrf] == ["alpha", "beta", "alpha", "beta"]
        assert [h.source for h in rl] == ["alpha", "beta", "beta", "alpha"]

    async def test_min_max_makes_every_source_top_out_at_one(self, two_sources):
        """Why the modes can only diverge in the tail -- and the reason an
        earlier version of the test above passed on a one-ulp accident."""
        rl = fuse_rl(
            {
                "alpha": [(c, s) for c, s in chunks("alpha", [0.9, 0.6, 0.3])],
                "beta": [(c, s) for c, s in chunks("beta", [0.4, 0.39, 0.38])],
            }
        )
        tops = [h.score for h in rl if h.rank == 0]
        assert tops == [1.0, 1.0]

    async def test_a_floor_drops_candidates_before_fusion(self, two_sources):
        """After fusion the magnitude is gone, so a floor could not work there."""
        hits = await federated_search(
            two_sources, "q", sources=["alpha", "beta"], search_type="fts", floor=0.5
        )
        assert {h.source for h in hits} == {"alpha"}
        assert all(h.raw >= 0.5 for h in hits)

    async def test_a_floor_above_everything_returns_nothing(self, two_sources):
        assert (
            await federated_search(
                two_sources, "q", sources=["alpha"], search_type="fts", floor=99.0
            )
            == []
        )

    async def test_fetch_depth_reaches_the_repository(self, two_sources):
        await federated_search(
            two_sources, "q", sources=["alpha"], search_type="fts", mode="rl", limit=2
        )
        assert _call(two_sources, "alpha")["limit"] == 10

    async def test_hits_carry_source_raw_and_rank(self, two_sources):
        hits = await federated_search(
            two_sources, "q", sources=["alpha"], search_type="fts", limit=1
        )
        assert isinstance(hits[0], Hit)
        assert (hits[0].source, hits[0].raw, hits[0].rank) == ("alpha", 0.9, 0)
