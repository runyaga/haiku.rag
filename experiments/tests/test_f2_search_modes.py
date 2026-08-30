"""Suite F2 -- filter x search mode, and the full-scan slope on local disk.

Two halves. The correctness half asserts; the performance half MEASURES and
records raw observations, asserting only what is structural. A timing assertion
on a shared machine measures the machine.

Nothing relevant is indexed -- `metadata` and `title` carry no index, and
`uri`'s BTree cannot serve `LIKE '%...%'` (store/schema.py) -- so every filter
here is a full scan of document_meta. This is the local-filesystem baseline the
object-store suite (F6) will be compared against.
"""

import json
import time
from pathlib import Path

import pytest

from fusionlab import corpus as c
from fusionlab import filters as f
from fusionlab.measure import Point, Series, monotonic_in, slope

pytestmark = pytest.mark.integration

DB = "alpha_db"
QUERY = "quartz feldspar mica basalt granite"
REPEATS = 5
RESULTS = Path(__file__).resolve().parent.parent / "results"

# Selectivity ladder, exact by construction: the zero-padded document number.
LADDER = [
    ("1", f.contains("uri", "doc-0000"), 1),
    ("10", f.contains("uri", "doc-000"), 10),
    ("100", f.contains("uri", "doc-00"), 100),
    ("300", f.contains("uri", "doc-0"), 300),
]

UUID_CHARS = 39  # "'<36-char uuid>', "


@pytest.fixture(scope="module")
def manifest() -> str:
    path = Path(__file__).resolve().parent.parent / "synth" / "manifest.json"
    return json.loads(path.read_text())["manifest"]


@pytest.fixture
async def client():
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(read_only=True) as rag:
        yield (await rag.clients_for([DB]))[0]


async def timed(client, predicate: str | None, search_type: str) -> list[float]:
    await client.search(QUERY, limit=5, search_type=search_type, filter=predicate)
    out = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        await client.search(QUERY, limit=5, search_type=search_type, filter=predicate)
        out.append(time.perf_counter() - start)
    return out


class TestTheLadderIsExact:
    """The instrument first. If the rungs are not the sizes claimed, every
    timing below is plotted against the wrong x."""

    @pytest.mark.parametrize(("label", "predicate", "expected"), LADDER)
    async def test_each_rung_matches_its_claimed_count(
        self, client, label, predicate, expected
    ):
        assert await client.count_documents(filter=predicate) == expected


class TestFilterNarrowsInEverySearchMode:
    @pytest.mark.parametrize("search_type", ["fts", "vector", "hybrid"])
    async def test_a_filter_restricts_results_to_matching_documents(
        self, client, docs_uris, search_type
    ):
        predicate = f.contains("uri", "doc-000")
        results = await client.search(
            QUERY, limit=20, search_type=search_type, filter=predicate
        )
        assert results, f"{search_type} returned nothing; the arm proves nothing"
        assert all("doc-000" in (r.document_uri or "") for r in results)

    @pytest.mark.parametrize("search_type", ["fts", "vector", "hybrid"])
    async def test_a_filter_matching_nothing_returns_nothing(self, client, search_type):
        results = await client.search(
            QUERY,
            limit=5,
            search_type=search_type,
            filter=f.meta_eq("owner", "nobody-here"),
        )
        assert results == []

    async def test_vector_returns_something_for_an_off_topic_query(self, client):
        """Vector has no notion of no-match: it returns nearest neighbours
        however far away. fts self-limits. This is why a raw-score floor is
        needed, and why the two modes need different floors."""
        off_topic = "chocolate cake baking recipe sugar flour"
        assert await client.search(off_topic, limit=5, search_type="vector")

    async def test_fts_can_return_nothing_for_an_off_topic_query(self, client):
        narrow = f.contains("uri", "doc-0000")
        off_topic = "chocolate cake baking recipe sugar flour"
        assert (
            await client.search(off_topic, limit=5, search_type="fts", filter=narrow)
            == []
        )


class TestFullScanSlope:
    """Measured, not asserted. Raw observations are recorded; only the shape is
    checked, and the shape differs by search type for a measured reason.

    MEASURED: embed_query to bizon costs ~12.6 ms; a full fts search costs
    ~3.3 ms; a full vector search ~17.1 ms. So for vector and hybrid the query
    embedding is a CONSTANT that dominates, and filter selectivity moves the
    total only a few ms. Only fts exposes the scan cost directly.
    """

    @staticmethod
    async def _ladder(client, manifest, search_type: str) -> Series:
        series = Series(f"f2-local-{search_type}", manifest)
        for label, predicate, matched in LADDER:
            series.add(
                Point(
                    label=label,
                    matched=matched,
                    seconds=tuple(await timed(client, predicate, search_type)),
                    predicate_chars=matched * UUID_CHARS,
                )
            )
        series.add(
            Point(
                label="no-filter",
                matched=0,
                seconds=tuple(await timed(client, None, search_type)),
            )
        )
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{series.name}.json").write_text(
            json.dumps(series.to_dict(), indent=1)
        )
        return series

    async def test_fts_cost_rises_with_the_scan_size(self, client, manifest):
        """No embedding, so the scan IS the cost and the slope is visible."""
        series = await self._ladder(client, manifest, "fts")
        assert monotonic_in(series.points), [
            (p.label, round(p.best * 1000, 1)) for p in series.points
        ]
        assert slope(series.points) > 0

    @pytest.mark.parametrize("search_type", ["vector", "hybrid"])
    async def test_embedding_dominates_so_selectivity_barely_moves_the_total(
        self, client, manifest, search_type
    ):
        """The filter is not free -- it is masked. A constant network round trip
        swamps it, which is why fts is the sensitive instrument for comparing
        local disk against object storage."""
        series = await self._ladder(client, manifest, search_type)
        floor = min(p.best for p in series.points)
        ladder_spread = max(p.best for p in series.points) - floor
        moved_ms = ladder_spread * 1000
        floor_ms = floor * 1000
        assert ladder_spread < floor, (
            f"{search_type}: selectivity moved the total by {moved_ms:.1f} ms "
            f"against a {floor_ms:.1f} ms floor -- the embedding no longer "
            "dominates, so this arm now measures the scan and should use the "
            "fts assertion instead"
        )

    async def test_fts_is_the_sensitive_instrument_for_the_object_store_comparison(
        self, client, manifest
    ):
        """Recorded as the baseline F6 will be compared against.

        Compares RELATIVE variation (range / floor), not absolute ranges. An
        earlier version compared `fts_range > vec_range * 0.5` and flaked: the
        vector range is dominated by network jitter to the embedder, so a noisy
        run widened it and failed the test for the OPPOSITE reason to the one
        the message gave. Dividing by each arm's own floor removes the constant
        the arms do not share.
        """
        fts = await self._ladder(client, manifest, "fts")
        vector = await self._ladder(client, manifest, "vector")

        def relative_variation(series) -> float:
            floor = min(p.best for p in series.points)
            return (max(p.best for p in series.points) - floor) / floor

        fts_rv = relative_variation(fts)
        vec_rv = relative_variation(vector)
        assert fts_rv > vec_rv * 2, (
            f"fts varies {fts_rv:.2f}x its floor and vector {vec_rv:.2f}x its "
            "own -- fts is no longer the more sensitive instrument, so the "
            "storage-tier comparison should not be measured through it"
        )


@pytest.fixture(scope="module")
def docs_uris() -> set[str]:
    return {d.uri for d in c.build_corpus() if d.db == DB}
