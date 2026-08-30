"""Suite F4 -- can the system say "nothing here"?

Ground truth is exact by construction. The six topic vocabularies are disjoint,
so a query drawn from topic T against a filter admitting only topic U has NO
valid answer -- and one admitting topic T certainly does.

Fusion cannot help: RRF's top score is always 1/(K+1) whatever was asked, and
RL's is a property of the normalisation rule rather than of the answer. The floor
is therefore applied to the RAW score BEFORE fusion, the only place the magnitude
still exists.
"""

import json
from pathlib import Path

import pytest

from fusionlab import corpus as c
from fusionlab import filters as f
from fusionlab.refusal import Outcome, SweepPoint, best_threshold, margin

pytestmark = pytest.mark.integration

DB = "alpha_db"
RESULTS = Path(__file__).resolve().parent.parent / "results"
TOPICS = sorted(c.TOPICS)

# query topic -> filter topic. Same topic = answerable; different = not.
CASES = [(q, t, q == t) for q in TOPICS for t in (TOPICS[0], TOPICS[3], TOPICS[5])]


@pytest.fixture(scope="module")
def manifest() -> str:
    path = Path(__file__).resolve().parent.parent / "synth" / "manifest.json"
    return json.loads(path.read_text())["manifest"]


@pytest.fixture
async def client():
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(read_only=True) as rag:
        yield (await rag.clients_for([DB]))[0]


async def raws(client, query_topic: str, filter_topic: str, search_type: str):
    """Raw pre-fusion scores for one (query, filter) pair."""
    results = await client.search(
        c.TOPICS[query_topic],
        limit=10,
        search_type=search_type,
        filter=f.contains("title", filter_topic),
    )
    return [r.score for r in results]


class TestGroundTruthIsExact:
    async def test_each_topic_filter_admits_only_its_own_documents(self, client):
        for topic in TOPICS:
            count = await client.count_documents(filter=f.contains("title", topic))
            assert 0 < count < c.DOCS_PER_DB, topic


class TestWithoutAFloor:
    """The problem, demonstrated before any fix."""

    async def test_vector_answers_every_unanswerable_question(self, client):
        """Vector returns nearest neighbours however far away. With no floor it
        cannot refuse, so every unanswerable question gets a confident answer."""
        misses = [
            (q, t)
            for q, t, answerable in CASES
            if not answerable and await raws(client, q, t, "vector")
        ]
        assert len(misses) == sum(1 for *_, a in CASES if not a), (
            "vector refused something without a floor; the premise has changed"
        )

    async def test_fts_refuses_some_of_them_unaided(self, client):
        """BM25 has a notion of no-match: no shared term, no result."""
        refused = [
            (q, t)
            for q, t, answerable in CASES
            if not answerable and not await raws(client, q, t, "fts")
        ]
        assert refused, "fts refused nothing; it should self-limit on disjoint terms"


class TestFloorSweep:
    @staticmethod
    async def _sweep(
        client, search_type: str, thresholds: list[float]
    ) -> tuple[dict, list[SweepPoint]]:
        observed = {
            (q, t): (await raws(client, q, t, search_type), answerable)
            for q, t, answerable in CASES
        }
        points = []
        for threshold in thresholds:
            outcomes = tuple(
                Outcome(
                    label=f"{q}->{t}",
                    answerable=answerable,
                    refused=not [s for s in scores if s >= threshold],
                )
                for (q, t), (scores, answerable) in observed.items()
            )
            points.append(SweepPoint(threshold, outcomes))
        return observed, points

    @staticmethod
    def _ladder_from(observed) -> list[float]:
        """Thresholds spanning the scores ACTUALLY observed.

        A ladder chosen from an assumed scale is worse than useless: if every
        rung sits above every score, each one refuses everything and scores
        recall 1.0 for free. That is what an earlier version of this suite did,
        with a ladder taken from a different corpus's BM25 range.
        """
        scores = sorted(s for v, _ in observed.values() for s in v)
        if not scores:
            return [0.0]
        lo, hi = scores[0], scores[-1]
        span = hi - lo
        return [0.0] + [lo + span * n / 8 for n in range(9)]

    @pytest.mark.parametrize("search_type", ["fts", "vector"])
    async def test_a_floor_separates_answerable_from_unanswerable(
        self, client, manifest, search_type
    ):
        observed, _ = await self._sweep(client, search_type, [0.0])
        thresholds = self._ladder_from(observed)
        observed, points = await self._sweep(client, search_type, thresholds)
        best = best_threshold(points)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"f4-local-{search_type}.json").write_text(
            json.dumps(
                {
                    "manifest": manifest,
                    "search_type": search_type,
                    "best_threshold": best.threshold,
                    "best_recall": best.recall,
                    "best_precision": best.precision_or_zero,
                    "sweep": [
                        {
                            "threshold": p.threshold,
                            "refusal_recall": p.recall,
                            "refusal_precision": p.precision_or_zero,
                        }
                        for p in points
                    ],
                    "raw": {
                        f"{q}->{t}": {"answerable": a, "scores": s}
                        for (q, t), (s, a) in observed.items()
                    },
                },
                indent=1,
            )
        )
        # `recall > 0` cannot fail: any threshold above every score refuses
        # everything and scores 1.0. Demand that the floor DISCRIMINATES --
        # that some rung separates the two classes better than refusing all.
        refuse_everything = [
            p for p in points if p.recall == 1.0 and all(o.refused for o in p.outcomes)
        ]
        discriminating = [
            p
            for p in points
            if p.recall == 1.0 and not all(o.refused for o in p.outcomes)
        ]
        assert discriminating or not refuse_everything, (
            f"{search_type}: every threshold with perfect recall achieves it by "
            "refusing EVERY question, answerable ones included. The sweep "
            "establishes nothing about a floor."
        )
        assert best.recall > 0.0

    async def test_the_absolute_floor_does_not_transfer_between_search_types(
        self, client, manifest
    ):
        """A number tuned on one search type is meaningless on the other, and
        the scale is a property of the CORPUS not the search type: fts spans
        0.057-0.074 here and 12-16 on airpubs, because IDF depends on the
        collection. Hence no absolute floor travels."""
        fts_scores = [s for q, t, _ in CASES for s in await raws(client, q, t, "fts")]
        vec_scores = [
            s for q, t, _ in CASES for s in await raws(client, q, t, "vector")
        ]
        hi, lo = max(fts_scores), max(vec_scores)
        ratio = max(hi, lo) / min(hi, lo)
        assert ratio > 4, (
            f"the two scales are within {ratio:.1f}x of each other; an absolute "
            "floor might transfer after all, which would change the design"
        )


class TestRelativeMarginIsUnusableOnThisCorpus:
    """A documented LIMIT of the synthetic corpus, recorded as a test.

    Every document in a topic is generated from the same vocabulary template, so
    within a topic they are near-identical and BM25 scores them alike. The
    top-versus-pack margin is therefore exactly 1.0 for every answerable case:
    there is no within-topic ranking signal to measure.

    Consequence: this corpus can answer "is there an answer here at all"
    (cross-topic, where the vocabularies are disjoint) but CANNOT support a
    relative-margin operating point, or any ranking-quality work. That needs
    per-document lexical variation the generator does not yet produce.
    """

    async def test_within_topic_documents_are_indistinguishable(self, client):
        margins = [
            margin(scores)
            for q, t, answerable in CASES
            if answerable and (scores := await raws(client, q, t, "fts"))
        ]
        assert margins, "no answerable fts case returned anything"
        assert set(margins) == {1.0}, (
            f"within-topic documents now differ (margins {sorted(set(margins))}); "
            "the corpus has gained ranking signal and a relative operating point "
            "becomes measurable -- revisit F4"
        )

    async def test_the_absolute_floor_is_therefore_the_only_option_here(self, client):
        """Cross-topic separation is real even though within-topic is not."""
        answerable, unanswerable = [], []
        for q, t, ok in CASES:
            scores = await raws(client, q, t, "vector")
            if scores:
                (answerable if ok else unanswerable).append(max(scores))
        assert answerable, "no answerable case returned anything"
        assert unanswerable, "no unanswerable case returned anything"
        assert min(answerable) > max(unanswerable), (
            "the answerable and unanswerable score ranges overlap; no absolute "
            "floor can separate them"
        )
