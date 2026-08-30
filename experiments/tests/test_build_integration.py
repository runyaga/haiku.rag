"""Integration: the built corpora, against the oracle. Needs a real database.

Deselected by default (`-m 'not integration'`). This is also the coverage for
`build.py`, which is excluded from the unit gate because it exists to call
LanceDB.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from fusionlab import corpus as c
from fusionlab import filters as f

pytestmark = pytest.mark.integration

SYNTH = Path(__file__).resolve().parent.parent / "synth"


@pytest.fixture(scope="module")
def docs() -> list[c.SynthDoc]:
    return c.build_corpus()


@pytest.fixture(scope="module")
def manifest() -> str:
    return json.loads((SYNTH / "manifest.json").read_text())["manifest"]


def test_the_built_corpus_matches_the_spec(docs, manifest):
    """A result whose manifest does not match the tree is not evidence."""
    assert c.manifest_hash(docs) == manifest


def test_truth_json_has_a_row_per_document(docs):
    assert len(json.loads((SYNTH / "truth.json").read_text())) == len(docs)


async def _uris(rag, source: str, predicate: str | None) -> set[str]:
    client = (await rag.clients_for([source]))[0]
    found = await client.list_documents(limit=10_000, filter=predicate)
    return {d.uri for d in found if d.uri}


@pytest.fixture
async def rag():
    from haiku.rag.client import HaikuRAG

    async with HaikuRAG(read_only=True) as client:
        yield client


class TestCorpusLanded:
    async def test_every_database_holds_its_documents(self, rag):
        for name in c.DATABASES:
            client = (await rag.clients_for([name]))[0]
            assert await client.count_documents() == c.DOCS_PER_DB


class TestDateRangeIsReal:
    """The capability the whole uri-encoding design exists for."""

    async def test_a_uri_date_range_returns_exactly_the_oracle_set(self, rag, docs):
        start, end = date(2024, 1, 1), date(2024, 4, 1)
        predicate = f.uri_date_range(start, end)
        expected = c.expected(
            docs,
            lambda d: d.db == "alpha_db" and start <= d.published < end,
        )
        assert await _uris(rag, "alpha_db", predicate) == expected
        assert expected, "the window must not be empty, or this proves nothing"

    async def test_the_range_discriminates(self, rag, docs):
        """A filter returning everything is indistinguishable from one ignored."""
        everything = await _uris(rag, "alpha_db", None)
        window = await _uris(
            rag, "alpha_db", f.uri_date_range(date(2024, 1, 1), date(2024, 4, 1))
        )
        assert 0 < len(window) < len(everything)

    async def test_an_empty_window_returns_nothing(self, rag):
        predicate = f.uri_date_range(date(2024, 1, 1), date(2024, 1, 1))
        assert await _uris(rag, "alpha_db", predicate) == set()

    async def test_a_metadata_date_cannot_be_ranged(self, rag):
        """The controlled half of the pair: substr is unsupported, so the JSON
        copy of the same date supports equality only."""
        with pytest.raises(Exception, match=r"not supported|Invalid"):
            await _uris(
                rag,
                "alpha_db",
                "substr(metadata, 1, 4) >= '2024'",
            )

    async def test_but_a_metadata_date_can_be_matched_exactly(self, rag, docs):
        # A date the corpus actually contains. An arbitrary date makes both
        # sides empty and the assertion vacuous -- it passes while proving
        # nothing, which is the failure mode this whole suite exists to avoid.
        day = max(
            (d.published for d in docs if d.db == "alpha_db"),
            key=lambda p: sum(
                1 for d in docs if d.db == "alpha_db" and d.published == p
            ),
        )
        expected = c.expected(docs, lambda d: d.db == "alpha_db" and d.published == day)
        assert expected, "no documents on the chosen day; the test would be vacuous"
        got = await _uris(rag, "alpha_db", f.meta_eq("published_meta", day.isoformat()))
        assert got == expected


class TestCollisionTrap:
    """Over alpha_db: 158 documents carry the alpha tag, a naive LIKE matches
    166. (Whole-corpus those figures are 461 and 489 -- every test in this class
    queries alpha_db alone, so the smaller pair is the relevant one.)"""

    async def test_anchored_regex_returns_exactly_the_tagged_documents(self, rag, docs):
        expected = c.expected(docs, lambda d: d.db == "alpha_db" and "alpha" in d.tags)
        assert await _uris(rag, "alpha_db", f.meta_has("tags", "alpha")) == expected

    async def test_a_naive_like_over_matches_and_the_gap_is_the_trap(self, rag, docs):
        naive = "metadata LIKE '%alpha%'"
        truth = c.expected(docs, lambda d: d.db == "alpha_db" and "alpha" in d.tags)
        got = await _uris(rag, "alpha_db", naive)
        assert got > truth, "the trap did not fire; the corpus is wrong"
        assert got - truth, "no false positives means no trap"


class TestEveryAttributeType:
    @pytest.mark.parametrize(
        ("predicate", "oracle"),
        [
            (f.meta_eq("active", True), lambda d: d.active),
            (f.meta_eq("level", 3), lambda d: d.level == 3),
            (f.meta_eq("owner", "north"), lambda d: d.owner == "north"),
            # NOT a key-presence case: `lambda d: True` matched all 300, which
            # the strict-subset guard now correctly rejects. Key presence is
            # tested where it can discriminate -- against a key only some
            # documents carry.
            (f.meta_eq("level", 5), lambda d: d.level == 5),
        ],
        ids=["boolean", "integer", "enum", "integer-high"],
    )
    async def test_it_returns_the_oracle_set(self, rag, docs, predicate, oracle):
        in_db = [d for d in docs if d.db == "alpha_db"]
        expected = c.expected(in_db, oracle)
        assert expected, "an empty oracle makes the comparison vacuous"
        assert len(expected) < len(in_db), (
            "the oracle matches every document; a filter that returns everything "
            "is indistinguishable from one being silently ignored"
        )
        assert await _uris(rag, "alpha_db", predicate) == expected
