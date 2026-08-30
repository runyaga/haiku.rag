"""The corpus spec and its oracle. Pure: no database, no network, no clock."""

import json
from datetime import date

import pytest

from fusionlab import corpus as c


class TestDeterminism:
    def test_the_same_seed_gives_a_byte_identical_corpus(self):
        assert c.build_corpus() == c.build_corpus()

    def test_a_different_seed_gives_a_different_corpus(self):
        assert c.build_corpus(seed=1) != c.build_corpus(seed=2)

    def test_the_manifest_is_stable_across_builds(self):
        assert c.manifest_hash(c.build_corpus()) == c.manifest_hash(c.build_corpus())

    def test_the_manifest_is_order_independent(self):
        docs = c.build_corpus(docs_per_db=5)
        assert c.manifest_hash(docs) == c.manifest_hash(list(reversed(docs)))

    def test_the_manifest_changes_when_the_corpus_does(self):
        assert c.manifest_hash(c.build_corpus(docs_per_db=5)) != c.manifest_hash(
            c.build_corpus(docs_per_db=6)
        )


class TestShape:
    def test_every_database_gets_its_share(self):
        docs = c.build_corpus(docs_per_db=10)
        assert len(docs) == 30
        for name in c.DATABASES:
            assert sum(d.db == name for d in docs) == 10

    def test_uris_are_unique(self):
        docs = c.build_corpus()
        assert len({d.uri for d in docs}) == len(docs)

    def test_the_date_is_the_first_uri_segment(self):
        """So lexicographic uri order is date order across every database."""
        doc = c.build_corpus(docs_per_db=1)[0]
        assert doc.uri.startswith(f"{c.URI_SCHEME}://{doc.published.isoformat()}/")

    def test_sorting_by_uri_sorts_by_date_across_databases(self):
        docs = c.build_corpus(docs_per_db=40)
        dates = [d.published for d in sorted(docs, key=lambda d: d.uri)]
        assert dates == sorted(dates)

    def test_every_topic_vocabulary_is_disjoint(self):
        seen: set[str] = set()
        for words in c.TOPICS.values():
            terms = set(words.split())
            assert not (terms & seen)
            seen |= terms

    def test_published_stays_inside_the_declared_span(self):
        for doc in c.build_corpus(docs_per_db=50):
            assert (
                c.EPOCH
                <= doc.published
                < c.EPOCH + __import__("datetime").timedelta(days=c.SPAN_DAYS)
            )


class TestMetadata:
    def test_it_carries_every_attribute_type(self):
        meta = c.build_corpus(docs_per_db=1)[0].metadata
        assert set(meta) == {"active", "tags", "owner", "level", "published_meta"}
        assert isinstance(meta["active"], bool)
        assert isinstance(meta["tags"], list)
        assert isinstance(meta["level"], int)

    def test_published_meta_duplicates_the_uri_date(self):
        """The controlled pair: the uri copy supports a range query, this one
        cannot, because substr is unsupported."""
        doc = c.build_corpus(docs_per_db=1)[0]
        assert doc.metadata["published_meta"] == doc.published.isoformat()
        assert doc.published.isoformat() in doc.uri


class TestCollisionTrap:
    def test_the_collider_owner_contains_a_tag_as_a_substring(self):
        assert "alpha" in c.COLLIDER_OWNER
        assert c.COLLIDER_OWNER not in c.TAGS

    def test_colliders_are_planted_at_a_known_rate(self):
        docs = c.build_corpus(docs_per_db=100)
        assert sum(d.owner == c.COLLIDER_OWNER for d in docs) == 15

    def test_a_naive_like_matches_strictly_more_than_the_truth(self):
        """This gap is what an anchored regexp_like must close."""
        docs = c.build_corpus()
        truth = c.expected(docs, lambda d: "alpha" in d.tags)
        naive = c.expected(docs, c.naive_tag_match("alpha"))
        assert truth < naive
        assert naive - truth


class TestOracle:
    def test_expected_returns_the_matching_uris(self):
        docs = c.build_corpus(docs_per_db=10)
        assert c.expected(docs, lambda d: d.db == "alpha_db") == {
            d.uri for d in docs if d.db == "alpha_db"
        }

    def test_expected_can_match_nothing(self):
        assert c.expected(c.build_corpus(docs_per_db=2), lambda _: False) == set()

    def test_in_databases_restricts_the_oracle(self):
        docs = c.build_corpus(docs_per_db=5)
        predicate = c.in_databases("alpha_db", "beta_db")
        assert {d.db for d in docs if predicate(d)} == {"alpha_db", "beta_db"}

    def test_naive_tag_match_sees_any_field(self):
        docs = c.build_corpus(docs_per_db=40)
        colliders = [d for d in docs if d.owner == c.COLLIDER_OWNER]
        untagged = [d for d in colliders if "alpha" not in d.tags]
        assert untagged, "expected at least one collider without the alpha tag"
        assert all(c.naive_tag_match("alpha")(d) for d in untagged)


class TestTruth:
    def test_every_document_becomes_a_json_ready_row(self):
        docs = c.build_corpus(docs_per_db=3)
        rows = c.to_truth(docs)
        assert len(rows) == len(docs)
        json.dumps(rows)

    def test_the_date_is_serialised_as_an_iso_string(self):
        row = c.to_truth(c.build_corpus(docs_per_db=1))[0]
        assert row["published"] == date.fromisoformat(row["published"]).isoformat()

    def test_a_row_carries_the_fields_a_suite_needs(self):
        row = c.to_truth(c.build_corpus(docs_per_db=1))[0]
        assert {"db", "uri", "topic", "tags", "owner", "level"} <= set(row)


def test_the_vocabulary_is_not_the_target_domain():
    """Dressing invented data in the target domain's clothes invites reading a
    plumbing result as a corpus result."""
    aviation = {"engine", "hydraulic", "avionics", "egress", "aircrew", "flight"}
    assert not (aviation & {w for words in c.TOPICS.values() for w in words.split()})


@pytest.mark.parametrize("size", [1, 2, 21])
def test_small_corpora_stay_well_formed(size):
    docs = c.build_corpus(docs_per_db=size)
    assert len(docs) == size * len(c.DATABASES)
    assert all(d.level in range(1, 6) for d in docs)
    assert all(1 <= len(d.tags) <= 3 for d in docs)
