"""The synthetic corpus specification and its oracle.

Pure: no database, no network, no clock. ``build_corpus`` is a function of its
seed alone, so the 900 documents are byte-identical everywhere, and
``manifest_hash`` proves a result came from the corpus in the tree rather than a
stale one.

The vocabulary is deliberately not aviation. These corpora exercise haiku.rag's
filter plumbing; dressing invented data in a target domain's clothes invites
reading a plumbing result as a corpus result.
"""

import hashlib
import json
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta

SEED = 20260830
DATABASES = ("alpha_db", "beta_db", "gamma_db")
DOCS_PER_DB = 300
"""Above the 256-chunk floor below which no ANN index is ever built."""

TOPICS = {
    "mineralogy": "quartz feldspar mica basalt granite schist cleavage",
    "horology": "escapement mainspring balance jewel tourbillon pallet",
    "cartography": "isoline projection datum contour azimuth graticule",
    "ceramics": "kiln glaze porcelain bisque slipware earthenware",
    "textiles": "warp weft selvedge loom worsted heddle",
    "apiculture": "brood forager nectar propolis queen swarm",
}
TAGS = ("alpha", "bravo", "charlie", "delta")
OWNERS = ("north", "south", "east", "shared")

COLLIDER_OWNER = "alphanumeric-review"
"""Contains the tag ``alpha`` as a substring.

A naive ``LIKE '%alpha%'`` multiselect filter must wrongly match these, and an
anchored ``regexp_like`` must not. A harness that fails to catch that is wrong.
"""

EPOCH = date(2023, 1, 1)
SPAN_DAYS = 900
URI_SCHEME = "synth"


@dataclass(frozen=True, slots=True)
class SynthDoc:
    """One authored document.

    ``uri`` carries the date, because a real column is the only place a date
    supports a range query -- and the date segment comes **first**, so
    lexicographic order is date order across every database. Putting the
    database segment first would make a global range impossible.
    """

    db: str
    uri: str
    title: str
    topic: str
    text: str
    active: bool
    tags: tuple[str, ...]
    owner: str
    level: int
    published: date

    @property
    def metadata(self) -> dict[str, object]:
        """What gets written to the JSON ``metadata`` column.

        ``published_meta`` duplicates ``published`` deliberately: the uri copy
        supports a range query and this one cannot, which is the controlled pair
        that isolates the unsupported ``substr``.
        """
        return {
            "active": self.active,
            "tags": list(self.tags),
            "owner": self.owner,
            "level": self.level,
            "published_meta": self.published.isoformat(),
        }


def build_corpus(seed: int = SEED, docs_per_db: int = DOCS_PER_DB) -> list[SynthDoc]:
    """Every document, in a stable order. A function of the seed alone."""
    rng = random.Random(seed)
    topics = sorted(TOPICS)
    docs: list[SynthDoc] = []
    for db in DATABASES:
        for i in range(docs_per_db):
            topic = rng.choice(topics)
            tags = tuple(sorted(rng.sample(TAGS, rng.randint(1, 3))))
            owner = COLLIDER_OWNER if i % 20 == 0 else rng.choice(OWNERS)
            published = EPOCH + timedelta(days=rng.randrange(SPAN_DAYS))
            level = rng.randint(1, 5)
            docs.append(
                SynthDoc(
                    db=db,
                    uri=f"{URI_SCHEME}://{published.isoformat()}/{db}/doc-{i:04d}",
                    title=f"{db} {topic} record {i:04d}",
                    topic=topic,
                    text=(
                        f"{db} {topic} record {i:04d}. {TOPICS[topic]}. "
                        f"Owner {owner}, level {level}."
                    ),
                    active=rng.random() < 0.5,
                    tags=tags,
                    owner=owner,
                    level=level,
                    published=published,
                )
            )
    return docs


def manifest_hash(docs: Iterable[SynthDoc]) -> str:
    """SHA-256 over the corpus content, order-independent.

    Recorded alongside every result. A number whose manifest does not match the
    corpus in the tree is not evidence.
    """
    digest = hashlib.sha256()
    for line in sorted(
        json.dumps(asdict(doc), sort_keys=True, default=str) for doc in docs
    ):
        digest.update(line.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def to_truth(docs: Sequence[SynthDoc]) -> list[dict[str, object]]:
    """The committed oracle, as JSON-ready rows."""
    return [asdict(doc) | {"published": doc.published.isoformat()} for doc in docs]


Predicate = Callable[[SynthDoc], bool]


def expected(docs: Iterable[SynthDoc], predicate: Predicate) -> set[str]:
    """The uris a correct filter must return. Computed, never eyeballed."""
    return {doc.uri for doc in docs if predicate(doc)}


def in_databases(*names: str) -> Predicate:
    """Restrict the oracle to a source selection, mirroring ``sources=``."""
    chosen = frozenset(names)
    return lambda doc: doc.db in chosen


def naive_tag_match(value: str) -> Predicate:
    """What an unanchored ``LIKE '%value%'`` over ``metadata`` actually matches.

    Not what a caller wants -- it is the collision, expressed as an oracle so the
    trap has an expected set of its own rather than being asserted by hand.
    """
    return lambda doc: value in json.dumps(doc.metadata)
