"""Stubs standing in for haiku.rag, so every unit test runs with no network.

The same technique haiku.rag's own suite uses (``tests/multi_db/helpers.py``
StubReranker, and the ``query_embedding`` fixture monkeypatching
``embed_query``).
"""

from dataclasses import dataclass

import pytest


@dataclass
class StubChunk:
    id: str
    content: str


class StubRepository:
    """Records every call, so a test can assert what each database was asked."""

    def __init__(self, results: list[tuple[StubChunk, float]]) -> None:
        self._results = results
        self.calls: list[dict] = []

    async def search(self, **kwargs: object) -> list[tuple[StubChunk, float]]:
        self.calls.append(kwargs)
        return list(self._results)


class StubEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2, 0.3]


class StubOwner:
    def __init__(self, results: list[tuple[StubChunk, float]]) -> None:
        self.chunk_repository = StubRepository(results)
        self.embedder = StubEmbedder()


class StubClient:
    """Supplies the three attributes ``federated_search`` actually touches."""

    def __init__(self, owners: dict[str, StubOwner]) -> None:
        self._owners = owners
        self.requested: list[list[str]] = []

    async def clients_for(self, names: list[str]) -> list[StubOwner]:
        self.requested.append(list(names))
        return [self._owners[name] for name in names]


def chunks(prefix: str, scores: list[float]) -> list[tuple[StubChunk, float]]:
    return [
        (StubChunk(id=f"{prefix}-{i}", content=f"{prefix} body {i}"), score)
        for i, score in enumerate(scores)
    ]


@pytest.fixture
def divergent_shapes() -> StubClient:
    """Sources whose NORMALISED shapes differ, not merely their ranges.

    Under min-max every source's top becomes 1.0, so RL can only diverge from
    RRF in the tail. beta's runner-up sits high in beta's own range (0.975)
    while alpha's sits mid (0.5), which separates the two orderings by 0.475.

    The previous fixture separated them by ONE ULP: alpha normalised to
    [1.0, 0.4999999999999999, 0.0] and beta to [1.0, 0.5, 0.0], so the test
    asserting the two modes order differently passed on floating-point noise
    and would have flipped on any change to a fixture score.
    """
    return StubClient(
        {
            "alpha": StubOwner(chunks("alpha", [0.90, 0.50, 0.10])),
            "beta": StubOwner(chunks("beta", [0.40, 0.39, 0.00])),
        }
    )


@pytest.fixture
def two_sources() -> StubClient:
    """alpha's raw scores are strong, beta's are weak but tightly clustered."""
    return StubClient(
        {
            "alpha": StubOwner(chunks("alpha", [0.90, 0.60, 0.30])),
            "beta": StubOwner(chunks("beta", [0.40, 0.39, 0.38])),
        }
    )
