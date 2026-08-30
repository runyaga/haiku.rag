"""Federated search with per-source filters and a selectable fusion mode.

haiku.rag 0.79.0 offers neither: ``search_sources`` takes one filter for every
database, and the fusion mode is hardcoded (``_RRF_K = 60`` in
``client/search.py``, bypassed entirely when a reranker is configured). Both are
supplied here by fanning out over the public API instead of forking.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

RRF_K = 60
"""Reciprocal-rank smoothing constant, the value the literature uses."""

Mode = Literal["rrf", "rl"]
SearchType = Literal["fts", "vector", "hybrid"]


@dataclass(frozen=True, slots=True)
class Hit:
    """One fused result, keeping everything fusion would otherwise discard.

    ``score`` is the fused value and is **not** a relevance signal. RRF's top hit
    is always ``1/(RRF_K+1)`` whatever the query -- that one is unconditional.
    RL's top is ``1.0`` only in the default case: it is ``0.0`` for a flat span,
    the raw score under ``normalize=False``, and ``weight x 1.0`` when weighted.
    ``raw`` is the database's own score and is the only field that separates a
    real match from nonsense, so a relevance floor is applied to ``raw``.
    """

    source: str
    chunk_id: str | None
    content: str
    score: float
    raw: float
    rank: int


class _Repository(Protocol):
    async def search(
        self,
        query: str,
        limit: int,
        search_type: str,
        filter: str | None,  # the library's parameter name; shadows a builtin
        query_vector: list[float] | None,
    ) -> list[tuple[Any, float]]: ...


class _Client(Protocol):
    @property
    def chunk_repository(self) -> _Repository: ...
    @property
    def embedder(self) -> Any: ...


def fuse_rrf(
    per_source: Mapping[str, Sequence[tuple[Any, float]]], k: int = RRF_K
) -> list[Hit]:
    """Rank-based fusion. Discards the raw score entirely.

    Classic RRF earns its keep by *summing* a document's reciprocal ranks across
    rankers. Over disjoint corpora nothing appears twice, so nothing is ever
    summed and this degenerates to round-robin interleaving in source order --
    measured, and the reason a fused score carries no quality information.
    """
    hits = [
        Hit(source, chunk.id, chunk.content, 1.0 / (k + rank + 1), raw, rank)
        for source, candidates in per_source.items()
        for rank, (chunk, raw) in enumerate(candidates)
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def fuse_rl(
    per_source: Mapping[str, Sequence[tuple[Any, float]]],
    weights: Mapping[str, float] | None = None,
    *,
    normalize: bool = True,
) -> list[Hit]:
    """Relative-score fusion, keeping the magnitude RRF throws away.

    ``normalize=True`` min-max scales each source onto [0,1], which makes scores
    from incommensurable indexes comparable -- necessary for ``fts``, whose IDF
    is per-index. It is also *harmful* for ``vector``: every database applies the
    same ``1/(distance+1)`` transform and shares one embedder, so those scores
    are already comparable and normalising them only flattens the difference.
    Pass ``normalize=False`` there.
    """
    hits: list[Hit] = []
    for source, candidates in per_source.items():
        if not candidates:
            continue
        weight = (weights or {}).get(source, 1.0)
        raws = [raw for _, raw in candidates]
        low, high = min(raws), max(raws)
        span = high - low
        for rank, (chunk, raw) in enumerate(candidates):
            if not normalize:
                value = raw
            elif span == 0:
                # A flat span carries no ordering. Anchoring at 1.0 would promote
                # a uniformly-bad source to the top of the fused list.
                value = 0.0
            else:
                value = (raw - low) / span
            hits.append(Hit(source, chunk.id, chunk.content, value * weight, raw, rank))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits


def resolve_fetch(mode: Mode, limit: int, multiplier: int | None) -> int:
    """Candidates to pull per source.

    Measured: RRF returns byte-identical output at multipliers 1, 3, 5 and 10 --
    a rank-only fusion cannot promote anything that ranked worse than ``limit``
    in its own source, so depth is wasted. RL differs at every multiplier,
    because depth anchors the normalisation floor nearer the true background.
    """
    if multiplier is None:
        multiplier = 1 if mode == "rrf" else 5
    if multiplier < 1:
        raise ValueError(f"fetch multiplier must be >= 1, got {multiplier}")
    return limit * multiplier


def resolve_filters(
    filters: Mapping[str, str] | str | None, sources: Sequence[str]
) -> dict[str, str | None]:
    """One filter for every source, or a different filter per source.

    A source missing from a mapping is searched unfiltered; map it to ``None`` to
    say so explicitly.
    """
    if filters is None or isinstance(filters, str):
        return dict.fromkeys(sources, filters)
    return {source: filters.get(source) for source in sources}


async def federated_search(
    client: Any,
    query: str,
    *,
    sources: Sequence[str],
    filters: Mapping[str, str] | str | None = None,
    mode: Mode = "rrf",
    search_type: SearchType = "hybrid",
    limit: int = 6,
    fetch_multiplier: int | None = None,
    weights: Mapping[str, float] | None = None,
    normalize: bool = True,
    floor: float | None = None,
) -> list[Hit]:
    """Search several databases, each with its own filter, and fuse the results.

    ``floor`` is applied to the *raw* score before fusion. After fusion the
    magnitude is gone, so a floor applied there could not work.
    """
    if not sources:
        # The library returns [] for an empty selection (client/search.py); match
        # it rather than raising from clients[0] below.
        return []

    clients = await client.clients_for(list(sources))
    per_filter = resolve_filters(filters, sources)
    vector = None
    if search_type != "fts":
        vector = await clients[0].embedder.embed_query(query)

    fetch = resolve_fetch(mode, limit, fetch_multiplier)
    per_source: dict[str, list[tuple[Any, float]]] = {}
    for source, owner in zip(sources, clients, strict=True):
        candidates = await owner.chunk_repository.search(
            query=query,
            limit=fetch,
            search_type=search_type,
            filter=per_filter[source],
            query_vector=vector,
        )
        if floor is not None:
            candidates = [(c, raw) for c, raw in candidates if raw >= floor]
        per_source[source] = candidates

    fused = (
        fuse_rrf(per_source)
        if mode == "rrf"
        else fuse_rl(per_source, weights, normalize=normalize)
    )
    return fused[:limit]
