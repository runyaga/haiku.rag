"""The federated query surface, declared as data.

`surface.py` enumerates WHAT can be filtered. This enumerates HOW a filtered
query can be spread across databases: the filter's shape, the retrieval path,
the fusion rule, and how many sources are in play.

As in `surface.py`, cells are enumerated rather than hand-picked, and every cell
the enumeration calls degenerate carries a reason. A cell can be:

  * meaningful   -- worth running and worth asserting on
  * degenerate   -- runnable, but the dimensions collapse so it asserts nothing
                    the simpler cell does not already assert
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import product


class FilterShape(Enum):
    NONE = "none"
    SHARED = "shared"
    """One predicate applied to every source -- all the library can express."""
    PER_SOURCE = "per_source"
    """A different predicate per source -- what this layer adds."""
    PARTIAL = "partial"
    """Some sources filtered, others deliberately left open."""


class SearchType(Enum):
    FTS = "fts"
    VECTOR = "vector"
    HYBRID = "hybrid"


class Fusion(Enum):
    RRF = "rrf"
    RL_NORMALIZED = "rl_norm"
    RL_RAW = "rl_raw"
    """Relative fusion on the raw score, no normalisation."""


SCOPES = (1, 2, 3)


@dataclass(frozen=True, slots=True)
class FedCell:
    shape: FilterShape
    search: SearchType
    fusion: Fusion
    sources: int

    @property
    def id(self) -> str:
        return (
            f"{self.shape.value}-{self.search.value}-"
            f"{self.fusion.value}-n{self.sources}"
        )


def all_cells() -> tuple[FedCell, ...]:
    return tuple(
        FedCell(shape, search, fusion, n)
        for shape, search, fusion, n in product(FilterShape, SearchType, Fusion, SCOPES)
    )


def degenerate_reason(cell: FedCell) -> str | None:
    """Why this cell asserts nothing a simpler cell does not, or None.

    Degenerate is not the same as invalid: every cell here RUNS. The question is
    whether running it tells you anything.
    """
    if cell.sources == 1 and cell.shape in (
        FilterShape.PER_SOURCE,
        FilterShape.PARTIAL,
    ):
        return (
            "a per-source or partial filter over one source is just a shared "
            "filter; the shape dimension collapses"
        )
    if cell.sources == 1 and cell.fusion is not Fusion.RRF:
        return (
            "with one source every fusion rule preserves that source's own "
            "ordering, so the modes are indistinguishable; the RRF cell covers it"
        )
    if cell.search is SearchType.HYBRID and cell.fusion in (
        Fusion.RL_NORMALIZED,
        Fusion.RL_RAW,
    ):
        return (
            "under hybrid the per-source scores are ALREADY LanceDB RRF output, "
            "so relative fusion would be normalising ranks that were laundered "
            "into scores -- measured, and the reason RL is defined for fts and "
            "vector only"
        )
    return None


def meaningful_cells() -> tuple[FedCell, ...]:
    return tuple(c for c in all_cells() if degenerate_reason(c) is None)


def to_kwargs(
    cell: FedCell,
    names: Sequence[str],
    predicate: str,
) -> Mapping[str, object]:
    """The `federated_search` arguments for one cell.

    `names` is the full set of database names; the cell's `sources` count picks
    a prefix, so the same cell always selects the same databases.
    """
    chosen = list(names[: cell.sources])
    if cell.shape is FilterShape.NONE:
        filters: object = None
    elif cell.shape is FilterShape.SHARED:
        filters = predicate
    elif cell.shape is FilterShape.PER_SOURCE:
        filters = {name: predicate for name in chosen}
    else:  # PARTIAL -- first source filtered, the rest deliberately open
        filters = {chosen[0]: predicate}
    return {
        "sources": chosen,
        "filters": filters,
        "search_type": cell.search.value,
        "mode": "rrf" if cell.fusion is Fusion.RRF else "rl",
        "normalize": cell.fusion is not Fusion.RL_RAW,
    }


def report() -> dict[str, object]:
    total = all_cells()
    live = meaningful_cells()
    return {
        "dimensions": {
            "filter_shape": len(FilterShape),
            "search_type": len(SearchType),
            "fusion": len(Fusion),
            "scope": len(SCOPES),
        },
        "cells_total": len(total),
        "cells_meaningful": len(live),
        "cells_degenerate": len(total) - len(live),
        "degenerate": {
            c.id: degenerate_reason(c)
            for c in total
            if degenerate_reason(c) is not None
        },
    }
