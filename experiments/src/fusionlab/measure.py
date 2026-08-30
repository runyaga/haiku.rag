"""Timing observations and their summary. Pure: no clock, no IO.

The caller supplies the timings; this module only reduces them. A reducer is a
hypothesis, so ``Series`` keeps every raw observation and ``summary`` is derived
on demand -- storing only the scalar would make a later question unanswerable
without re-running.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from statistics import median


class EmptySeriesError(ValueError):
    """A summary was asked of a series with no observations."""


@dataclass(frozen=True, slots=True)
class Point:
    """One arm of a measurement."""

    label: str
    matched: int
    """Documents the filter admitted -- the independent variable."""
    seconds: tuple[float, ...]
    predicate_chars: int = 0
    """Length of the SQL predicate, which grows with an ``IN (...)`` list."""

    def __post_init__(self) -> None:
        if not self.seconds:
            raise EmptySeriesError(f"{self.label}: no observations")

    @property
    def median(self) -> float:
        return median(self.seconds)

    @property
    def best(self) -> float:
        """The floor. Less noisy than the mean under a shared machine."""
        return min(self.seconds)

    @property
    def spread(self) -> float:
        return max(self.seconds) - min(self.seconds)


@dataclass
class Series:
    """Points from one experiment, plus the manifest that produced them."""

    name: str
    manifest: str
    points: list[Point] = field(default_factory=list)

    def add(self, point: Point) -> None:
        self.points.append(point)

    def as_rows(self) -> list[dict[str, object]]:
        """Raw observations, one row per point. Every timing survives."""
        return [
            {
                "label": p.label,
                "matched": p.matched,
                "predicate_chars": p.predicate_chars,
                "seconds": list(p.seconds),
                "median": p.median,
                "best": p.best,
            }
            for p in self.points
        ]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "manifest": self.manifest, "points": self.as_rows()}


def slope(points: Sequence[Point]) -> float:
    """Least-squares slope of best-time against documents matched, in s/doc.

    Uses ``best`` rather than ``median``: the floor is what the work costs, and
    the tail is whatever else the machine was doing.
    """
    usable = [p for p in points if p.matched > 0]
    if len(usable) < 2:
        raise EmptySeriesError("a slope needs at least two points with matches")
    xs = [float(p.matched) for p in usable]
    ys = [p.best for p in usable]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator == 0:
        raise EmptySeriesError("every point matched the same number of documents")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denominator


def monotonic_in(points: Iterable[Point], tolerance: float = 0.25) -> bool:
    """Does cost broadly rise with the number of documents matched?

    Deliberately loose: a strict check on a shared machine measures the machine.
    ``tolerance`` is the fraction of adjacent pairs allowed to invert.
    """
    ordered = sorted((p for p in points if p.matched > 0), key=lambda p: p.matched)
    if len(ordered) < 2:
        raise EmptySeriesError("monotonicity needs at least two points")
    pairs = list(pairwise(ordered))
    inversions = sum(1 for a, b in pairs if b.best < a.best)
    return inversions / len(pairs) <= tolerance
