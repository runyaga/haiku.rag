"""Scoring a retrieval system's ability to say "nothing here".

Pure. The caller supplies outcomes; this module only reduces them.

Fusion destroys the evidence needed for this. RRF's top score is always
1/(K+1) whatever the query. RL's is 1.0 in the default case, and 0.0, the raw
score, or a weighted value in the others -- either way the fused number is a
property of the fusion rule, not of how good the answer was. So a floor is
applied to the RAW score BEFORE fusion, and this module scores how well it
separates answerable questions from unanswerable ones.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median


class NoOutcomesError(ValueError):
    """A rate was asked of an empty set of outcomes."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """One question, and what the system did with it."""

    label: str
    answerable: bool
    """Whether a correct answer exists inside the filtered subset."""
    refused: bool
    """Whether the system returned nothing."""

    @property
    def correct(self) -> bool:
        return self.refused != self.answerable


def _rate(numerator: int, denominator: int, what: str) -> float:
    if denominator == 0:
        raise NoOutcomesError(f"no {what} to compute a rate over")
    return numerator / denominator


def refusal_recall(outcomes: Sequence[Outcome]) -> float:
    """Of the questions with no answer, how many were refused.

    Low recall is the dangerous direction: the system answered confidently when
    nothing relevant was available.
    """
    unanswerable = [o for o in outcomes if not o.answerable]
    return _rate(
        sum(1 for o in unanswerable if o.refused),
        len(unanswerable),
        "unanswerable cases",
    )


def refusal_precision(outcomes: Sequence[Outcome]) -> float:
    """Of the refusals, how many were right.

    Low precision means the floor is discarding good answers.
    """
    refusals = [o for o in outcomes if o.refused]
    return _rate(
        sum(1 for o in refusals if not o.answerable), len(refusals), "refusals"
    )


def accuracy(outcomes: Sequence[Outcome]) -> float:
    return _rate(sum(1 for o in outcomes if o.correct), len(outcomes), "outcomes")


def margin(raws: Sequence[float]) -> float:
    """How far the best candidate stands above the pack, scale-free.

    An absolute floor cannot transfer between corpora or between search types.
    MEASURED on the synthetic corpus: fts scores span 0.057-0.074 and cosine
    0.35-0.62 -- fts is the SMALLER scale here, though on the airpubs corpus fts
    ran 12-16. The scale is a property of the corpus (IDF), not of the search
    type, which is exactly why an absolute floor does not travel. A ratio can.
    Returns 1.0 when nothing
    stands out and grows as the top pulls away.

    Zero or negative medians fall back to 1.0 rather than dividing: a ratio is
    meaningless there, and reporting "no separation" is the honest answer.
    """
    if not raws:
        raise NoOutcomesError("margin needs at least one score")
    mid = median(raws)
    if mid <= 0:
        return 1.0
    return max(raws) / mid


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One threshold and how it scored."""

    threshold: float
    outcomes: tuple[Outcome, ...]

    @property
    def recall(self) -> float:
        return refusal_recall(self.outcomes)

    @property
    def precision(self) -> float:
        return refusal_precision(self.outcomes)

    @property
    def precision_or_zero(self) -> float:
        """Precision, or 0.0 when the threshold refused nothing.

        Undefined precision must not win a comparison. A threshold that never
        refuses has no precision to speak of, and scoring it as perfect would
        select exactly the floor that answers everything -- the failure this
        suite exists to catch.
        """
        try:
            return self.precision
        except NoOutcomesError:
            return 0.0


def best_threshold(points: Iterable[SweepPoint]) -> SweepPoint:
    """The threshold refusing every unanswerable question while keeping the most
    answerable ones.

    Ordered that way deliberately: answering when nothing is available is the
    failure that matters, so recall is the primary key and precision breaks ties.
    """
    ordered = list(points)
    if not ordered:
        raise NoOutcomesError("no sweep points")
    return max(ordered, key=lambda p: (p.recall, p.precision_or_zero, -p.threshold))
