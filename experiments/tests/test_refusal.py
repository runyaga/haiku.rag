"""The refusal metrics. Pure, so fully covered offline."""

import pytest

from fusionlab.refusal import (
    NoOutcomesError,
    Outcome,
    SweepPoint,
    accuracy,
    best_threshold,
    margin,
    refusal_precision,
    refusal_recall,
)

ANSWERED_OK = Outcome("a", answerable=True, refused=False)
ANSWERED_BAD = Outcome("b", answerable=False, refused=False)
REFUSED_OK = Outcome("c", answerable=False, refused=True)
REFUSED_BAD = Outcome("d", answerable=True, refused=True)


class TestOutcome:
    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            (ANSWERED_OK, True),
            (REFUSED_OK, True),
            (ANSWERED_BAD, False),
            (REFUSED_BAD, False),
        ],
    )
    def test_correctness(self, outcome, expected):
        assert outcome.correct is expected


class TestRates:
    def test_recall_counts_only_unanswerable_cases(self):
        assert refusal_recall([REFUSED_OK, ANSWERED_BAD, ANSWERED_OK]) == 0.5

    def test_perfect_recall(self):
        assert refusal_recall([REFUSED_OK, REFUSED_OK]) == 1.0

    def test_precision_counts_only_refusals(self):
        assert refusal_precision([REFUSED_OK, REFUSED_BAD, ANSWERED_OK]) == 0.5

    def test_accuracy_counts_everything(self):
        assert accuracy([ANSWERED_OK, REFUSED_OK, ANSWERED_BAD, REFUSED_BAD]) == 0.5

    def test_recall_with_no_unanswerable_cases_is_undefined(self):
        with pytest.raises(NoOutcomesError, match="unanswerable"):
            refusal_recall([ANSWERED_OK])

    def test_precision_with_no_refusals_is_undefined(self):
        with pytest.raises(NoOutcomesError, match="refusals"):
            refusal_precision([ANSWERED_OK])

    def test_accuracy_of_nothing_is_undefined(self):
        with pytest.raises(NoOutcomesError, match="outcomes"):
            accuracy([])


class TestMargin:
    def test_a_flat_pack_has_no_separation(self):
        assert margin([0.5, 0.5, 0.5]) == 1.0

    def test_a_standout_top_raises_the_margin(self):
        assert margin([10.0, 1.0, 1.0]) == 10.0

    def test_it_is_scale_free(self):
        """The point of a ratio: the two scales differ, and by how much depends
        on the corpus -- fts spans 0.057-0.074 on the synthetic corpus and
        12-16 on airpubs, because IDF depends on the collection."""
        assert margin([16.0, 8.0, 8.0]) == margin([0.5, 0.25, 0.25])

    def test_a_single_score_has_no_pack_to_stand_above(self):
        assert margin([0.7]) == 1.0

    @pytest.mark.parametrize(
        "scores", [[1.0, 0.0, 0.0], [1.0, -1.0, -2.0]], ids=["zero", "negative"]
    )
    def test_a_non_positive_median_reports_no_separation_rather_than_dividing(
        self, scores
    ):
        """Both take the `mid <= 0` branch. Kept separate from the flat-pack
        test above, which takes the OTHER path (median > 0, returns max/median)
        and only coincidentally also returns 1.0."""
        assert margin(scores) == 1.0

    def test_no_scores_is_an_error(self):
        with pytest.raises(NoOutcomesError, match="at least one"):
            margin([])


class TestSweep:
    def test_a_point_exposes_its_rates(self):
        point = SweepPoint(0.5, (REFUSED_OK, ANSWERED_OK))
        assert (point.recall, point.precision) == (1.0, 1.0)

    def test_a_point_that_refuses_nothing_scores_zero_precision(self):
        """Undefined precision must not win: scoring it as perfect would pick
        the floor that answers everything, which is the failure being hunted."""
        assert SweepPoint(0.0, (ANSWERED_BAD, ANSWERED_OK)).precision_or_zero == 0.0

    def test_precision_or_zero_passes_a_real_value_through(self):
        assert SweepPoint(0.5, (REFUSED_OK, ANSWERED_OK)).precision_or_zero == 1.0

    def test_the_best_point_maximises_recall_first(self):
        weak = SweepPoint(0.1, (ANSWERED_BAD, ANSWERED_OK))
        strong = SweepPoint(0.9, (REFUSED_OK, REFUSED_BAD))
        assert best_threshold([weak, strong]) is strong

    def test_precision_breaks_a_recall_tie(self):
        loose = SweepPoint(0.2, (REFUSED_OK, REFUSED_BAD))
        tight = SweepPoint(0.4, (REFUSED_OK, ANSWERED_OK))
        assert best_threshold([loose, tight]) is tight

    def test_the_lowest_threshold_breaks_a_full_tie(self):
        """Prefer the least aggressive floor that achieves the same scores."""
        low = SweepPoint(0.2, (REFUSED_OK, ANSWERED_OK))
        high = SweepPoint(0.8, (REFUSED_OK, ANSWERED_OK))
        assert best_threshold([low, high]) is low

    def test_no_points_is_an_error(self):
        with pytest.raises(NoOutcomesError, match="sweep points"):
            best_threshold([])
