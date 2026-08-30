"""The measurement reducers. Pure, so fully covered offline."""

import pytest

from fusionlab.measure import EmptySeriesError, Point, Series, monotonic_in, slope


class TestPoint:
    def test_it_reduces_its_observations(self):
        p = Point("a", 10, (0.3, 0.1, 0.2))
        assert (p.median, p.best, p.spread) == (0.2, 0.1, pytest.approx(0.2))

    def test_a_single_observation_is_allowed(self):
        assert Point("a", 1, (0.5,)).median == 0.5

    def test_no_observations_is_refused(self):
        with pytest.raises(EmptySeriesError, match="no observations"):
            Point("a", 1, ())


class TestSeries:
    def test_rows_keep_every_raw_timing(self):
        """A reducer is a hypothesis; the raw observations must survive it."""
        s = Series("f2", "abc")
        s.add(Point("wide", 100, (0.4, 0.5)))
        assert s.as_rows()[0]["seconds"] == [0.4, 0.5]

    def test_to_dict_carries_the_manifest(self):
        """A number whose manifest does not match the corpus is not evidence."""
        assert Series("f2", "abc").to_dict()["manifest"] == "abc"

    def test_it_starts_empty(self):
        assert Series("f2", "abc").as_rows() == []


class TestSlope:
    def test_it_is_positive_when_cost_grows_with_size(self):
        points = [Point("a", 10, (0.1,)), Point("b", 100, (1.0,))]
        assert slope(points) == pytest.approx((1.0 - 0.1) / (100 - 10))

    def test_it_is_zero_for_flat_cost(self):
        assert slope([Point("a", 10, (0.5,)), Point("b", 100, (0.5,))]) == 0.0

    def test_points_matching_nothing_are_ignored(self):
        points = [Point("z", 0, (9.9,)), Point("a", 10, (0.1,)), Point("b", 20, (0.2,))]
        assert slope(points) > 0

    def test_one_point_is_not_a_slope(self):
        with pytest.raises(EmptySeriesError, match="at least two"):
            slope([Point("a", 10, (0.1,))])

    def test_identical_sizes_have_no_slope(self):
        with pytest.raises(EmptySeriesError, match="same number"):
            slope([Point("a", 10, (0.1,)), Point("b", 10, (0.2,))])


class TestMonotonic:
    def test_rising_cost_is_monotonic(self):
        assert monotonic_in([Point("a", 10, (0.1,)), Point("b", 100, (1.0,))])

    def test_a_single_inversion_is_tolerated(self):
        points = [
            Point("a", 10, (0.10,)),
            Point("b", 20, (0.09,)),
            Point("c", 30, (0.30,)),
            Point("d", 40, (0.40,)),
            Point("e", 50, (0.50,)),
        ]
        assert monotonic_in(points)

    def test_a_falling_series_is_not_monotonic(self):
        points = [Point("a", 10, (1.0,)), Point("b", 100, (0.1,))]
        assert not monotonic_in(points)

    def test_it_needs_two_points(self):
        with pytest.raises(EmptySeriesError, match="at least two"):
            monotonic_in([Point("a", 10, (0.1,))])
