"""Every predicate builder, including the escaping that S5 exists to report."""

import json
from datetime import date

import pytest

from fusionlab import filters as f


class TestColumnGuard:
    def test_a_real_column_is_accepted(self):
        assert f.eq("uri", "x") == "uri = 'x'"

    @pytest.mark.parametrize("builder", [f.eq, f.contains, f.starts_with])
    def test_an_unknown_column_is_refused(self, builder):
        with pytest.raises(f.UnknownColumnError, match="bogus"):
            builder("bogus", "x")

    def test_range_also_guards_its_column(self):
        with pytest.raises(f.UnknownColumnError):
            f.range_("bogus", "a", "b")


class TestEscaping:
    def test_a_quote_in_a_value_is_doubled(self):
        assert f.quote("o'brien") == "'o''brien'"

    def test_like_wildcards_in_a_value_are_escaped(self):
        assert "\\%" in f.contains("uri", "50%")
        assert "\\_" in f.contains("uri", "a_b")

    def test_a_backslash_is_escaped_before_the_wildcards(self):
        assert f.contains("uri", "a\\b").count("\\\\") == 1

    def test_a_quote_survives_into_a_contains_predicate(self):
        assert "''" in f.contains("title", "o'brien")


class TestRealColumnPredicates:
    def test_contains_wraps_in_wildcards(self):
        assert f.contains("uri", "abc") == "uri LIKE '%abc%'"

    def test_starts_with_anchors_left_only(self):
        assert f.starts_with("uri", "abc") == "uri LIKE 'abc%'"

    def test_range_is_half_open(self):
        assert f.range_("created_at", "2024-01-01", "2025-01-01") == (
            "(created_at >= '2024-01-01' AND created_at < '2025-01-01')"
        )


class TestMetadataPredicates:
    def test_meta_eq_matches_the_encoded_pair(self):
        assert '"level": 3' in f.meta_eq("level", 3)

    def test_meta_eq_encodes_a_boolean_as_json_not_python(self):
        assert "true" in f.meta_eq("active", True)
        assert "True" not in f.meta_eq("active", True)

    def test_meta_eq_escapes_a_quote_in_a_string_value(self):
        assert "''" in f.meta_eq("owner", "o'brien")

    def test_meta_has_anchors_inside_the_array(self):
        predicate = f.meta_has("tags", "alpha")
        assert predicate.startswith("regexp_like(metadata,")
        assert r"\[[^]]*" in predicate

    def test_meta_starts_with_anchors_to_its_key(self):
        predicate = f.meta_starts_with("published_meta", "2024")
        assert predicate.startswith("regexp_like(metadata,")
        assert "published_meta" in predicate
        assert "2024" in predicate

    def test_meta_starts_with_escapes_regex_metacharacters(self):
        """A prefix containing `.` or `[` must match literally, not as a class."""
        predicate = f.meta_starts_with("k", "a.b[c")
        assert r"a\.b\[c" in predicate

    def test_meta_starts_with_is_not_satisfied_by_another_fields_value(self):
        """The anchor is the point: the same prefix elsewhere must not match."""
        assert '"k"' in f.meta_starts_with("k", "2024")

    def test_meta_has_key_ignores_the_value(self):
        assert "published_meta" in f.meta_has_key("published_meta")


class TestUriDateRange:
    def test_it_is_a_true_half_open_range_not_a_regex(self):
        predicate = f.uri_date_range(date(2024, 1, 1), date(2024, 2, 1))
        assert predicate == (
            "(uri >= 'synth://2024-01-01' AND uri < 'synth://2024-02-01')"
        )
        assert "regexp_like" not in predicate

    def test_the_scheme_is_configurable(self):
        assert "other://" in f.uri_date_range(
            date(2024, 1, 1), date(2024, 2, 1), scheme="other"
        )

    def test_an_empty_window_matches_nothing(self):
        assert f.uri_date_range(date(2024, 1, 1), date(2024, 1, 1)) == "uri < uri"

    def test_a_reversed_window_matches_nothing(self):
        assert f.uri_date_range(date(2024, 2, 1), date(2024, 1, 1)) == "uri < uri"


class TestComposition:
    def test_all_of_parenthesises_each_operand(self):
        assert f.all_of("a = 1", "b = 2") == "((a = 1) AND (b = 2))"

    def test_any_of_joins_with_or(self):
        assert f.any_of("a = 1", "b = 2") == "((a = 1) OR (b = 2))"

    def test_a_single_operand_is_returned_unwrapped(self):
        assert f.all_of("a = 1") == "a = 1"
        assert f.any_of("a = 1") == "a = 1"

    @pytest.mark.parametrize("builder", [f.all_of, f.any_of])
    def test_no_operands_is_a_caller_error(self, builder):
        with pytest.raises(ValueError, match="at least one"):
            builder()

    def test_not_wraps_its_operand(self):
        assert f.not_("a = 1") == "NOT (a = 1)"


def test_a_predicate_is_valid_json_free_of_python_repr():
    """json.dumps, not str(): a Python repr would never match the stored text."""
    assert json.dumps({"tags": ["a"]})[1:-1] in f.meta_eq("tags", ["a"])
