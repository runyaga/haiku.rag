"""Typed filter predicates for haiku.rag's ``filter`` parameter.

haiku.rag passes ``filter`` straight to LanceDB's ``.where()`` without escaping,
parsing or validation (``store/repositories/chunk.py``). Suite 5 reports that as
a defect, so this harness must not reproduce it: every value here is escaped at
construction, and nothing accepts pre-built SQL.

``metadata`` is a JSON *text* column, so predicates over it are string matches on
the encoded JSON. ``substr`` is unsupported by the engine, which is why a date
inside ``metadata`` cannot be range-queried and a date in ``uri`` can.
"""

import json
import re
from datetime import date

# Columns that exist on document_meta. A predicate naming anything else is a
# caller bug, not a filter that returns nothing.
COLUMNS = frozenset({"id", "uri", "title", "metadata", "created_at", "updated_at"})


class UnknownColumnError(ValueError):
    """A predicate named a column that document_meta does not have."""


def _column(name: str) -> str:
    if name not in COLUMNS:
        raise UnknownColumnError(
            f"{name!r} is not a document_meta column; expected one of "
            f"{', '.join(sorted(COLUMNS))}"
        )
    return name


def quote(value: str) -> str:
    """A SQL string literal. Doubling is the engine's escape for a quote."""
    return "'" + value.replace("'", "''") + "'"


def _like_literal(value: str) -> str:
    """Escape LIKE wildcards so a value containing % or _ matches literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def eq(column: str, value: str) -> str:
    """``column = 'value'`` on a real column."""
    return f"{_column(column)} = {quote(value)}"


def contains(column: str, value: str) -> str:
    """``column LIKE '%value%'`` on a real column, wildcards in `value` escaped."""
    return f"{_column(column)} LIKE {quote('%' + _like_literal(value) + '%')}"


def starts_with(column: str, prefix: str) -> str:
    """``column LIKE 'prefix%'`` on a real column."""
    return f"{_column(column)} LIKE {quote(_like_literal(prefix) + '%')}"


def range_(column: str, start: str, end: str) -> str:
    """Half-open ``start <= column < end``.

    ISO-8601 sorts lexicographically, so this is a true date range on a real
    text column -- the reason document dates belong in ``uri`` rather than in
    ``metadata``.
    """
    col = _column(column)
    return f"({col} >= {quote(start)} AND {col} < {quote(end)})"


def uri_date_range(start: date, end: date, scheme: str = "synth") -> str:
    """Documents whose uri-encoded date falls in ``[start, end)``.

    A true lexicographic range on a real column -- no regex, no scan-only
    predicate. It works because the corpus writes the date as the **first**
    segment (``synth://2024-03-15/alpha_db/doc-0041``) and ISO-8601 sorts
    lexicographically, so uri order is date order across every database.

    This is the capability the matrix found and ``metadata`` cannot offer:
    ``substr`` is unsupported, so a date inside the JSON blob can be compared
    for equality but never bounded.
    """
    if end <= start:
        # An empty window. `uri < uri` is false for every row and needs no
        # special-casing downstream.
        return "uri < uri"
    return range_(
        "uri", f"{scheme}://{start.isoformat()}", f"{scheme}://{end.isoformat()}"
    )


def _regex(column: str, pattern: str) -> str:
    return f"regexp_like({_column(column)}, {quote(pattern)})"


def meta_eq(key: str, value: str | int | bool) -> str:
    """A scalar ``metadata`` key equals ``value``.

    Matched against the encoded JSON pair, so ``{"level": 3}`` cannot be
    satisfied by ``{"other_level": 3}``.
    """
    encoded = json.dumps({key: value})[1:-1]
    return f"metadata LIKE {quote('%' + _like_literal(encoded) + '%')}"


def meta_has(key: str, value: str) -> str:
    """``value`` appears in the list under ``metadata[key]``.

    Anchored inside that key's array, so a value that also occurs as a substring
    of some other field does not match. A plain ``LIKE '%value%'`` does, which is
    the collision the corpus deliberately plants.
    """
    pattern = re.escape(json.dumps(key)) + r":\s*\[[^]]*" + re.escape(json.dumps(value))
    return _regex("metadata", pattern)


def meta_starts_with(key: str, prefix: str) -> str:
    """The value under ``metadata[key]`` begins with ``prefix``.

    Anchored to the key, so it cannot match a prefix sitting in another field.
    This is the coarsest date selection still available inside the JSON text:
    ``substr`` is unsupported so the value cannot be bounded, but a year or
    year-month prefix can be matched exactly.
    """
    return _regex("metadata", re.escape(json.dumps(key)) + r':\s*"' + re.escape(prefix))


def meta_has_key(key: str) -> str:
    """``metadata`` carries ``key`` at all, whatever its value."""
    return _regex("metadata", re.escape(json.dumps(key)) + r"\s*:")


def all_of(*predicates: str) -> str:
    """Conjunction. Each operand is parenthesised, so precedence cannot bite."""
    return _join("AND", predicates)


def any_of(*predicates: str) -> str:
    """Disjunction."""
    return _join("OR", predicates)


def _join(op: str, predicates: tuple[str, ...]) -> str:
    if not predicates:
        raise ValueError(f"{op.lower()}_of() needs at least one predicate")
    if len(predicates) == 1:
        return predicates[0]
    return "(" + f" {op} ".join(f"({p})" for p in predicates) + ")"


def not_(predicate: str) -> str:
    """Negation."""
    return f"NOT ({predicate})"
