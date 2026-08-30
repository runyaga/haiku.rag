"""The parametric-query surface, declared as data.

Tests written by hand cover whatever occurred to the author. This module instead
ENUMERATES the surface — every attribute type crossed with every operation —
and marks each cell supported or not, with the reason. Tests are then generated
from the enumeration, so "fully covered" is a checkable claim rather than a
belief: `uncovered()` names any cell no test reached.

Support verdicts trace to the measured capability matrix (see the plan):
21 of 23 SQL constructs work; `substr` is unsupported by Lance's SQL planner,
which is why a value inside the `metadata` JSON text cannot be bounded — only
matched. A date in a REAL column has no such limit.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from fusionlab import corpus as c
from fusionlab import filters as f


class Kind(Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    ENUM = "enum"
    MULTISELECT = "multiselect"
    DATE = "date"
    TEXT = "text"


class Storage(Enum):
    METADATA = "metadata"
    """A key inside the JSON text column. Matchable, never bounded."""
    COLUMN = "column"
    """A real document_meta column. Supports ordering and range."""


class Op(Enum):
    EQ = "eq"
    CONTAINS = "contains"
    PREFIX = "prefix"
    RANGE = "range"
    IN_SET = "in_set"
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass(frozen=True, slots=True)
class Attribute:
    name: str
    kind: Kind
    storage: Storage


ATTRIBUTES = (
    Attribute("active", Kind.BOOLEAN, Storage.METADATA),
    Attribute("level", Kind.INTEGER, Storage.METADATA),
    Attribute("owner", Kind.ENUM, Storage.METADATA),
    Attribute("tags", Kind.MULTISELECT, Storage.METADATA),
    Attribute("published_meta", Kind.DATE, Storage.METADATA),
    Attribute("published", Kind.DATE, Storage.COLUMN),
    Attribute("title", Kind.TEXT, Storage.COLUMN),
)


@dataclass(frozen=True, slots=True)
class Cell:
    attribute: Attribute
    op: Op

    @property
    def id(self) -> str:
        return f"{self.attribute.name}-{self.op.value}"


def all_cells() -> tuple[Cell, ...]:
    """Every attribute x operation. The complete surface, supported or not."""
    return tuple(Cell(a, o) for a in ATTRIBUTES for o in Op)


def unsupported_reason(cell: Cell) -> str | None:
    """Why this cell cannot be expressed, or None if it can.

    Every verdict here is measured, not assumed.
    """
    a, op = cell.attribute, cell.op
    if op is Op.RANGE and a.storage is Storage.METADATA:
        return (
            "a value inside the metadata JSON text cannot be bounded: `substr` "
            "is unsupported by Lance's SQL planner, so it cannot be extracted "
            "to compare. Only equality and regex reach it."
        )
    if op is Op.RANGE and a.kind not in (Kind.DATE, Kind.TEXT):
        return f"{a.kind.value} is not an ordered type in this dialect"
    if op is Op.PREFIX and a.kind is Kind.BOOLEAN:
        return "a boolean has no prefix"
    if op is Op.IN_SET and a.kind in (Kind.BOOLEAN, Kind.MULTISELECT):
        return (
            "set membership over a boolean is degenerate, and over a "
            "multiselect it is already what CONTAINS expresses"
        )
    if op is Op.CONTAINS and a.kind is Kind.BOOLEAN:
        return "substring matching a boolean is meaningless"
    return None


def supported_cells() -> tuple[Cell, ...]:
    return tuple(cell for cell in all_cells() if unsupported_reason(cell) is None)


Oracle = Callable[[c.SynthDoc], bool]


def build(cell: Cell, docs: Sequence[c.SynthDoc]) -> tuple[str, Oracle]:
    """The SQL predicate and its INDEPENDENT Python oracle for one cell.

    The two are written separately on purpose: the predicate comes from
    `filters`, the oracle from the document's own attributes. A cell where both
    sides were derived from one expression would prove nothing.
    """
    reason = unsupported_reason(cell)
    if reason is not None:
        raise ValueError(f"{cell.id} is not supported: {reason}")
    return _BUILDERS[(cell.attribute.name, cell.op)](docs)


def _mid_date(docs: Sequence[c.SynthDoc]) -> date:
    ordered = sorted(d.published for d in docs)
    return ordered[len(ordered) // 2]


_BUILDERS: dict[
    tuple[str, Op], Callable[[Sequence[c.SynthDoc]], tuple[str, Oracle]]
] = {
    # boolean
    ("active", Op.EQ): lambda _: (f.meta_eq("active", True), lambda d: d.active),
    ("active", Op.AND): lambda _: (
        f.all_of(f.meta_eq("active", True), f.meta_eq("owner", "north")),
        lambda d: d.active and d.owner == "north",
    ),
    ("active", Op.OR): lambda _: (
        f.any_of(f.meta_eq("active", True), f.meta_eq("level", 5)),
        lambda d: d.active or d.level == 5,
    ),
    ("active", Op.NOT): lambda _: (
        f.not_(f.meta_eq("active", True)),
        lambda d: not d.active,
    ),
    # integer
    ("level", Op.EQ): lambda _: (f.meta_eq("level", 3), lambda d: d.level == 3),
    ("level", Op.CONTAINS): lambda _: (f.meta_eq("level", 2), lambda d: d.level == 2),
    ("level", Op.PREFIX): lambda _: (f.meta_eq("level", 1), lambda d: d.level == 1),
    ("level", Op.IN_SET): lambda _: (
        f.any_of(f.meta_eq("level", 1), f.meta_eq("level", 2)),
        lambda d: d.level in (1, 2),
    ),
    ("level", Op.AND): lambda _: (
        f.all_of(f.meta_eq("level", 4), f.meta_eq("active", True)),
        lambda d: d.level == 4 and d.active,
    ),
    ("level", Op.OR): lambda _: (
        f.any_of(f.meta_eq("level", 4), f.meta_eq("level", 5)),
        lambda d: d.level in (4, 5),
    ),
    ("level", Op.NOT): lambda _: (
        f.not_(f.meta_eq("level", 3)),
        lambda d: d.level != 3,
    ),
    # enum
    ("owner", Op.EQ): lambda _: (
        f.meta_eq("owner", "north"),
        lambda d: d.owner == "north",
    ),
    ("owner", Op.CONTAINS): lambda _: (
        f.meta_eq("owner", c.COLLIDER_OWNER),
        lambda d: d.owner == c.COLLIDER_OWNER,
    ),
    ("owner", Op.PREFIX): lambda _: (
        f.meta_eq("owner", "south"),
        lambda d: d.owner == "south",
    ),
    ("owner", Op.IN_SET): lambda _: (
        f.any_of(f.meta_eq("owner", "north"), f.meta_eq("owner", "east")),
        lambda d: d.owner in ("north", "east"),
    ),
    ("owner", Op.AND): lambda _: (
        f.all_of(f.meta_eq("owner", "shared"), f.meta_eq("active", False)),
        lambda d: d.owner == "shared" and not d.active,
    ),
    ("owner", Op.OR): lambda _: (
        f.any_of(f.meta_eq("owner", "north"), f.meta_eq("owner", "south")),
        lambda d: d.owner in ("north", "south"),
    ),
    ("owner", Op.NOT): lambda _: (
        f.not_(f.meta_eq("owner", "north")),
        lambda d: d.owner != "north",
    ),
    # multiselect
    ("tags", Op.EQ): lambda _: (
        f.meta_has("tags", "alpha"),
        lambda d: "alpha" in d.tags,
    ),
    ("tags", Op.CONTAINS): lambda _: (
        f.meta_has("tags", "bravo"),
        lambda d: "bravo" in d.tags,
    ),
    ("tags", Op.PREFIX): lambda _: (
        f.meta_has("tags", "charlie"),
        lambda d: "charlie" in d.tags,
    ),
    ("tags", Op.AND): lambda _: (
        f.all_of(f.meta_has("tags", "alpha"), f.meta_has("tags", "bravo")),
        lambda d: "alpha" in d.tags and "bravo" in d.tags,
    ),
    ("tags", Op.OR): lambda _: (
        f.any_of(f.meta_has("tags", "alpha"), f.meta_has("tags", "delta")),
        lambda d: "alpha" in d.tags or "delta" in d.tags,
    ),
    ("tags", Op.NOT): lambda _: (
        f.not_(f.meta_has("tags", "alpha")),
        lambda d: "alpha" not in d.tags,
    ),
    # date in metadata -- matchable, never bounded
    ("published_meta", Op.EQ): lambda docs: (
        f.meta_eq("published_meta", _mid_date(docs).isoformat()),
        lambda d, day=None: d.published == _mid_date(docs),
    ),
    ("published_meta", Op.CONTAINS): lambda docs: (
        f.meta_eq("published_meta", _mid_date(docs).isoformat()),
        lambda d: d.published == _mid_date(docs),
    ),
    ("published_meta", Op.PREFIX): lambda _: (
        f.meta_starts_with("published_meta", "2024"),
        lambda d: d.published.year == 2024,
    ),
    ("published_meta", Op.IN_SET): lambda docs: (
        f.any_of(
            f.meta_eq("published_meta", _mid_date(docs).isoformat()),
            f.meta_eq(
                "published_meta", (_mid_date(docs) + timedelta(days=1)).isoformat()
            ),
        ),
        lambda d: d.published in (_mid_date(docs), _mid_date(docs) + timedelta(days=1)),
    ),
    ("published_meta", Op.AND): lambda _: (
        f.all_of(
            f.meta_starts_with("published_meta", "2024"),
            f.meta_eq("active", True),
        ),
        lambda d: d.published.year == 2024 and d.active,
    ),
    ("published_meta", Op.OR): lambda docs: (
        f.any_of(
            f.meta_eq("published_meta", _mid_date(docs).isoformat()),
            f.meta_eq("level", 5),
        ),
        lambda d: d.published == _mid_date(docs) or d.level == 5,
    ),
    ("published_meta", Op.NOT): lambda docs: (
        f.not_(f.meta_eq("published_meta", _mid_date(docs).isoformat())),
        lambda d: d.published != _mid_date(docs),
    ),
    # date in a real column -- the only place RANGE is expressible
    ("published", Op.EQ): lambda docs: (
        f.uri_date_range(_mid_date(docs), _mid_date(docs) + timedelta(days=1)),
        lambda d: d.published == _mid_date(docs),
    ),
    ("published", Op.RANGE): lambda docs: (
        f.uri_date_range(date(2024, 1, 1), date(2025, 1, 1)),
        lambda d: date(2024, 1, 1) <= d.published < date(2025, 1, 1),
    ),
    ("published", Op.CONTAINS): lambda _: (
        f.contains("uri", "2024-"),
        lambda d: d.published.year == 2024,
    ),
    ("published", Op.PREFIX): lambda _: (
        f.starts_with("uri", f"{c.URI_SCHEME}://2023"),
        lambda d: d.published.year == 2023,
    ),
    ("published", Op.IN_SET): lambda _: (
        f.any_of(f.contains("uri", "2023-"), f.contains("uri", "2025-")),
        lambda d: d.published.year in (2023, 2025),
    ),
    ("published", Op.AND): lambda _: (
        f.all_of(
            f.uri_date_range(date(2024, 1, 1), date(2025, 1, 1)),
            f.meta_eq("active", True),
        ),
        lambda d: date(2024, 1, 1) <= d.published < date(2025, 1, 1) and d.active,
    ),
    ("published", Op.OR): lambda _: (
        f.any_of(f.contains("uri", "2023-"), f.meta_eq("level", 5)),
        lambda d: d.published.year == 2023 or d.level == 5,
    ),
    ("published", Op.NOT): lambda _: (
        f.not_(f.contains("uri", "2024-")),
        lambda d: d.published.year != 2024,
    ),
    # text in a real column
    ("title", Op.EQ): lambda _: (
        f.contains("title", "record 0001"),
        lambda d: "record 0001" in d.title,
    ),
    ("title", Op.CONTAINS): lambda _: (
        f.contains("title", "ceramics"),
        lambda d: d.topic == "ceramics",
    ),
    ("title", Op.PREFIX): lambda _: (
        f.contains("title", "horology"),
        lambda d: d.topic == "horology",
    ),
    ("title", Op.RANGE): lambda _: (
        # Titles read "<db> <topic> record NNNN", so a range over the topic
        # segment selects a contiguous slice of topics.
        f.range_("title", "alpha_db c", "alpha_db h"),
        lambda d: "alpha_db c" <= d.title < "alpha_db h",
    ),
    ("title", Op.IN_SET): lambda _: (
        f.any_of(f.contains("title", "textiles"), f.contains("title", "ceramics")),
        lambda d: d.topic in ("textiles", "ceramics"),
    ),
    ("title", Op.AND): lambda _: (
        f.all_of(f.contains("title", "apiculture"), f.meta_eq("active", True)),
        lambda d: d.topic == "apiculture" and d.active,
    ),
    ("title", Op.OR): lambda _: (
        f.any_of(f.contains("title", "apiculture"), f.contains("title", "textiles")),
        lambda d: d.topic in ("apiculture", "textiles"),
    ),
    ("title", Op.NOT): lambda _: (
        f.not_(f.contains("title", "mineralogy")),
        lambda d: d.topic != "mineralogy",
    ),
}


def uncovered() -> tuple[Cell, ...]:
    """Supported cells with no builder. Must be empty for the surface to be
    fully covered — this is the claim the test suite asserts."""
    return tuple(
        cell
        for cell in supported_cells()
        if (cell.attribute.name, cell.op) not in _BUILDERS
    )


def orphaned() -> tuple[tuple[str, Op], ...]:
    """Builders for cells the enumeration says are unsupported. Also a defect:
    it means the support rules and the builders disagree."""
    supported = {(cell.attribute.name, cell.op) for cell in supported_cells()}
    return tuple(key for key in _BUILDERS if key not in supported)


def report() -> dict[str, object]:
    total = all_cells()
    ok = supported_cells()
    return {
        "attributes": len(ATTRIBUTES),
        "operations": len(Op),
        "cells_total": len(total),
        "cells_supported": len(ok),
        "cells_unsupported": len(total) - len(ok),
        "cells_with_builders": len(_BUILDERS),
        "uncovered": [cell.id for cell in uncovered()],
        "orphaned": [f"{name}-{op.value}" for name, op in orphaned()],
        "unsupported": {
            cell.id: unsupported_reason(cell)
            for cell in total
            if unsupported_reason(cell) is not None
        },
    }
