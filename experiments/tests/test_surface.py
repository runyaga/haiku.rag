"""The surface enumeration itself. Pure, so it runs in the default gate.

These are the tests that make "fully covered" checkable: if the enumeration
grows an attribute or an operation and no builder follows, `uncovered()` is
non-empty and this suite fails before any database is touched.
"""

import pytest

from fusionlab import corpus as c
from fusionlab.surface import (
    Attribute,
    Cell,
    Kind,
    Op,
    Storage,
    all_cells,
    build,
    orphaned,
    report,
    supported_cells,
    uncovered,
    unsupported_reason,
)


class TestCompleteness:
    def test_every_supported_cell_has_a_builder(self):
        """The claim the whole surface exists to make."""
        assert uncovered() == (), [cell.id for cell in uncovered()]

    def test_no_builder_exists_for_an_unsupported_cell(self):
        """Catches the support rules and the builders drifting apart."""
        assert orphaned() == (), orphaned()

    def test_the_surface_is_not_trivially_small(self):
        """A guard against the enumeration silently collapsing."""
        assert len(supported_cells()) >= 40


class TestSupportRules:
    def test_a_metadata_value_cannot_be_ranged(self):
        """`substr` is unsupported, so it cannot be extracted to compare."""
        cell = Cell(Attribute("level", Kind.INTEGER, Storage.METADATA), Op.RANGE)
        assert "substr" in unsupported_reason(cell)

    def test_a_real_column_date_can_be_ranged(self):
        cell = Cell(Attribute("published", Kind.DATE, Storage.COLUMN), Op.RANGE)
        assert unsupported_reason(cell) is None

    def test_an_unordered_kind_cannot_be_ranged_even_in_a_column(self):
        cell = Cell(Attribute("x", Kind.ENUM, Storage.COLUMN), Op.RANGE)
        assert "ordered" in unsupported_reason(cell)

    def test_a_boolean_has_no_prefix(self):
        cell = Cell(Attribute("active", Kind.BOOLEAN, Storage.METADATA), Op.PREFIX)
        assert "prefix" in unsupported_reason(cell)

    def test_a_boolean_has_no_substring(self):
        cell = Cell(Attribute("active", Kind.BOOLEAN, Storage.METADATA), Op.CONTAINS)
        assert "meaningless" in unsupported_reason(cell)

    @pytest.mark.parametrize("kind", [Kind.BOOLEAN, Kind.MULTISELECT])
    def test_set_membership_is_degenerate_for_some_kinds(self, kind):
        cell = Cell(Attribute("x", kind, Storage.METADATA), Op.IN_SET)
        assert unsupported_reason(cell) is not None

    def test_every_unsupported_cell_states_a_reason(self):
        for cell in all_cells():
            reason = unsupported_reason(cell)
            assert reason is None or len(reason) > 20, cell.id


class TestBuild:
    @pytest.fixture(scope="class")
    def docs(self) -> list[c.SynthDoc]:
        return [d for d in c.build_corpus(docs_per_db=60) if d.db == "alpha_db"]

    @pytest.mark.parametrize("cell", supported_cells(), ids=lambda c: c.id)
    def test_each_cell_yields_a_predicate_and_a_discriminating_oracle(self, cell, docs):
        predicate, oracle = build(cell, docs)
        assert predicate.strip(), f"{cell.id}: empty predicate"
        matched = [d for d in docs if oracle(d)]
        assert matched, f"{cell.id}: oracle matches nothing -- the cell is vacuous"
        assert len(matched) < len(docs), (
            f"{cell.id}: oracle matches every document -- indistinguishable from "
            "a filter being ignored"
        )

    def test_building_an_unsupported_cell_is_refused(self, docs):
        cell = Cell(Attribute("level", Kind.INTEGER, Storage.METADATA), Op.RANGE)
        with pytest.raises(ValueError, match="not supported"):
            build(cell, docs)


class TestReport:
    def test_it_names_nothing_uncovered_or_orphaned(self):
        r = report()
        assert r["uncovered"] == []
        assert r["orphaned"] == []

    def test_every_unsupported_cell_appears_with_its_reason(self):
        r = report()
        assert len(r["unsupported"]) == r["cells_unsupported"]
        assert all(isinstance(v, str) and v for v in r["unsupported"].values())


def test_cell_ids_are_unique():
    """Test ids are generated from these; a collision would silently drop a cell."""
    ids = [cell.id for cell in all_cells()]
    assert len(set(ids)) == len(ids)
