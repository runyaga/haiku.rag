"""The federated enumeration. Pure, runs in the default gate."""

import pytest

from fusionlab.federation import (
    FedCell,
    FilterShape,
    Fusion,
    SearchType,
    all_cells,
    degenerate_reason,
    meaningful_cells,
    report,
    to_kwargs,
)

NAMES = ("alpha_db", "beta_db", "gamma_db")
PRED = "metadata LIKE '%x%'"


class TestEnumeration:
    def test_every_degenerate_cell_states_why(self):
        for cell in all_cells():
            reason = degenerate_reason(cell)
            assert reason is None or len(reason) > 25, cell.id

    def test_cell_ids_are_unique(self):
        ids = [c.id for c in all_cells()]
        assert len(set(ids)) == len(ids)

    def test_the_surface_is_not_trivially_small(self):
        assert len(meaningful_cells()) >= 50


class TestDegeneracyRules:
    def test_a_per_source_filter_over_one_source_is_just_a_shared_one(self):
        cell = FedCell(FilterShape.PER_SOURCE, SearchType.FTS, Fusion.RRF, 1)
        assert "collapses" in degenerate_reason(cell)

    def test_fusion_modes_are_indistinguishable_over_one_source(self):
        cell = FedCell(FilterShape.NONE, SearchType.FTS, Fusion.RL_NORMALIZED, 1)
        assert "indistinguishable" in degenerate_reason(cell)

    def test_rrf_over_one_source_is_kept_as_the_representative(self):
        cell = FedCell(FilterShape.NONE, SearchType.FTS, Fusion.RRF, 1)
        assert degenerate_reason(cell) is None

    @pytest.mark.parametrize("fusion", [Fusion.RL_NORMALIZED, Fusion.RL_RAW])
    def test_relative_fusion_over_hybrid_is_normalising_laundered_ranks(self, fusion):
        cell = FedCell(FilterShape.SHARED, SearchType.HYBRID, fusion, 2)
        assert "laundered" in degenerate_reason(cell)

    def test_rrf_over_hybrid_is_fine(self):
        cell = FedCell(FilterShape.SHARED, SearchType.HYBRID, Fusion.RRF, 2)
        assert degenerate_reason(cell) is None


class TestToKwargs:
    def test_the_scope_picks_a_stable_prefix_of_the_names(self):
        cell = FedCell(FilterShape.NONE, SearchType.FTS, Fusion.RRF, 2)
        assert to_kwargs(cell, NAMES, PRED)["sources"] == ["alpha_db", "beta_db"]

    def test_shape_none_passes_no_filter(self):
        cell = FedCell(FilterShape.NONE, SearchType.FTS, Fusion.RRF, 2)
        assert to_kwargs(cell, NAMES, PRED)["filters"] is None

    def test_shape_shared_passes_one_string_for_every_source(self):
        cell = FedCell(FilterShape.SHARED, SearchType.FTS, Fusion.RRF, 2)
        assert to_kwargs(cell, NAMES, PRED)["filters"] == PRED

    def test_shape_per_source_passes_a_mapping_covering_every_source(self):
        cell = FedCell(FilterShape.PER_SOURCE, SearchType.FTS, Fusion.RRF, 3)
        filters = to_kwargs(cell, NAMES, PRED)["filters"]
        assert set(filters) == set(NAMES)

    def test_shape_partial_filters_only_the_first_source(self):
        cell = FedCell(FilterShape.PARTIAL, SearchType.FTS, Fusion.RRF, 3)
        filters = to_kwargs(cell, NAMES, PRED)["filters"]
        assert set(filters) == {"alpha_db"}

    @pytest.mark.parametrize(
        ("fusion", "mode", "normalize"),
        [
            (Fusion.RRF, "rrf", True),
            (Fusion.RL_NORMALIZED, "rl", True),
            (Fusion.RL_RAW, "rl", False),
        ],
    )
    def test_the_fusion_dimension_maps_onto_mode_and_normalize(
        self, fusion, mode, normalize
    ):
        cell = FedCell(FilterShape.NONE, SearchType.FTS, fusion, 2)
        kwargs = to_kwargs(cell, NAMES, PRED)
        assert (kwargs["mode"], kwargs["normalize"]) == (mode, normalize)

    def test_the_search_type_passes_through(self):
        cell = FedCell(FilterShape.NONE, SearchType.VECTOR, Fusion.RRF, 2)
        assert to_kwargs(cell, NAMES, PRED)["search_type"] == "vector"


class TestReport:
    def test_every_degenerate_cell_appears_with_its_reason(self):
        r = report()
        assert len(r["degenerate"]) == r["cells_degenerate"]
        assert all(isinstance(v, str) and v for v in r["degenerate"].values())
