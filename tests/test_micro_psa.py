"""
Tests for MicroCell PSA addressing system.

Validates display-address generation and monotonic refinement. Semantic-class
candidate generation does not use MicroCell PSA lookups.

Author: GitHub Copilot
Created: January 2026
"""

import numpy as np
import pytest

from consolidated_clustering_v2.routing.psa_mapper import (
    MICRO_CELL_COUNT,
    MICRO_GRID_N,
    MICRO_MAX_DEPTH,
    MicroPSAAddress,
    PSACoordinate,
    PSAMapper,
    VoxelBounds,
)


def assert_hierarchical_addresses_same_or_adjacent(
    first: MicroPSAAddress,
    second: MicroPSAAddress,
) -> None:
    first_voxel = tuple(int(part) for part in first.coarse_voxel.split("_")[2:5])
    second_voxel = tuple(int(part) for part in second.coarse_voxel.split("_")[2:5])
    assert all(abs(a - b) <= 1 for a, b in zip(first_voxel, second_voxel, strict=False))

    if first.coarse_voxel != second.coarse_voxel:
        return

    for first_cell, second_cell in zip(first.micro_cells, second.micro_cells, strict=False):
        if first_cell == second_cell:
            continue

        first_ix, first_iy = first_cell % MICRO_GRID_N, first_cell // MICRO_GRID_N
        second_ix, second_iy = second_cell % MICRO_GRID_N, second_cell // MICRO_GRID_N
        assert abs(first_ix - second_ix) <= 1
        assert abs(first_iy - second_iy) <= 1
        return

    assert first.micro_cells == second.micro_cells


class TestMicroPSAAddress:
    """Test MicroPSAAddress dataclass."""

    def test_coarse_only_address(self):
        """Test address with no micro-cells."""
        addr = MicroPSAAddress(coarse_voxel="L1_v_9_9_12", micro_cells=[])

        assert addr.depth == 0
        assert addr.full_address == "L1_v_9_9_12"

    def test_depth_1_address(self):
        """Test address with one micro-cell level."""
        addr = MicroPSAAddress(coarse_voxel="L1_v_9_9_12", micro_cells=[47])

        assert addr.depth == 1
        assert addr.full_address == "L1_v_9_9_12.47"

    def test_depth_3_address(self):
        """Test address with max depth micro-cells."""
        addr = MicroPSAAddress(coarse_voxel="L1_v_9_9_12", micro_cells=[47, 23, 81])

        assert addr.depth == 3
        assert addr.full_address == "L1_v_9_9_12.47.23.81"


class TestVoxelBounds:
    """Test VoxelBounds and subdivision."""

    def test_basic_bounds(self):
        """Test basic bounds properties."""
        bounds = VoxelBounds(x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.1)

        assert bounds.width == 0.1
        assert bounds.height == 0.1

    def test_subdivide_corner(self):
        """Test subdivision into corner micro-cell."""
        bounds = VoxelBounds(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

        # Index 0 = bottom-left corner (ix=0, iy=0)
        sub = bounds.subdivide(0)
        assert sub.x_min == 0.0
        assert sub.x_max == 0.1
        assert sub.y_min == 0.0
        assert sub.y_max == 0.1

    def test_subdivide_center(self):
        """Test subdivision into center micro-cell."""
        bounds = VoxelBounds(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

        # Index 55 = center-ish (ix=5, iy=5)
        sub = bounds.subdivide(55)
        assert sub.x_min == pytest.approx(0.5)
        assert sub.x_max == pytest.approx(0.6)
        assert sub.y_min == pytest.approx(0.5)
        assert sub.y_max == pytest.approx(0.6)

    def test_subdivide_top_right(self):
        """Test subdivision into top-right corner."""
        bounds = VoxelBounds(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

        # Index 99 = top-right corner (ix=9, iy=9)
        sub = bounds.subdivide(99)
        assert sub.x_min == 0.9
        assert sub.x_max == 1.0
        assert sub.y_min == 0.9
        assert sub.y_max == 1.0


class TestMicroCellComputation:
    """Test micro-cell index computation."""

    @pytest.fixture
    def mapper(self):
        """Create PSAMapper with dummy star positions."""
        # Create 75 random 3D positions on unit sphere
        np.random.seed(42)
        positions = np.random.randn(75, 3)
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)

        mapper = PSAMapper()
        mapper.set_star_positions(positions)
        return mapper

    def test_corner_indices(self, mapper):
        """Test that corners map to expected indices."""
        bounds = VoxelBounds(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

        # Bottom-left corner → index 0
        idx = mapper.compute_micro_cell_index(0.05, 0.05, bounds)
        assert idx == 0

        # Top-right corner → index 99
        idx = mapper.compute_micro_cell_index(0.95, 0.95, bounds)
        assert idx == 99

    def test_monotonic_x(self, mapper):
        """Test monotonic behavior along x-axis."""
        bounds = VoxelBounds(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

        # Moving along x should increase index (within same row)
        prev_idx = -1
        for x in [0.05, 0.15, 0.25, 0.35, 0.45]:
            idx = mapper.compute_micro_cell_index(x, 0.05, bounds)
            assert idx > prev_idx or (idx == prev_idx)  # Monotonic
            prev_idx = idx

    def test_monotonic_y(self, mapper):
        """Test monotonic behavior along y-axis."""
        bounds = VoxelBounds(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)

        # Moving along y should increase index by 10 (row stride)
        prev_idx = -1
        for y in [0.05, 0.15, 0.25, 0.35, 0.45]:
            idx = mapper.compute_micro_cell_index(0.05, y, bounds)
            assert idx > prev_idx
            prev_idx = idx


class TestMonotonicRefinement:
    """Test that refinement is monotonic (close coords stay together)."""

    @pytest.fixture
    def mapper(self):
        """Create PSAMapper with dummy star positions."""
        np.random.seed(42)
        positions = np.random.randn(75, 3)
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)

        mapper = PSAMapper()
        mapper.set_star_positions(positions)
        return mapper

    def test_very_close_interior_coords_same_micro_cell(self, mapper):
        """Very close interior coordinates should land in the same micro-cell."""
        coord1 = PSACoordinate(
            x=0.12302,
            y=0.45602,
            z=0.789,
            ra=0,
            dec=0,
            depth=1.0,
            confidence="anchored",
            top_stars=[],
        )
        coord2 = PSACoordinate(
            x=0.12303,
            y=0.45603,
            z=0.7891,
            ra=0,
            dec=0,
            depth=1.0,
            confidence="anchored",
            top_stars=[],
        )

        addr1 = mapper.position_to_micro_psa(coord1)
        addr2 = mapper.position_to_micro_psa(coord2)

        # Should have same address at max depth
        assert addr1.full_address == addr2.full_address

    def test_near_micro_boundary_coords_stay_adjacent(self, mapper):
        """Close coordinates near a micro-boundary may split, but only adjacently."""
        coord1 = PSACoordinate(
            x=0.123, y=0.456, z=0.789, ra=0, dec=0, depth=1.0, confidence="anchored", top_stars=[]
        )
        coord2 = PSACoordinate(
            x=0.1231,
            y=0.4561,
            z=0.7891,
            ra=0,
            dec=0,
            depth=1.0,
            confidence="anchored",
            top_stars=[],
        )

        addr1 = mapper.position_to_micro_psa(coord1)
        addr2 = mapper.position_to_micro_psa(coord2)

        assert_hierarchical_addresses_same_or_adjacent(addr1, addr2)

    def test_close_coords_adjacent_or_same(self, mapper):
        """Close coordinates should be in same or adjacent micro-cells."""
        coord1 = PSACoordinate(
            x=0.10, y=0.20, z=0.30, ra=0, dec=0, depth=1.0, confidence="anchored", top_stars=[]
        )
        coord2 = PSACoordinate(
            x=0.11,
            y=0.21,
            z=0.31,  # Small difference
            ra=0,
            dec=0,
            depth=1.0,
            confidence="anchored",
            top_stars=[],
        )

        addr1 = mapper.position_to_micro_psa(coord1)
        addr2 = mapper.position_to_micro_psa(coord2)

        assert_hierarchical_addresses_same_or_adjacent(addr1, addr2)

        # At depth 1, micro-cells should be same or adjacent when the coarse
        # voxel is shared. Across a coarse boundary, the adjacent voxel is the
        # locality guarantee and micro-cell indices are in different frames.
        if addr1.micro_cells and addr2.micro_cells:
            if addr1.coarse_voxel != addr2.coarse_voxel:
                return
            mc1, mc2 = addr1.micro_cells[0], addr2.micro_cells[0]
            ix1, iy1 = mc1 % MICRO_GRID_N, mc1 // MICRO_GRID_N
            ix2, iy2 = mc2 % MICRO_GRID_N, mc2 // MICRO_GRID_N

            # Should be within 1 cell of each other
            assert abs(ix1 - ix2) <= 1
            assert abs(iy1 - iy2) <= 1


class TestConstants:
    """Test MicroCell PSA constants."""

    def test_grid_size(self):
        """Grid should be 10x10."""
        assert MICRO_GRID_N == 10

    def test_cell_count(self):
        """Cell count should be 100."""
        assert MICRO_CELL_COUNT == 100

    def test_max_depth(self):
        """Max depth should be 3."""
        assert MICRO_MAX_DEPTH == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
