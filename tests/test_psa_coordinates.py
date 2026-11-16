from consolidated_clustering_v2.routing.psa_coordinates import (
    VOXEL_DISTANCE_MODE,
    canonical_voxel_coords,
    parse_psa_voxel,
    voxel_distance_from_ids,
    voxel_level,
)


def test_multiresolution_voxels_share_canonical_frame():
    assert VOXEL_DISTANCE_MODE == "canonical_l1_normalized"
    assert canonical_voxel_coords("k:L1_v_10_10_10") == (10.0, 10.0, 10.0)
    assert canonical_voxel_coords("k:L2_v_20_20_20") == (10.0, 10.0, 10.0)
    assert canonical_voxel_coords("k:L3_v_40_40_40") == (10.0, 10.0, 10.0)

    assert voxel_distance_from_ids("k:L1_v_10_10_10", "k:L3_v_40_40_40") == 0.0


def test_fractional_canonical_coordinates_for_fine_layers():
    assert canonical_voxel_coords("k:L3_v_39_41_40") == (9.75, 10.25, 10.0)


def test_parse_level_and_raw_coordinates():
    namespaced_key = (
        "ns_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef:"
        "clustering:voxel:L2_v_19_20_21"
    )
    parsed = parse_psa_voxel(namespaced_key)

    assert parsed is not None
    assert parsed.level == 2
    assert parsed.raw == (19, 20, 21)
    assert parsed.canonical == (9.5, 10.0, 10.5)
    assert voxel_level(namespaced_key) == 2


def test_invalid_voxel_returns_none():
    assert parse_psa_voxel("not-a-voxel") is None
    assert canonical_voxel_coords("not-a-voxel") is None
    assert voxel_distance_from_ids("not-a-voxel", "k:L1_v_1_1_1") is None
