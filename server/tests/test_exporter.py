"""World-size computation for the Unity mesh.

Regression guard: a georeferenced tile in geographic CRS (EPSG:4326) must have
its degree spans converted to meters, or the mesh renders as a collapsed
near-1D strip (e.g. a 0.2x0.2 unit floor under ~2000 m of elevation).
"""
import math

import pytest

from app.pipeline.exporter import compute_world_dimensions


def test_geographic_dims_convert_degrees_to_meters():
    # hilly_uttarakhand: 720x720, EPSG:4326, ~30.45N, 0.2 degree span
    bounds = [77.99986111114526, 30.35013888888496, 78.1998611111453, 30.550138888884987]
    w = 720
    cell_size = (bounds[2] - bounds[0]) / w
    world_width, world_depth = compute_world_dimensions(
        w, w, cell_size, bounds=bounds, crs="EPSG:4326")

    center_lat = math.radians((bounds[1] + bounds[3]) / 2.0)
    assert world_width == pytest.approx(
        (bounds[2] - bounds[0]) * 111320.0 * math.cos(center_lat))
    assert world_depth == pytest.approx((bounds[3] - bounds[1]) * 111320.0)

    # order of magnitude in meters, not degree-fraction units
    assert 15000.0 < world_width < 25000.0
    assert 15000.0 < world_depth < 25000.0
    # correct aspect: a 0.2x0.2 degree tile is wider-than-deep by cos(lat)
    assert world_width / world_depth == pytest.approx(math.cos(center_lat))


def test_projected_crs_unchanged():
    # EPSG:32633 metric bounds, 40 px at 30 m/px -> 1200 x 1200 m
    bounds = [500000.0, 4648800.0, 501200.0, 4650000.0]
    w = 40
    cell_size = (bounds[2] - bounds[0]) / w
    world_width, world_depth = compute_world_dimensions(
        w, w, cell_size, bounds=bounds, crs="EPSG:32633")
    assert world_width == 1200.0
    assert world_depth == 1200.0


def test_non_georeferenced_fallback():
    assert compute_world_dimensions(100, 200, None) == (1000.0, 1000.0)