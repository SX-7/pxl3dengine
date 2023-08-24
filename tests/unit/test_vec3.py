import pytest
import logging
from utils import Vec3, Vec2

logger = logging.getLogger("__Vec4__")
logger.setLevel(logging.INFO)


def test_place_on_plane():
    plane_point_a = Vec3(0, 0, 0)
    plane_point_b = Vec3(10, 10, 5)
    plane_point_c = Vec3(-10, 10, 5)
    test_point = Vec2(0, 10)
    assert Vec3.place_on_plane(
        test_point, plane_point_a, plane_point_b, plane_point_c
    ) == Vec3(0, 10, 5)
    test_point = Vec2(0, 5)
    assert Vec3.place_on_plane(
        test_point, plane_point_a, plane_point_b, plane_point_c
    ) == Vec3(0, 5, 2.5)
    plane_point_c = Vec3(-10, 10, 10)
    test_point = Vec2(5, 5)
    assert Vec3.place_on_plane(
        test_point, plane_point_a, plane_point_b, plane_point_c
    ) == Vec3(5, 5, 2.5)
    test_point = Vec2(-10, -10)
    assert Vec3.place_on_plane(
        test_point, plane_point_a, plane_point_b, plane_point_c
    ) == Vec3(-10, -10, -5)