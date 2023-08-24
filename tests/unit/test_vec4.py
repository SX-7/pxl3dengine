import pytest
import logging
from utils import Vec4, Shape

logger = logging.getLogger("__Vec4__")
logger.setLevel(logging.INFO)


def test_compute_shape_intersection():
    clipping_polygon = [
        Vec4(0, 0, 1, 1),
        Vec4(0, 2, 1, 1),
        Vec4(2, 2, 1, 1),
        Vec4(2, 0, 1, 1),
    ]
    some_triangle = [Vec4(-1, -1, 1, 1), Vec4(-1, 3, 1, 1), Vec4(3, 3, 1, 1)]
    result = Vec4.compute_shape_intersection(
        some_triangle, clipping_polygon)
    assert result == Shape([Vec4(0, 0, 1, 1), Vec4(0, 2, 1, 1), Vec4(2, 2, 1, 1)])