import logging
from utils import Vec4

logger = logging.getLogger("__Vec4__")
logger.setLevel(logging.INFO)


def test_shape_intersection():
    """Due to shape intersection being possibly run on ccw and/or cw
    coordinates (cuz y-flipping), both need to work"""
    clip_poly = [
        Vec4(-10, -10, 1, 1),
        Vec4(10, -10, 1, 1),
        Vec4(10, 10, 1, 1),
        Vec4(-10, 10, 1, 1),
    ]
    subject_poly = [
        Vec4(-20, 10, 1, 1),
        Vec4(-20, 35, 1, 1),
        Vec4(30, -15, 1, 1),
    ]
    result = Vec4.shape_intersection(subject_poly, clip_poly)
    # order? Any
    assert (
        result
        == [
            Vec4(-10, 5, 1, 1),
            Vec4(-10, 10, 1, 1),
            Vec4(5, 10, 1, 1),
            Vec4(10, 5, 1, 1),
            Vec4(10, -5, 1, 1),
        ]
        or result
        == [
            Vec4(10, -5, 1, 1),
            Vec4(-10, 5, 1, 1),
            Vec4(-10, 10, 1, 1),
            Vec4(5, 10, 1, 1),
            Vec4(10, 5, 1, 1),
        ]
        or result
        == [
            Vec4(10, 5, 1, 1),
            Vec4(10, -5, 1, 1),
            Vec4(-10, 5, 1, 1),
            Vec4(-10, 10, 1, 1),
            Vec4(5, 10, 1, 1),
        ]
        or result
        == [
            Vec4(5, 10, 1, 1),
            Vec4(10, 5, 1, 1),
            Vec4(10, -5, 1, 1),
            Vec4(-10, 5, 1, 1),
            Vec4(-10, 10, 1, 1),
        ]
        or result
        == [
            Vec4(-10, 10, 1, 1),
            Vec4(5, 10, 1, 1),
            Vec4(10, 5, 1, 1),
            Vec4(10, -5, 1, 1),
            Vec4(-10, 5, 1, 1),
        ]
        or result
        == [
            Vec4(10, -5, 1, 1),
            Vec4(10, 5, 1, 1),
            Vec4(5, 10, 1, 1),
            Vec4(-10, 10, 1, 1),
            Vec4(-10, 5, 1, 1),
        ]
        or result
        == [
            Vec4(-10, 5, 1, 1),
            Vec4(10, -5, 1, 1),
            Vec4(10, 5, 1, 1),
            Vec4(5, 10, 1, 1),
            Vec4(-10, 10, 1, 1),
        ]
        or result
        == [
            Vec4(-10, 10, 1, 1),
            Vec4(-10, 5, 1, 1),
            Vec4(10, -5, 1, 1),
            Vec4(10, 5, 1, 1),
            Vec4(5, 10, 1, 1),
        ]
        or result
        == [
            Vec4(5, 10, 1, 1),
            Vec4(-10, 10, 1, 1),
            Vec4(-10, 5, 1, 1),
            Vec4(10, -5, 1, 1),
            Vec4(10, 5, 1, 1),
        ]
        or result
        == [
            Vec4(10, 5, 1, 1),
            Vec4(5, 10, 1, 1),
            Vec4(-10, 10, 1, 1),
            Vec4(-10, 5, 1, 1),
            Vec4(10, -5, 1, 1),
        ]
    )
