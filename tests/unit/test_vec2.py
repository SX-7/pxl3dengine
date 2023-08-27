import logging
from utils import Vec2, Vec3, Vec4

logger = logging.getLogger("__Vec2__")
logger.setLevel(logging.INFO)


def test_ccw():
    A = Vec2(-10, 10)
    B = Vec2(10, -10)
    ccw = Vec2(10, 10)
    cw = Vec2(-10, -10)
    assert Vec2.ccw(A, B, ccw)
    assert not Vec2.ccw(A, B, cw)


def test_ccw_compatibility():
    A = Vec3(-10, 10, 1)
    B = Vec3(10, -10, 5)
    ccw = Vec3(10, 10, 25)
    cw = Vec3(-10, -10, 125)
    assert Vec2.ccw(A, B, ccw)
    assert not Vec2.ccw(A, B, cw)
    A = Vec4(-10, 10, 1, 1)
    B = Vec4(10, -10, 2, 2)
    ccw = Vec4(10, 10, 3, 3)
    cw = Vec4(-10, -10, 4, 4)
    assert Vec2.ccw(A, B, ccw)
    assert not Vec2.ccw(A, B, cw)


def test_intersect():
    A = Vec2(0, 10)
    B = Vec2(0, 0)
    C = Vec2(-5, 5)
    D = Vec2(5, 5)
    assert Vec2.intersect(A, B, C, D)
    assert not Vec2.intersect(A, D, B, C)
    assert not Vec2.intersect(A,B,Vec2(5,10),Vec2(5,0))


def test_intersection_point():
    A = Vec2(0, 10)
    B = Vec2(0, 0)
    C = Vec2(-5, 5)
    D = Vec2(5, 5)
    # overlap
    assert Vec2.intersection_point(A,B,C,D) == Vec2(0,5)
    # edge
    assert Vec2.intersection_point(A,B,A,Vec2(10,10)) == A
    # outside the edge
    assert Vec2.intersection_point(A,B,C,Vec2(-2,5)) == Vec2(0,5)
    # double extensions
    assert Vec2.intersection_point(A,Vec2(0,8),C,Vec2(-2,5)) == Vec2(0,5)
    assert not Vec2.intersection_point(A,B,A,B) 
    assert not Vec2.intersection_point(A,A,A,A)
    assert not Vec2.intersection_point(A,A,A,B)
    
def test_is_in_triangle():
    A = Vec2(0, 10)
    B = Vec2(0, 0)
    C = Vec2(-5, 5)
    D = Vec2(5, 5)
    assert not Vec2.is_in_triangle(D,A,B,C)
    assert Vec2.is_in_triangle(A,A,B,C)
    assert Vec2.is_in_triangle(Vec2(-2,5),A,B,C)
    
def test_shape_intersection():
    """Due to shape intersection being possibly run on ccw and/or cw
    coordinates (cuz y-flipping), both need to work"""
    clip_poly = [
        Vec2(-10, -10),
        Vec2(10, -10),
        Vec2(10, 10),
        Vec2(-10, 10),
    ]
    subject_poly = [
        Vec2(-20, 10),
        Vec2(-20, 35),
        Vec2(30, -15),
    ]
    result = Vec2.shape_intersection(subject_poly, clip_poly)
    # order? Any
    assert (
        result
        == [
            Vec2(-10, 5),
            Vec2(-10, 10),
            Vec2(5, 10),
            Vec2(10, 5),
            Vec2(10, -5),
        ]
        or result
        == [
            Vec2(10, -5),
            Vec2(-10, 5),
            Vec2(-10, 10),
            Vec2(5, 10),
            Vec2(10, 5),
        ]
        or result
        == [
            Vec2(10, 5),
            Vec2(10, -5),
            Vec2(-10, 5),
            Vec2(-10, 10),
            Vec2(5, 10),
        ]
        or result
        == [
            Vec2(5, 10),
            Vec2(10, 5),
            Vec2(10, -5),
            Vec2(-10, 5),
            Vec2(-10, 10),
        ]
        or result
        == [
            Vec2(-10, 10),
            Vec2(5, 10),
            Vec2(10, 5),
            Vec2(10, -5),
            Vec2(-10, 5),
        ]
        or result
        == [
            Vec2(10, -5),
            Vec2(10, 5),
            Vec2(5, 10),
            Vec2(-10, 10),
            Vec2(-10, 5),
        ]
        or result
        == [
            Vec2(-10, 5),
            Vec2(10, -5),
            Vec2(10, 5),
            Vec2(5, 10),
            Vec2(-10, 10),
        ]
        or result
        == [
            Vec2(-10, 10),
            Vec2(-10, 5),
            Vec2(10, -5),
            Vec2(10, 5),
            Vec2(5, 10),
        ]
        or result
        == [
            Vec2(5, 10),
            Vec2(-10, 10),
            Vec2(-10, 5),
            Vec2(10, -5),
            Vec2(10, 5),
        ]
        or result
        == [
            Vec2(10, 5),
            Vec2(5, 10),
            Vec2(-10, 10),
            Vec2(-10, 5),
            Vec2(10, -5),
        ]
    )
