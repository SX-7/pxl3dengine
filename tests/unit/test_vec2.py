import pytest
import logging
from utils import Vec2, Vec3, Vec4

logger = logging.getLogger("__Vec4__")
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
    assert Vec2.intersection_point(A,B,C,D) == Vec2(0,5)
