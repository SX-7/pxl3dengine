import logging
from utils import Vec4, Shape

logger = logging.getLogger("__Shape__")
logger.setLevel(logging.INFO)


def test_Shape___init__():
    vertices = [
        Vec4(1, 2, 3, 4),
        Vec4(3, 67, 8, 1),
        Vec4(0, 0, 0, 0),
        Vec4(9.999999999, 15.5, 1.1, 3),
    ]
    new_shape = Shape(vertices)
    assert new_shape.vertices == vertices
    assert new_shape.count == 4


def test_decompose_to_triangles():
    vertices = [
        Vec4(1, 2, 3, 4),
        Vec4(3, 67, 8, 1),
        Vec4(0, 0, 0, 0),
        Vec4(9.999999999, 15.5, 1.1, 3),
        Vec4(1, 1, 1, 1),
    ]
    shape = Shape(vertices)
    new_shapes = shape.decompose_to_triangles()
    expected_shapes = [
        Shape([Vec4(1, 2, 3, 4), Vec4(3, 67, 8, 1), Vec4(0, 0, 0, 0)]),
        Shape(
            [
                Vec4(1, 2, 3, 4),
                Vec4(0, 0, 0, 0),
                Vec4(9.999999999, 15.5, 1.1, 3),
            ]
        ),
        Shape(
            [
                Vec4(1, 2, 3, 4),
                Vec4(9.999999999, 15.5, 1.1, 3),
                Vec4(1, 1, 1, 1),
            ]
        ),
    ]
    assert new_shapes == expected_shapes
