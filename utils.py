import math
import functools
import time
import copy


def speed_test(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def _verify_type(compared_object: object, *types: type) -> int:
    """Throws a descriptive error when types of two objects don't align.
    Returns an int corrensponding to position of the type in `types` if
    they do, starting from 0. Remember that passing *tuple into `types`
    unpacks the tuple into the function.

    Args:
        compared_object (object): Object to check the type of
        *types (type): Types of the object to be checked against
    Raises:
        TypeError: Incorrect type
    Returns:
        position (int): Which one of the requested types to check the object is
    """
    if not isinstance(compared_object, types):
        raise TypeError(
            f"{compared_object.__class__} is not one of the supported types: {types}"  # noqa: E501
        )
    else:
        for i in range(len(types)):
            if isinstance(compared_object, types[i]):
                return i


class Vec3:
    pass


class Shape:
    pass


class Shape:
    def __init__(self, vertices: list[Vec3]) -> None:
        self.vertices: list[Vec3] = vertices
        self.count = len(vertices)

    def __eq__(self, __value: object) -> bool:
        return self.vertices == __value.vertices

    def insert_vertice(self, vertice: Vec3, index: int) -> None:
        self.vertices.insert(index, vertice)
        self.count = len(self.vertices)

    def add_vertice(self, vertice: Vec3) -> None:
        self.vertices.append(vertice)
        self.count = len(self.vertices)

    def del_vertice(self, index: int) -> None:
        self.vertices.pop(index)
        self.count = len(self.vertices)

    def decompose_to_triangles(self) -> list[Shape]:
        """Can behave unexpectedly if used on shapes bigger than
        triangles, whose vertices are not on the same plane.
        Vertice no.0 will be used as common point for all triangles

        Raises:
            IndexError: If the shape is smaller than 3 vertices

        Returns:
            list[Shape]: List of triangles
        """
        if self.count < 3:
            raise IndexError
        elif self.count == 3:
            return [self]
        else:
            shapes = []
            for vertice in range(1, self.count - 1):
                shapes.append(
                    Shape(
                        [
                            self.vertices[0],
                            self.vertices[vertice],
                            self.vertices[vertice + 1],
                        ]
                    )
                )
            return shapes

class DataShape:
    pass
class DataShape:
    def __init__(self, vertices: list[tuple[Vec3,dict]]) -> None:
        self.vertices: list[tuple[Vec3,dict]] = vertices
        self.count = len(vertices)

    def __eq__(self, __value: object) -> bool:
        return self.vertices == __value.vertices

    def insert_vertice(self, vertice: tuple[Vec3,dict], index: int) -> None:
        self.vertices.insert(index, vertice)
        self.count = len(self.vertices)

    def add_vertice(self, vertice: tuple[Vec3,dict]) -> None:
        self.vertices.append(vertice)
        self.count = len(self.vertices)

    def del_vertice(self, index: int) -> None:
        self.vertices.pop(index)
        self.count = len(self.vertices)

    def decompose_to_triangles(self) -> list[DataShape]:
        """Can behave unexpectedly if used on shapes bigger than
        triangles, whose vertices are not on the same plane.
        Vertice no.0 will be used as common point for all triangles

        Raises:
            IndexError: If the shape is smaller than 3 vertices

        Returns:
            list[DataShape]: List of triangles
        """
        if self.count < 3:
            raise IndexError
        elif self.count == 3:
            return [self]
        else:
            shapes = []
            for vertice in range(1, self.count - 1):
                shapes.append(
                    DataShape(
                        [
                            self.vertices[0],
                            self.vertices[vertice],
                            self.vertices[vertice + 1],
                        ]
                    )
                )
            return shapes

class Vec2:
    def __init__(self, x: float | int, y: float | int) -> None:
        self.x: float = float(x)
        self.y: float = float(y)

    def __eq__(self, __value: object) -> bool:
        return self.x == __value.x and self.y == __value.y

    def __str__(self) -> str:
        return f"Vec2({self.x}, {self.y})"

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        _verify_type(other, int, float)
        return Vec2(self.x * other, self.y * other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __neg__(self):
        return self.__mul__(-1)

    @staticmethod
    def ccw(A, B, C) -> bool:
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

    # https://gist.github.com/alexcpn/45c5026397e11751583891831cc36456
    @staticmethod
    def intersect(A, B, C, D):
        """Works for infinitely long lines"""
        return Vec2.ccw(A, C, D) != Vec2.ccw(B, C, D) and Vec2.ccw(
            A, B, C
        ) != Vec2.ccw(A, B, D)

    @staticmethod
    def intersection_point(A, B, C, D):
        """Works for infinitely long lines"""
        xdiff = Vec2(A.x - B.x, C.x - D.x)
        ydiff = Vec2(A.y - B.y, C.y - D.y)

        def det(a, b):
            return a.x * b.y - a.y * b.x

        div = det(xdiff, ydiff)
        if div == 0:
            return False

        d = Vec2(det(A, B), det(C, D))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return Vec2(x, y)

    @staticmethod
    def is_in_triangle(point, A, B, C):
        return not (
            (
                (not Vec2.ccw(point, A, B))
                or (not Vec2.ccw(point, B, C))
                or (not Vec2.ccw(point, C, A))
            )
            and (
                Vec2.ccw(point, A, B)
                or Vec2.ccw(point, B, C)
                or Vec2.ccw(point, C, A)
            )
        )

    @staticmethod
    def shape_intersection(subject_poly: list, clip_poly: list):
        output_list = copy.copy(subject_poly)
        # Greiner-Hormann algo, for the pointers we're using .w
        # step one from wikipedia
        for clip_index in range(len(clip_poly)):
            clip_begin = clip_poly[clip_index]
            clip_end = clip_poly[(clip_index + 1) % len(clip_poly)]
            input_list = copy.copy(output_list)
            output_list = []
            for subject_index in range(len(input_list)):
                curr_point = input_list[subject_index]
                prev_point = input_list[(subject_index - 1) % len(input_list)]
                inter_point = Vec2.intersection_point(
                    clip_begin, clip_end, curr_point, prev_point
                )  # make one for infinite length

                if Vec2.ccw(
                    clip_begin,
                    clip_end,
                    clip_poly[(clip_index + 2) % len(clip_poly)],
                ) == Vec2.ccw(clip_begin, clip_end, curr_point):
                    if Vec2.ccw(
                        clip_begin,
                        clip_end,
                        clip_poly[(clip_index + 2) % len(clip_poly)],
                    ) != Vec2.ccw(clip_begin, clip_end, prev_point):
                        output_list.append(inter_point)
                    output_list.append(curr_point)
                elif Vec2.ccw(
                    clip_begin,
                    clip_end,
                    clip_poly[(clip_index + 2) % len(clip_poly)],
                ) == Vec2.ccw(clip_begin, clip_end, prev_point):
                    output_list.append(inter_point)

        return output_list


class Mat3:
    pass


class Vec3:
    def __init__(self, x: float | int, y: float | int, z: float | int) -> None:
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

    def __eq__(self, __value: object) -> bool:
        return (
            self.x == __value.x and self.y == __value.y and self.z == __value.z
        )

    def __str__(self) -> str:
        return f"Vec3({self.x}, {self.y}, {self.z})"

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        _verify_type(other, int, float)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        self = Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
        return self

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __neg__(self):
        return self.__mul__(-1)

    def normalize(self):
        result = self
        try:
            scaler = math.sqrt(
                result.x * result.x + result.y * result.y + result.z * result.z
            )
            result.x /= scaler
            result.y /= scaler
            result.z /= scaler
        except ZeroDivisionError:
            return self
        return result

    def x_rotation(self) -> Mat3:
        return Mat3(
            [
                [1, 0, 0],
                [0, math.cos(self.x), math.sin(self.x)],
                [0, -math.sin(self.x), math.cos(self.x)],
            ]
        )

    def y_rotation(self) -> Mat3:
        return Mat3(
            [
                [math.cos(self.y), 0, -math.sin(self.y)],
                [0, 1, 0],
                [math.sin(self.y), 0, math.cos(self.y)],
            ]
        )

    def z_rotation(self) -> Mat3:
        return Mat3(
            [
                [math.cos(self.z), math.sin(self.z), 0],
                [-math.sin(self.z), math.cos(self.z), 0],
                [0, 0, 1],
            ]
        )

    def place_on_plane(t_point: Vec2, A, B, C, missing_coord="Z"):
        normal = (A - B).cross(C - B)
        match missing_coord:
            case "X":
                new_x = (
                    (
                        normal.y * (t_point.x - A.y)
                        + normal.z * (t_point.y - A.z)
                    )
                    / (-normal.x)
                ) + A.x
                return Vec3(new_x, t_point.x, t_point.y)
            case "Y":
                new_y = (
                    (
                        normal.z * (t_point.y - A.z)
                        + normal.x * (t_point.x - A.x)
                    )
                    / (-normal.y)
                ) + A.y
                return Vec3(t_point.x, new_y, t_point.y)
            case "Z":
                new_z = (
                    (
                        normal.x * (t_point.x - A.x)
                        + normal.y * (t_point.y - A.y)
                    )
                    / (-normal.z)
                ) + A.z
                return Vec3(t_point.x, t_point.y, new_z)


class Mat3:
    """List of lists.
    Structure is: [[row1_data][r2_d][r3_d]].
    First row is the upmost one, first cell in it is leftmost one.
    """

    def __init__(self, matrix_data: [[float], [float], [float]]) -> None:
        self.matrix = matrix_data

    def __mul__(self, other):
        choice = _verify_type(other, Mat3, int, float, Vec3)
        match choice:
            case 0:
                result = result = Mat3([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
                for x in range(3):
                    for y in range(3):
                        for order in range(3):
                            result.matrix[y][x] += (
                                self.matrix[y][order] * other.matrix[order][x]
                            )
                self = result
            case 1 | 2:
                for row in self.matrix:
                    for cell in row:
                        cell = cell * other
            case 3:
                return Vec3(
                    self.matrix[0][0] * other.x
                    + self.matrix[0][1] * other.y
                    + self.matrix[0][2] * other.z,
                    self.matrix[1][0] * other.x
                    + self.matrix[1][1] * other.y
                    + self.matrix[1][2] * other.z,
                    self.matrix[2][0] * other.x
                    + self.matrix[2][1] * other.y
                    + self.matrix[2][2] * other.z,
                )
            case _:
                raise IndexError
        return self


class Mat4:
    pass


class Vec4:
    def __init__(
        self, x: float | int, y: float | int, z: float | int, w: float | int
    ) -> None:
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)
        self.w: float = float(w)

    def __eq__(self, __value: object) -> bool:
        return (
            self.x == __value.x
            and self.y == __value.y
            and self.z == __value.z
            and self.w == __value.w
        )

    def __str__(self) -> str:
        return f"Vec4({self.x}, {self.y}, {self.z}, {self.w})"

    def __add__(self, other):
        return Vec4(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )

    # WHY DID I WRITE IT WRONG WAY BACK
    def __sub__(self, other):
        return Vec4(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )

    # IT'D SAVE ME SO MUCH TIME IF I FIXED IT !!PROPERLY!!
    def __mul__(self, other):
        _verify_type(other, int, float)
        return Vec4(
            self.x * other, self.y * other, self.z * other, self.w * other
        )

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __neg__(self):
        return self.__mul__(-1)

    def cross(self, other):
        t = Vec3.cross(self, other)
        return Vec4(t.x, t.y, t.z, 1)

    def rotation_matrix(self) -> Mat4:
        sin = math.sin(self.w)
        cos = math.cos(self.w)
        return Mat4(
            [
                [
                    cos + (self.x * self.x * (1 - cos)),
                    self.x * self.y * (1 - cos) - self.z * sin,
                    self.x * self.z * (1 - cos) + self.y * sin,
                    0,
                ],
                [
                    self.y * self.x * (1 - cos) + self.z * sin,
                    cos + (self.y * self.y * (1 - cos)),
                    self.y * self.z * (1 - cos) - self.x * sin,
                    0,
                ],
                [
                    self.z * self.x * (1 - cos) - self.y * sin,
                    self.z * self.y * (1 - cos) + self.x * sin,
                    cos + (self.z * self.z * (1 - cos)),
                    0,
                ],
                [0, 0, 0, 1],
            ]
        )

    def normalize(self):
        result = self
        try:
            scaler = math.sqrt(
                result.x * result.x + result.y * result.y + result.z * result.z
            )
            result.x /= scaler
            result.y /= scaler
            result.z /= scaler
        except ZeroDivisionError:
            return self
        return result


class Mat4:
    def __init__(
        self,
        matrix_data: [[float], [float], [float], [float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    ) -> None:
        self.matrix = matrix_data

    def __mul__(self, other):
        choice = _verify_type(other, Mat4, int, float, Vec4)
        match choice:
            case 0:
                result = Mat4(
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                )
                for x in range(4):
                    for y in range(4):
                        for order in range(4):
                            result.matrix[y][x] += (
                                self.matrix[y][order] * other.matrix[order][x]
                            )
                self = result
            case 1 | 2:
                for row in self.matrix:
                    for cell in row:
                        cell = cell * other
            case 3:
                return Vec4(
                    self.matrix[0][0] * other.x
                    + self.matrix[0][1] * other.y
                    + self.matrix[0][2] * other.z
                    + self.matrix[0][3] * other.w,
                    self.matrix[1][0] * other.x
                    + self.matrix[1][1] * other.y
                    + self.matrix[1][2] * other.z
                    + self.matrix[1][3] * other.w,
                    self.matrix[2][0] * other.x
                    + self.matrix[2][1] * other.y
                    + self.matrix[2][2] * other.z
                    + self.matrix[2][3] * other.w,
                    self.matrix[3][0] * other.x
                    + self.matrix[3][1] * other.y
                    + self.matrix[3][2] * other.z
                    + self.matrix[3][3] * other.w,
                )
            case _:
                raise IndexError
        return self

    def identity_matrix():
        return Mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def scaling_matrix(scaling_vector: Vec3):
        return Mat4(
            [
                [scaling_vector.x, 0, 0, 0],
                [0, scaling_vector.y, 0, 0],
                [0, 0, scaling_vector.z, 0],
                [0, 0, 0, 1],
            ]
        )

    def translation_matrix(translation_vector: Vec3):
        return Mat4(
            [
                [1, 0, 0, translation_vector.x],
                [0, 1, 0, translation_vector.y],
                [0, 0, 1, translation_vector.z],
                [0, 0, 0, 1],
            ]
        )

    # https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix.html # noqa
    def perspective_matrix(fov: float, aspect: float, near: float, far: float):
        nea = near
        top = math.tan(fov / 2) * nea
        bot = -top
        rig = top * aspect
        lef = bot * aspect
        return Mat4(
            [
                [(2 * near) / (rig - lef), 0, (rig + lef) / (rig - lef), 0],
                [0, (2 * nea) / (top - bot), (top + bot) / (top - bot), 0],
                [
                    0,
                    0,
                    -((far + nea) / (far - nea)),
                    -((2 * far * nea) / (far - nea)),
                ],
                [0, 0, -1, 0],
            ]
        )