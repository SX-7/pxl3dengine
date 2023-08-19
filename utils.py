import math
import functools
import time


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


class Vec2:
    def __init__(self, x: float | int, y: float | int) -> None:
        self.x: float = float(x)
        self.y: float = float(y)

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


class Mat3:
    pass


class Vec3:
    def __init__(self, x: float | int, y: float | int, z: float | int) -> None:
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

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
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __neg__(self):
        return self.__mul__(-1)

    def normalize(self):
        scaler = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.x /= scaler
        self.y /= scaler
        self.z /= scaler
        return self

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

    def __add__(self, other):
        return Vec4(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )

    def __sub__(self, other):
        return Vec4(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w + other.w,
        )

    def __mul__(self, other):
        _verify_type(other, int, float)
        return Vec4(
            self.x * other, self.y * other, self.z * other, self.w + other
        )

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __neg__(self):
        return self.__mul__(-1)

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
        scaler = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.x /= scaler
        self.y /= scaler
        self.z /= scaler
        return self


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

    def look_at(
        camera_position: Vec3, camera_orientation: Vec3, up_vector: Vec3
    ) -> Mat4:
        right = up_vector.cross(camera_orientation).normalize()
        camera_up = camera_orientation.cross(right).normalize()
        return Mat4(
            [
                [right.x, right.y, right.z, 0],
                [camera_up.x, camera_up.y, camera_up.z, 0],
                [
                    camera_orientation.x,
                    camera_orientation.y,
                    camera_orientation.z,
                    0,
                ],
                [0, 0, 0, 1],
            ]
        ) * Mat4([]).translation_matrix(
            Vec3(camera_position.x, camera_position.y, camera_position.z)
        )


class Camera:
    """Changes 3d coordinates to 2d ones"""

    def __init__(self):
        pass

    def get(
        self,
        entity: list[Vec4],
        screen_width: int,
        screen_heigth: int,
        world_coordinates: Vec3 = Vec3(0, 0, 0),
        rotation: Vec4 = Vec4(0, 0, 0, 0),
        scaling: Vec3 = Vec3(1, 1, 1),
        camera_pos: Vec3 = Vec3(64, 64, 0),
    ):
        world_matrix = Mat4.scaling_matrix(scaling)
        world_matrix = rotation.rotation_matrix() * world_matrix
        world_matrix = (
            Mat4.translation_matrix(world_coordinates) * world_matrix
        )

        view_matrix = Mat4.translation_matrix(camera_pos)

        perspective_matrix = Mat4.perspective_matrix(
            math.radians(100), screen_width / screen_heigth, 0.1, 100
        )

        result = []
        for point in entity:
            # (when I'm referring to objects, rn I mean pixels)
            # rn we're local space, objects have small values and all, basic
            # and unrotated, it's entirely possible to pre-do this step, but
            # that depends on the object. rn not skipping for completion sake
            point = world_matrix * point
            point = Vec4(
                point.x / point.w, point.y / point.w, point.z / point.w, 1
            )
            # world space, so the objects got into their position in the game
            point = view_matrix * point
            point = Vec4(
                point.x / point.w, point.y / point.w, point.z / point.w, 1
            )
            # camera/view space, we've moved the objects in front of our camera
            point = perspective_matrix * point
            if (
                (-point.w < point.x < point.w)
                and (-point.w < point.y < point.w)
                and (-point.w < point.z < point.w)
            ):
                point = Vec4(
                    point.x / point.w, point.y / point.w, point.z / point.w, 1
                )
            else:
                continue
            # clip space, objects have been translated to -1 to 1 coordinates
            # and clipped and perspective
            point += Vec4(1, 1, 0, 0)
            point.x *= screen_width/2
            point.y *= screen_heigth/2
            # viewport transform, so basically we move to the 128x128 space, or
            # whatever the wievport is
            result.append(point)
        return result


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
