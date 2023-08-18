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
                for x in range(3):
                    for y in range(3):
                        self.matrix[x][y] = (
                            self.matrix[x][y] * other.matrix[y][x]
                        )  # noqa
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


class Mat4:
    def __init__(
        self, matrix_data: [[float], [float], [float], [float]]
    ) -> None:
        self.matrix = matrix_data

    def __mul__(self, other):
        choice = _verify_type(other, Mat4, int, float, Vec4)
        match choice:
            case 0:
                for x in range(4):
                    for y in range(4):
                        self.matrix[x][y] = (
                            self.matrix[x][y] * other.matrix[y][x]
                        )  # noqa
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
    
    def translation_matrix(translation_vector: Vec3):
        pass


class Camera:
    """Changes 3d coordinates to 2d ones"""

    def __init__(
        self,
        camera_position: Vec3,
        camera_orientation: Vec3,
        display_surface: Vec3,
    ) -> None:
        self.update_camera(
            camera_position, camera_orientation, display_surface
        )

    def update_camera(
        self,
        camera_position: Vec3,
        camera_orientation: Vec3,
        display_surface: Vec3,
    ) -> None:
        self.camera_position = camera_position
        self.camera_orientation = camera_orientation
        self.display_surface = display_surface

    def get(self, point_position) -> Vec2:
        adjusted_point_position = self.camera_orientation.x_rotation() * (
            self.camera_orientation.y_rotation()
            * (
                self.camera_orientation.z_rotation()
                * (point_position - self.camera_position)
            )
        )
        if adjusted_point_position.z <= 0:
            return Vec2(-1, -1)
        try:
            return Vec2(
                self.display_surface.z
                / adjusted_point_position.z
                * adjusted_point_position.x
                + self.display_surface.x,
                self.display_surface.z
                / adjusted_point_position.z
                * adjusted_point_position.y
                + self.display_surface.y,
            )
        except ZeroDivisionError:
            return Vec2(-1, -1)
