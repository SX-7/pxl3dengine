import math
import functools
import time
import copy


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


class Camera:
    """Changes 3d coordinates to 2d ones.
    Input list of shapes (or a single shape) and parameters.
    Returns a list of shapes whose x and y coordinates of vertices can be used
    on a standard pyxel screen, with z values for what comes before/after.
    The list may be longer in case of spltting shapes into triangles, or
    shorter, in case of various face culling/removal techniques.

    CCW coordinates define visible faces
    """

    def __init__(self):
        pass

    def _clip_poly_to_nf_fustrum(self, poly, near, far):
        # fustrum is sadly in the shape of a cut pyramid
        # XY clipping is worthless, as Z impacts the cut plane
        # volume clipping is way too much work for this
        # instead, we're clipping twice, once XZ, and once YZ
        # (indeed one is redundant, but makes it easier for me)
        # to ensure absolutely no cuts, except for the Z, we're using max
        biggest_coord = 0
        for vec in poly:
            local_max = max(abs(vec.x), abs(vec.y), abs(vec.z), abs(vec.w))
            if local_max > biggest_coord:
                biggest_coord = local_max
        fustrum_xy_z = [
            Vec3(biggest_coord * 2, near, 1),
            Vec3(biggest_coord * 2, far, 1),
            Vec3(-biggest_coord * 2, far, 1),
            Vec3(-biggest_coord * 2, near, 1),
        ]
        # There's an edge case: if we have the shape perfectly aligned to one of the planes,
        # We might have issues
        # If it's the Z plane, we can just skip it
        # If it's X or Y planes, we need to select the other in advance
        if len(poly) < 3:
            raise IndexError
        Y = False
        normal = (poly[0] - poly[1]).cross(poly[2] - poly[1])
        if normal.y == 0:
            Y = True
        poly_xz = []
        if Y:
            for vec in poly:
                poly_xz.append(Vec4(vec.y, vec.z, vec.x, vec.w))
        else:
            for vec in poly:
                poly_xz.append(Vec4(vec.x, vec.z, vec.y, vec.w))
        # we're abusing 2 things here - preservation of order, and preservation of type
        # this funciton doesn't care what it'll get, as long as it's Vec with .x and .y
        # thus we can use types for signalization. Hacky asf, but works :)
        # should write some tests to ensure this functionality is preserved in the future
        clipped_xz = Vec2.shape_intersection(poly_xz, fustrum_xy_z)
        result = []
        try:
            for point in clipped_xz:
                if isinstance(point, Vec4):
                    if Y:
                        result.append(Vec4(point.z, point.x, point.y, point.w))
                    else:
                        result.append(Vec4(point.x, point.z, point.y, point.w))
                elif isinstance(point, Vec2):
                    if Y:
                        result.append(
                            Vec4(
                                Vec3.place_on_plane(
                                    point,
                                    poly[0],
                                    poly[1],
                                    poly[2],
                                    missing_coord="X",
                                ).x,
                                point.x,
                                point.y,
                                (point.y + 2 * near) * (far - 2 * near) / far,
                            )
                        )
                    else:
                        result.append(
                            Vec4(
                                point.x,
                                Vec3.place_on_plane(
                                    point,
                                    poly[0],
                                    poly[1],
                                    poly[2],
                                    missing_coord="Y",
                                ).y,
                                point.y,
                                (point.y + 2 * near) * (far - 2 * near) / far,
                            )
                        )
        except Exception as e:
            # if all on Z plane
            return []
        return result

    def _metadata_clip(self, subject, clip):
        output_list = copy.copy(subject)
        # Greiner-Hormann algo, for the pointers we're using .w
        # step one from wikipedia
        for clip_index in range(len(clip)):
            clip_begin = clip[clip_index]
            clip_end = clip[(clip_index + 1) % len(clip)]
            input_list = copy.copy(output_list)
            output_list = []
            for subject_index in range(len(input_list)):
                curr_point = input_list[subject_index]
                prev_point = input_list[(subject_index - 1) % len(input_list)]
                inter_point = Vec2.intersection_point(
                    clip_begin, clip_end, curr_point, prev_point
                )  # make one for infinite length
                # now we insert the data unto the inter_point
                # we assume that subject is Vec4
                if isinstance(inter_point, Vec2):
                    diff = curr_point - prev_point
                    try:
                        ratio = (inter_point.x - prev_point.x) / (
                            curr_point.x - prev_point.x
                        )
                    except:
                        ratio = (inter_point.y - prev_point.y) / (
                            curr_point.y - prev_point.y
                        )
                    inter_point = prev_point + diff * ratio
                if Vec2.ccw(
                    clip_begin,
                    clip_end,
                    clip[(clip_index + 2) % len(clip)],
                ) == Vec2.ccw(clip_begin, clip_end, curr_point):
                    if Vec2.ccw(
                        clip_begin,
                        clip_end,
                        clip[(clip_index + 2) % len(clip)],
                    ) != Vec2.ccw(clip_begin, clip_end, prev_point):
                        output_list.append(inter_point)
                    output_list.append(curr_point)
                elif Vec2.ccw(
                    clip_begin,
                    clip_end,
                    clip[(clip_index + 2) % len(clip)],
                ) == Vec2.ccw(clip_begin, clip_end, prev_point):
                    output_list.append(inter_point)
        return output_list

    def _clip_poly_sides_fustrum(self, poly, near, far, pov):
        # TODO: non-square aspect ratios
        xy_z_clip_coords = [
            Vec2(-2 * near, 0),
            Vec2(-far, far),
            Vec2(far, far),
            Vec2(2 * near, 0),
        ]
        swapped_xzyw = []
        for point in poly:
            swapped_xzyw.append(Vec4(point.x, point.z, point.y, point.w))
        clipped = self._metadata_clip(swapped_xzyw, xy_z_clip_coords)
        swapped_yzxw = []
        for point in clipped:
            swapped_yzxw.append(Vec4(point.z, point.y, point.x, point.w))
        clipped = self._metadata_clip(swapped_yzxw, xy_z_clip_coords)
        clipped_poly = []
        for point in clipped:
            clipped_poly.append(Vec4(point.z, point.x, point.y, point.w))
        return clipped_poly

    def get(
        self,
        shape: Shape,
        screen_width: int,
        screen_heigth: int,
        world_coordinates: Vec3 = Vec3(0, 0, 0),
        rotation: Vec4 = Vec4(0, 0, 0, 0),
        scaling: Vec3 = Vec3(1, 1, 1),
        camera_pos: Vec3 = Vec3(0, 0, 64),
        camera_front: Vec3 = Vec3(0, 0, 1),
        world_up: Vec3 = Vec3(0, 1, 0),
        pov: float = 100,
        near: float = 0.1,
        far: float = 100,
    ):
        # pre-pipeline shape transforms
        if shape.count < 3:
            raise Exception
        shape_4 = []
        for point in shape.vertices:
            shape_4.append(Vec4(point.x, point.y, point.z, 1))
        world_matrix = Mat4.scaling_matrix(scaling)
        world_matrix = rotation.rotation_matrix() * world_matrix
        world_matrix = (
            Mat4.translation_matrix(world_coordinates) * world_matrix
        )
        camera_direction = camera_front.normalize()
        right = world_up.cross(camera_direction).normalize()
        camera_up = camera_direction.cross(right)
        view_matrix = Mat4(
            [
                [right.x, right.y, right.z, 0],
                [camera_up.x, camera_up.y, camera_up.z, 0],
                [
                    camera_direction.x,
                    camera_direction.y,
                    camera_direction.z,
                    0,
                ],
                [0, 0, 0, 1],
            ]
        ) * Mat4.translation_matrix(-camera_pos)
        perspective_matrix = Mat4.perspective_matrix(
            math.radians(pov), screen_width / screen_heigth, near, far
        )
        clip_space = []
        for point in shape_4:
            point = world_matrix * point
            point = view_matrix * point
            point = perspective_matrix * point
            clip_space.append(point)
        # vertex processing - clipping
        clip_space = self._clip_poly_to_nf_fustrum(clip_space, near, far)
        # now we have clipped front-back. time to clip lrtb
        clip_space = self._clip_poly_sides_fustrum(clip_space, near, far, pov)
        result = []
        for point in clip_space:
            point.x = point.x / point.w
            point.y = point.y / point.w
            point.z = point.z / point.w
            point.w = 1
            point.y = -point.y
            point += Vec4(1, 1, 0, 0)
            point.x *= screen_width / 2
            point.y *= screen_heigth / 2
            result.append(point)

        if len(result) > 0:
            tris = Shape(result).decompose_to_triangles()
            result = []
            for triangle in tris:
                if Vec2.ccw(
                    triangle.vertices[0],
                    triangle.vertices[1],
                    triangle.vertices[2],
                ):
                    result.append(triangle)
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
