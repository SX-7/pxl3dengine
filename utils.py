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


class Vec4:
    pass


class Shape:
    pass


class Shape:
    def __init__(self, vertices: list[Vec4]) -> None:
        self.vertices: list[Vec4] = vertices
        self.count = len(vertices)

    def insert_vertice(self, vertice: Vec4, index: int) -> None:
        self.vertices.insert(index, vertice)
        self.count = len(self.vertices)

    def add_vertice(self, vertice: Vec4) -> None:
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
        return self.x==__value.x and self.y==__value.y

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
        '''Works for infinitely long lines'''
        return Vec2.ccw(A, C, D) != Vec2.ccw(B, C, D) and Vec2.ccw(
            A, B, C
        ) != Vec2.ccw(A, B, D)

    @staticmethod
    def intersection_point(A, B, C, D):
        '''Works for infinitely long lines'''
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


class Mat3:
    pass


class Vec3:
    def __init__(self, x: float | int, y: float | int, z: float | int) -> None:
        self.x: float = float(x)
        self.y: float = float(y)
        self.z: float = float(z)

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

    def place_on_plane(t_point: Vec2, A, B, C):
        normal = (A - B).cross(C - B)

        new_z = (
            (normal.x * (t_point.x - A.x) + normal.y * (t_point.y - A.y))
            / (-normal.z)
        ) + A.z
        return Vec4(t_point.x, t_point.y, new_z, 1)


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

    def __str__(self) -> str:
        return f"Vec4({self.x}, {self.y}, {self.z}, {self.w})"

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

    @staticmethod
    def compute_shape_intersection(
        subject_poly: list[Vec4], clip_poly: list[Vec4]
    ):
        """Requires CCW shapes"""
        import copy

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
                prev_point = input_list[
                    (subject_index - 1) % len(input_list)
                ]
                inter_point = Vec2.intersection_point(
                    clip_begin, clip_end, curr_point, prev_point
                )  # make one for infinite length

                if Vec2.ccw(clip_begin, clip_end, curr_point):
                    if Vec2.ccw(clip_begin, clip_end, prev_point):
                        output_list.append(inter_point)
                    output_list.append(curr_point)
                elif Vec2.ccw(clip_begin, clip_end, prev_point):
                    output_list.append(inter_point)

        return Shape(output_list)


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

    def get(
        self,
        shapes: list[Shape],
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
        # create object specific transforms
        world_matrix = Mat4.scaling_matrix(scaling)
        world_matrix = rotation.rotation_matrix() * world_matrix
        world_matrix = (
            Mat4.translation_matrix(world_coordinates) * world_matrix
        )
        # camera specific transforms
        camera_direction = camera_front.normalize()
        right = world_up.cross(camera_direction).normalize()
        camera_up = camera_direction.cross(right)
        # for some reason we're using a wrong system?
        # ideally, we'd like cartesian+screen x and y axes
        # to be 1:1 with pyxel cords. As such, a hack is needed.
        # Maybe an error in other part of the program, but this works
        # just as good
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
        # perspective transform
        perspective_matrix = Mat4.perspective_matrix(
            math.radians(pov), screen_width / screen_heigth, near, far
        )
        # we *DO NOT* want things bigger than triangles. Decompose, and then process
        good_shapes = []
        for shape in shapes:
            if shape.count > 3:
                good_shapes.extend(shape.decompose_to_triangles())
            else:
                good_shapes.append(shape)
        result = []
        for shape in good_shapes:
            behind = 0
            points = shape.vertices
            curr_shape_points = []
            view_points = []
            for point in points:
                # rn we're local space, objects have small values and all, basic
                # and unrotated, it's entirely possible to pre-do this step, but
                # that depends on the object. rn not skipping for completion sake
                point = world_matrix * point
                # world space, so the objects got into their position in the game
                point = view_matrix * point
                # camera/view space, we've moved the objects in front of our camera
                view_points.append(point)
            # now, we check for points behind the camera, and deal with offending vertices
            if len(view_points) == 1:
                if view_points[0].z > 0:
                    continue
            elif len(view_points) == 2:
                if (view_points[0].z) > 0 or (view_points[1].z > 0):
                    if (view_points[0].z > 0) and (view_points[1].z > 0):
                        continue
                    else:
                        if view_points[0].z > 0:
                            good = view_points[1]
                            bad = view_points[0]
                        else:
                            good = view_points[0]
                            bad = view_points[1]
                        diff = bad - good
                        # magic number, but actually necessary due to
                        # imperfections of floating point systems (:
                        # What was happening, is we'd get
                        diff = -((diff / diff.z) * good.z) * 0.99
                        bad = good + diff
                        view_points[0] = good
                        view_points[1] = bad
            elif len(view_points) == 3:
                which = []
                for i in range(len(view_points)):
                    if view_points[i].z > 0:
                        which.append(i)
                if len(which) == 3:
                    continue
                elif len(which) == 2:
                    which_not = 3 - sum(which)
                    good = view_points[which_not]
                    bad_one = view_points[which[0]]
                    bad_two = view_points[which[1]]
                    diff_one = bad_one - good
                    diff_two = bad_two - good
                    # magic number, but actually necessary due to
                    # imperfections of floating point systems (:
                    # What was happening, is we'd get
                    diff_one = -((diff_one / diff_one.z) * good.z) * 0.99
                    diff_two = -((diff_two / diff_two.z) * good.z) * 0.99
                    bad_one = good + diff_one
                    bad_two = good + diff_two
                    view_points[0] = good
                    view_points[1] = bad_one
                    view_points[2] = bad_two
                elif len(which) == 1:
                    match which[0]:
                        case 0:
                            good_one = view_points[1]
                            good_two = view_points[2]
                        case 1:
                            good_one = view_points[0]
                            good_two = view_points[2]
                        case 2:
                            good_one = view_points[0]
                            good_two = view_points[1]
                    bad_orig = view_points[which[0]]
                    diff_one = bad_orig - good_one
                    diff_two = bad_orig - good_two
                    # magic number, but actually necessary due to
                    # imperfections of floating point systems (:
                    # What was happening, is we'd get
                    diff_one = -((diff_one / diff_one.z) * good_one.z) * 0.99
                    diff_two = -((diff_two / diff_two.z) * good_two.z) * 0.99
                    bad_one = good_one + diff_one
                    bad_two = good_two + diff_two
                    view_points[0] = good_one
                    view_points[1] = bad_one
                    view_points[2] = bad_two
                    view_points.append(good_two)
                    # this case is problematic, since we have a triangle that's
                    # gonna get clipped in such a way, it'll have two good and
                    # two fake vertices. since we want only triangles, we will
                    # pack them into Shape and call the splitter
                    # ALSO don't forget about vertice order
            else:
                raise IndexError("Wtf??")
            for point in view_points:
                point = perspective_matrix * point
                # if (
                #    (-point.w < point.x < point.w)
                #    and (-point.w < point.y < point.w)
                #    and (-point.w < point.z < point.w)
                # ):
                try:
                    point = Vec4(
                        point.x / point.w,
                        point.y / point.w,
                        point.z / point.w,
                        1,
                    )
                    if point.z < 0 or point.z > 1:
                        behind += 1
                    # clip space, objects have been translated to -1 to 1 coordinates
                    # and clipped and perspective
                except:
                    # do something here later? currently less of a priority
                    continue
                point.y = -point.y
                point += Vec4(1, 1, 0, 0)
                point.x *= screen_width / 2
                point.y *= screen_heigth / 2
                # viewport transform, so basically we move to the 128x128 space, or
                # whatever the wievport is - this is pending to be incorporated
                # into the perspective matrix possibly? Depends on the calculation
                # savings
                curr_shape_points.append(point)
            if behind == len(
                curr_shape_points
            ):  # remember to add to close handling somewhere here later
                continue
            else:
                if len(curr_shape_points) > 3:
                    result.extend(
                        Shape(curr_shape_points).decompose_to_triangles()
                    )
                else:
                    result.append(Shape(curr_shape_points))
        # we only want to display shapes that are CCW towards us
        unclockwised_faces = result
        result = []
        for face in unclockwised_faces:
            # skip non lines
            if face.count < 3:
                result.append(face)
            elif face.count > 3:
                raise IndexError(
                    "This is right after decompose, this shouldn't break"
                )
            else:
                if Vec2.ccw(
                    Vec2(face.vertices[0].x, face.vertices[0].y),
                    Vec2(face.vertices[1].x, face.vertices[1].y),
                    Vec2(face.vertices[2].x, face.vertices[2].y),
                ):
                    result.append(face)
        # what we have, is shapes with possibly some insane values. Clip time
        unclipped_shapes = result
        result = []
        # heavy clipping which is necessary cuz otherwise pyxel dies, trying to draw stuff at 1e+18
        for shape in unclipped_shapes:
            # remove points, if for some reason they're still here
            if shape.count == 1:
                if (
                    0 < shape.vertices[0].x < screen_width
                    and 0 < shape.vertices[0].y < screen_heigth
                ):
                    result.append(shape)
                continue
            # shorten lines
            elif shape.count == 2:
                if (
                    0 < shape.vertices[0].x < screen_width
                    and 0 < shape.vertices[0].y < screen_heigth
                    and 0 < shape.vertices[1].x < screen_width
                    and 0 < shape.vertices[1].y < screen_heigth
                ):
                    result.append(shape)
                else:
                    append = False
                    if Vec2.intersect(
                        Vec2(shape.vertices[0].x, shape.vertices[0].y),
                        Vec2(shape.vertices[1].x, shape.vertices[1].y),
                        Vec2(0, 0),
                        Vec2(screen_width, 0),
                    ):
                        if shape.vertices[0].y <= 0:
                            diff = shape.vertices[0] - shape.vertices[1]
                            ratio = shape.vertices[1].y / diff.y
                            diff = diff * ratio
                            shape.vertices[0] = shape.vertices[1] - diff
                        else:
                            diff = shape.vertices[1] - shape.vertices[0]
                            ratio = shape.vertices[0].y / diff.y
                            diff = diff * ratio
                            shape.vertices[1] = shape.vertices[0] - diff
                        append = True
                    if Vec2.intersect(
                        Vec2(shape.vertices[0].x, shape.vertices[0].y),
                        Vec2(shape.vertices[1].x, shape.vertices[1].y),
                        Vec2(0, screen_heigth),
                        Vec2(screen_width, screen_heigth),
                    ):
                        if shape.vertices[0].y >= screen_heigth:
                            diff = shape.vertices[0] - shape.vertices[1]
                            ratio = (
                                screen_heigth - shape.vertices[1].y
                            ) / diff.y
                            diff = diff * ratio
                            shape.vertices[0] = shape.vertices[1] + diff
                        else:
                            diff = shape.vertices[1] - shape.vertices[0]
                            ratio = (
                                screen_heigth - shape.vertices[0].y
                            ) / diff.y
                            diff = diff * ratio
                            shape.vertices[1] = shape.vertices[0] + diff
                        append = True
                    if Vec2.intersect(
                        Vec2(shape.vertices[0].x, shape.vertices[0].y),
                        Vec2(shape.vertices[1].x, shape.vertices[1].y),
                        Vec2(0, 0),
                        Vec2(0, screen_heigth),
                    ):
                        if shape.vertices[0].x <= 0:
                            diff = shape.vertices[0] - shape.vertices[1]
                            ratio = shape.vertices[1].x / diff.x
                            diff = diff * ratio
                            shape.vertices[0] = shape.vertices[1] - diff
                        else:
                            diff = shape.vertices[1] - shape.vertices[0]
                            ratio = shape.vertices[0].x / diff.x
                            diff = diff * ratio
                            shape.vertices[1] = shape.vertices[0] - diff
                        append = True
                    if Vec2.intersect(
                        Vec2(shape.vertices[0].x, shape.vertices[0].y),
                        Vec2(shape.vertices[1].x, shape.vertices[1].y),
                        Vec2(screen_width, 0),
                        Vec2(screen_width, screen_heigth),
                    ):
                        if shape.vertices[0].x >= screen_width:
                            diff = shape.vertices[0] - shape.vertices[1]
                            ratio = (
                                screen_width - shape.vertices[1].x
                            ) / diff.x
                            diff = diff * ratio
                            shape.vertices[0] = shape.vertices[1] + diff
                        else:
                            diff = shape.vertices[1] - shape.vertices[0]
                            ratio = (
                                screen_width - shape.vertices[0].x
                            ) / diff.x
                            diff = diff * ratio
                            shape.vertices[1] = shape.vertices[0] + diff
                        append = True
                    if append:
                        result.append(shape)
            # clip triangles
            elif shape.count == 3:
                # my recommendation: look at clipping algos on wikipedia.
                # Also check out computational geometry, and painter's algorithm
                if (
                    0 < shape.vertices[0].x < screen_width
                    and 0 < shape.vertices[0].y < screen_heigth
                    and 0 < shape.vertices[1].x < screen_width
                    and 0 < shape.vertices[1].y < screen_heigth
                    and 0 < shape.vertices[2].x < screen_width
                    and 0 < shape.vertices[2].y < screen_heigth
                ):
                    result.append(shape)
                else:
                    new_shape = Shape([])
                    # find a plane of the triangle and place square vertices on it

                    viewport_vertices = [
                        Vec3.place_on_plane(
                            Vec2(0, 0),
                            shape.vertices[0],
                            shape.vertices[1],
                            shape.vertices[2],
                        ),
                        Vec3.place_on_plane(
                            Vec2(0, screen_heigth),
                            shape.vertices[0],
                            shape.vertices[1],
                            shape.vertices[2],
                        ),
                        Vec3.place_on_plane(
                            Vec2(screen_width, screen_heigth),
                            shape.vertices[0],
                            shape.vertices[1],
                            shape.vertices[2],
                        ),
                        Vec3.place_on_plane(
                            Vec2(screen_width, 0),
                            shape.vertices[0],
                            shape.vertices[1],
                            shape.vertices[2],
                        ),
                    ]

                    for vv in viewport_vertices:
                        vv = Vec4(vv.x, vv.y, vv.z, 1)
                    for vv in viewport_vertices:
                        print(vv)
                    new_shape = Vec4.compute_shape_intersection(
                        viewport_vertices, shape.vertices
                    )

                    try:
                        # consider putting triangle splitter from up there right below us, if we can handle polygons
                        triangles = new_shape.decompose_to_triangles()
                        # 6. Add the resulting list to results
                        result.extend(triangles)
                    except IndexError:
                        # that means that there was only 1 point remaining,
                        # so, it was on the border, or outside it, so goodbye
                        pass
            else:
                print(
                    "How did you get a 4+= sided shape in here. Test&report pls"
                )
                result.append(shape)
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
