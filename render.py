from utils import *


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

    def _clip_poly_sides_fustrum(self, poly, near, far):
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
        # vertex shader stage
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
        clip_space = self._clip_poly_sides_fustrum(clip_space, near, far)
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
        # vertex processing - face culling
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
        else:
            return []


# doing it as a func, gonnna swap to class (with some caching) later


def render(
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
    cam = Camera()
    result = []
    for shape in shapes:
        result.extend(
            cam.get(
                shape,
                screen_width,
                screen_heigth,
                world_coordinates,
                rotation,
                scaling,
                camera_pos,
                camera_front,
                world_up,
                pov,
                near,
                far,
            )
        )
    return result


def render(
    shapes: list[Shape],
    near: float,
    far: float,
    screen_width: float,
    screen_heigth: float,
    vertex_shader: function,
    vertex_shader_args: list[list[dict]],
    fragment_shader: function,
    fragment_shader_args:dict
):
    """Gives back a list of pixels to show, given a shape list

    Args:
        shapes (list[Shape]): List of shapes to work on
        near (float): How far the near plane is
        far (float): How far the far plane is
        vertex_shader (function): Function taking in (Vec4, dict) and returning Vec4, dict. The dict returned will be interpolated across the fragments during the rasterization, and passed to fragment shader
        vertex_shader_args (list[list[dict]]): List of list of dicts, if there's not enough dicts in a sublist it'll be reused for that shape, if there's not enough sublists the last sublist will be reused till the end
        fragment_shader (function): Function taking in (Fragment, dict) and returning float,int for Z depth and pyxel color value
        fragment_shader_args (dict): Since we can't predict the amount of fragments et al, we have one shared args - best for textures and the like
    """
    vs_shapes = []
    for i in range(shapes):
        shape = shapes[i]
        sublist = vertex_shader_args[i]
        new_vertices = []
        for j in range(shape.vertices):
            vertice = shape.vertices[j]
            args = sublist[j]
            new_vertice, new_args = vertex_shader(vertice,args)
            new_vertices.append((new_vertice,new_args))
        vs_shapes.append(new_vertices)
    # now we have transformed vertices, time to clip em, remember that points are actually tuples (point,args)
    clip_shapes = []
    for shape in vs_shapes:
        clip_shape = clip_poly_sides_fustrum(clip_poly_to_nf_fustrum(shape, near, far), near, far)
        if clip_shape:
            clip_shapes.append(clip_shape)
    # we now have a list of list of tuples, fyi
    result = []
    for shape in clip_shapes:
        curr = []
        for point in shape:
            point[0].x = point[0].x / point[0].w
            point[0].y = point[0].y / point[0].w
            point[0].z = point[0].z / point[0].w
            point[0].w = 1
            point[0].y = -point[0].y
            point[0] += Vec4(1, 1, 0, 0)
            point[0].x *= screen_width / 2
            point[0].y *= screen_heigth / 2
            curr.append(point)
        # vertex processing - face culling
        if len(curr) > 0:
            tris = Shape(curr).decompose_to_triangles()
            for triangle in tris:
                if Vec2.ccw(
                    triangle.vertices[0],
                    triangle.vertices[1],
                    triangle.vertices[2],
                ):
                    result.append(triangle)
    # now we have "screenspace" points, with proper values in their tuples et al
    # fragments format? *likely* something like what's on the opengl wiki, remember to interp values
    fragments = []
    for tri in result:
        fragments.extend(rasterize(tri,screen_heigth,screen_width))
    res_sheet=[]
    for line in screen_heigth:
        res_sheet.append([])
    # currently doing depth test only rn, might move it higher laterrrr
    for fragment in fragments:
        curr_res = fragment_shader(fragment,fragment_shader_args)
        try:
            if res_sheet[fragment.y][fragment.x].z<curr_res.z:
                res_sheet[fragment.y][fragment.x] = curr_res
        except:
            res_sheet[fragment.y][fragment.x] = curr_res
    # we now have a sheet of colors... rn we do be passing it to the user to do whatever they want with it
    # but, we could also import pyxel here and yeet it into it. rn let's just return a sheet of values
    return res_sheet