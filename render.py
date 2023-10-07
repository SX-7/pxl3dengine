from utils import *
from typing import Callable


class Fragment:
    ...


def _clip():
    ...


def _rasterize(
    shape: DataShape, screen_width: int, screen_height: float
) -> list[Fragment]:
    ...


def default_fragment_shader(fragment: Fragment, fs_data: dict) -> Fragment:
    ...


# doing it as a func, gonnna swap to class (with some caching) later


def default_vertex_shader(point: Vec4, data: dict) -> tuple[Vec4, dict]:
    scaling: Vec3 = data["scaling"]
    rotation: Vec4 = data["rotation"]
    world_coordinates: Vec3 = data["world_coordinates"]
    camera_front: Vec3 = data["camera_front"]
    world_up: Vec3 = data["world_up"]
    camera_pos: Vec3 = data["camera_position"]
    fov: float = data["fov"]
    screen_width: float = data["screen_width"]
    screen_heigth: float = data["screen_heigth"]
    near: float = data["near"]
    far: float = data["far"]

    world_matrix = Mat4.scaling_matrix(scaling)
    world_matrix = rotation.rotation_matrix() * world_matrix
    world_matrix = Mat4.translation_matrix(world_coordinates) * world_matrix
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
        math.radians(fov), screen_width / screen_heigth, near, far
    )
    clip_space = world_matrix * point
    clip_space = view_matrix * clip_space
    clip_space = perspective_matrix * clip_space
    return clip_space, {}


def render(
    shapes: list[Shape],
    screen_width: float,
    screen_heigth: float,
    near: float = 0.1,
    far: float = 100,
    vertex_shader: Callable[
        [Vec4, dict], tuple[Vec4, dict]
    ] = default_vertex_shader,
    vertex_shader_args: list[list[dict]] = [
        [
            {
                "scaling": Vec3(1, 1, 1),
                "rotation": Vec4(0, 0, 0, 0),
                "world_coordinates": Vec3(0, 0, 0),
                "camera_front": Vec3(0, 0, 1),
                "world_up": Vec3(0, 1, 0),
                "camera_position": Vec3(0, 0, 64),
                "fov": 100,
                "screen_width": 256,
                "screen_heigth": 256,
                "near": 0.1,
                "far": 100,
            }
        ]
    ],
    fragment_shader: Callable[
        [Fragment, dict], tuple[float, int, int]
    ] = default_fragment_shader,
    fragment_shader_args: dict = {},
):
    """Gives back a list of pixels to show, given a shape list

    Args:
        shapes (list[Shape]): List of shapes to work on
        near (float): How far the near plane is
        far (float): How far the far plane is
        vertex_shader (function): Function taking in (Vec4, dict) and returning Vec4, dict. The dict returned will be interpolated across the fragments during the rasterization, and passed to fragment shader
        vertex_shader_args (list[list[dict]]): List of list of dicts, if there's not enough dicts in a sublist it'll be reused for that shape, if there's not enough sublists the last sublist will be reused till the end
        fragment_shader (function): Function taking in (Fragment, fs_in_args:dict) and returning float,int for Z depth and pyxel color value
        fragment_shader_args (dict): Since we can't predict the amount of fragments et al, we have one shared args - best for textures and the like
    """
    vs_shapes = []
    for i in range(shapes):
        shape = shapes[i]
        try:
            sublist = vertex_shader_args[i]
        except:
            sublist = vertex_shader_args[-1]
        new_vertices = []
        for j in range(shape.vertices):
            vertice = shape.vertices[j]
            if isinstance(vertice, Vec3):
                vertice = Vec4(vertice.x, vertice.y, vertice.z, 1)
            try:
                args = sublist[j]
            except:
                args = sublist[-1]
            new_vertice, new_args = vertex_shader(vertice, args)
            new_vertices.append((new_vertice, new_args))
        vs_shapes.append(new_vertices)
    # now we have transformed vertices, time to clip em, remember that points are actually tuples (point,args)
    clip_shapes = []
    for shape in vs_shapes:
        clip_shape = _clip(shape, near, far)
        if clip_shape:
            clip_shapes.append(clip_shape)
    # we now have a list of list of tuples, fyi
    # now time for transform, and face culling
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
            tris = DataShape(curr).decompose_to_triangles()
            for triangle in tris:
                if Vec2.ccw(
                    triangle.vertices[0][0],
                    triangle.vertices[1][0],
                    triangle.vertices[2][0],
                ):
                    result.append(triangle)
    # now we have "screenspace" points, with proper values in their tuples et al
    # fragments format? *likely* something like what's on the opengl wiki, remember to interp values
    fragments = []
    for tri in result:
        fragments.extend(_rasterize(tri, screen_width, screen_heigth))
    res_sheet = []
    for line in screen_heigth:
        res_sheet.append([])
    # currently doing depth test only rn, might move it higher laterrrr
    for fragment in fragments:
        curr_res = fragment_shader(fragment, fragment_shader_args)
        try:
            if res_sheet[fragment.y][fragment.x].z < curr_res.z:
                res_sheet[fragment.y][fragment.x] = curr_res
        except:
            res_sheet[fragment.y][fragment.x] = curr_res
    # we now have a sheet of fragments
    # converting it to colors
    for line in res_sheet:
        for fragment in line:
            fragment = fragment.color
    # ... rn we do be passing it to the user to do whatever they want with it
    # but, we could also import pyxel here and yeet it into it. rn let's just return a sheet of values
    return res_sheet
