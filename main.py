import pyxel as px
from utils import Vec3, Camera, Vec2, Vec4, Shape
import math


class App:
    def __init__(self) -> None:
        # create screen
        px.init(256, 256)
        # set some info
        self.debug = True
        self.wireframe = True
        self.pov = 100
        self.camera_position = Vec3(0, 0, 30)
        self.camera = Camera()
        self.capture_mouse = True
        px.mouse(False)
        self.mouse_position_relative = Vec2(0, 0)
        self.camera_front = Vec3(0, 0, 1)
        self.camera_orientation_degrees = Vec2(90, 0)
        self.world_up = Vec3(0, 1, 0)
        self.screen_shapes: list[Shape] = []
        self.shape_data: list[Shape] = []
        self.shape_data.append(
            Shape(
                [
                    Vec3(-10, 10, 29),
                    Vec3(10, 10, 29),
                    Vec3(10, -10, 29),
                    Vec3(-10, -10, 29),
                ]
            )
        )
        self.shape_data.append(
            Shape(
                [
                    Vec3(-math.tan(math.radians(175/2))*100, -math.tan(math.radians(175/2))*100, -69.99),
                    Vec3(math.tan(math.radians(175/2))*100, -math.tan(math.radians(175/2))*100, -69.99),
                    Vec3(math.tan(math.radians(175/2))*100, math.tan(math.radians(175/2))*100, -69.99),
                    Vec3(-math.tan(math.radians(175/2))*100, math.tan(math.radians(175/2))*100, -69.99),
                ]
            )
        )
        # axis of rotation for the cube
        self.object_rotation = Vec4(0, 1, 0, 0)
        px.run(self.update, self.draw)

    # @speed_test
    def update(self):
        # rudimentary camera control
        self.movement_speed = 1
        if px.btn(px.KEY_A):
            self.camera_position += (
                self.camera_front.cross(self.world_up) * self.movement_speed
            )
        if px.btn(px.KEY_D):
            self.camera_position += (
                -self.camera_front.cross(self.world_up) * self.movement_speed
            )
        if px.btn(px.KEY_W):
            self.camera_position += -self.camera_front * self.movement_speed
        if px.btn(px.KEY_S):
            self.camera_position += self.camera_front * self.movement_speed
        if px.btn(px.KEY_SPACE):
            self.camera_position += Vec3(0, 1, 0) * self.movement_speed
        if px.btn(px.KEY_LSHIFT):
            self.camera_position += Vec3(0, -1, 0) * self.movement_speed
        if (px.mouse_wheel > 0) and (self.pov > 5):
            self.pov -= 5
        elif px.mouse_wheel < 0 and self.pov < 175:
            self.pov += 5
        elif px.btn(px.MOUSE_BUTTON_MIDDLE):
            self.pov = 100
        self.camera_position.x = round(self.camera_position.x, 3)
        self.camera_position.y = round(self.camera_position.y, 3)
        self.camera_position.z = round(self.camera_position.z, 3)
        # update object rotation to make it spin
        # self.object_rotation.w += 0.1
        # mouse movement processing code
        if self.capture_mouse:
            self.mouse_position_relative = Vec2(px.mouse_x, px.mouse_y) - Vec2(
                px.width // 2, px.height // 2
            )
            if px.btn(px.MOUSE_BUTTON_LEFT):
                # take screen aspect ratio into account?
                sensitivity = 0.02
                # we're currently only processing yaw+pitch, like in a FPS game
                self.camera_orientation_degrees.x += (
                    self.mouse_position_relative.x * sensitivity
                )
                self.camera_orientation_degrees.y += (
                    self.mouse_position_relative.y * sensitivity
                )

                if self.camera_orientation_degrees.y > 89:
                    self.camera_orientation_degrees.y = 89
                if self.camera_orientation_degrees.y < -89:
                    self.camera_orientation_degrees.y = -89

                direction = Vec3(
                    math.cos(math.radians(self.camera_orientation_degrees.y))
                    * math.cos(
                        math.radians(self.camera_orientation_degrees.x)
                    ),
                    math.sin(math.radians(self.camera_orientation_degrees.y)),
                    math.sin(math.radians(self.camera_orientation_degrees.x))
                    * math.cos(
                        math.radians(self.camera_orientation_degrees.y)
                    ),
                )
                self.camera_front = direction.normalize()
        else:
            self.mouse_position_relative = Vec2(0, 0)
        if px.btnp(px.KEY_BACKSLASH, 0, 10):
            if self.capture_mouse:
                self.capture_mouse = False
                px.mouse(True)
            else:
                self.capture_mouse = True
                px.mouse(False)
        if px.btnp(px.KEY_8, 0, 10):
            if self.wireframe:
                self.wireframe = False
            else:
                self.wireframe = True
        if px.btnp(px.KEY_9, 0, 10):
            if self.debug:
                self.debug = False
            else:
                self.debug = True
        # process pixel's positions
        self.screen_shapes = []
        for shape in self.shape_data:
            sh = self.camera.get(
                shape,
                camera_pos=self.camera_position,
                screen_heigth=px.height,
                screen_width=px.width,
                rotation=self.object_rotation,
                camera_front=self.camera_front,
                world_up=self.world_up,
                pov=self.pov,
            )
            if sh:
                self.screen_shapes.extend(sh)

    # @speed_test
    def draw(self):
        # clear screen
        px.cls(0)
        # draw pixels
        color = 0
        # self.screen_shapes.sort(key=lambda inp: -inp[0].vertices[0].z)
        for shape in self.screen_shapes:
            color += 1
            color %= 16
            if shape.count == 1:
                px.pset(shape.vertices[0].x, shape.vertices[0].y, color)
            elif shape.count == 2:
                px.line(
                    shape.vertices[0].x,
                    shape.vertices[0].y,
                    shape.vertices[1].x,
                    shape.vertices[1].y,
                    color,
                )
            else:
                if self.wireframe:
                    px.trib(
                        shape.vertices[0].x,
                        shape.vertices[0].y,
                        shape.vertices[1].x,
                        shape.vertices[1].y,
                        shape.vertices[2].x,
                        shape.vertices[2].y,
                        color,
                    )
                else:
                    px.tri(
                        shape.vertices[0].x,
                        shape.vertices[0].y,
                        shape.vertices[1].x,
                        shape.vertices[1].y,
                        shape.vertices[2].x,
                        shape.vertices[2].y,
                        color,
                    )
        # draw mouse pointer helper
        if self.capture_mouse:
            px.pset(px.width // 2, px.height // 2, 8)
            px.pset(
                px.width // 2 + self.mouse_position_relative.x // 2,
                px.height // 2 + self.mouse_position_relative.y // 2,
                9,
            )
            px.pset(
                px.mouse_x,
                px.mouse_y,
                10,
            )
        # some debug info
        if self.debug:
            px.text(0, 0, f"X={round(self.camera_position.x,2)}", 3)
            px.text(0, 6, f"Y={round(self.camera_position.y,2)}", 4)
            px.text(0, 12, f"Z={round(self.camera_position.z,2)}", 5)
            px.text(0, 18, f"Cm={self.capture_mouse}", 6)
            px.text(0, 24, f"Mx={px.mouse_x}", 7)
            px.text(0, 30, f"My={px.mouse_y}", 8)
            px.text(0, 36, f"Rx={self.mouse_position_relative.x}", 9)
            px.text(0, 42, f"Ry={self.mouse_position_relative.y}", 10)
            px.text(
                0, 48, f"Dx={round(self.camera_orientation_degrees.x,2)}", 11
            )
            px.text(
                0, 54, f"Ry={round(self.camera_orientation_degrees.y,2)}", 12
            )
            px.text(0, 60, f"Wf={self.wireframe}", 13)
            px.text(0, 66, f"Fo={self.pov}", 14)
            px.text(
                0, px.height - 6, "DEBUG=KEY_9 MOUSE=KEY_\\ WIREFRAME=KEY_8", 7
            )


App()
