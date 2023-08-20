import pyxel as px
from utils import Vec3, Camera, Vec2, Vec4  # noqa
import math


class App:
    def __init__(self) -> None:
        # create screen
        px.init(136, 145)
        # set some info
        self.camera_position = Vec3(0, 0, 30)
        self.camera = Camera()
        self.capture_mouse = True
        px.mouse(False)
        self.mouse_position_relative = Vec2(0, 0)
        self.camera_front = Vec3(0, 0, 1)
        self.camera_orientation_degrees = Vec2(90, 0)
        self.world_up = Vec3(0, 1, 0)
        # create "wireframe"
        self.screen_position = []
        self.point_data = []
        for i in range(-10, 10):
            self.point_data.append(Vec4(i, 10, 10, 1))
            self.point_data.append(Vec4(i, -10, 10, 1))
            self.point_data.append(Vec4(i, -10, -10, 1))
            self.point_data.append(Vec4(i, 10, -10, 1))
            self.point_data.append(Vec4(10, i, 10, 1))
            self.point_data.append(Vec4(-10, i, 10, 1))
            self.point_data.append(Vec4(-10, i, -10, 1))
            self.point_data.append(Vec4(10, i, -10, 1))
            self.point_data.append(Vec4(10, 10, i, 1))
            self.point_data.append(Vec4(-10, 10, i, 1))
            self.point_data.append(Vec4(-10, -10, i, 1))
            self.point_data.append(Vec4(10, -10, i, 1))
            # self.point_data.append(Vec4(0, 0, i+10, 1))
            # self.point_data.append(Vec4(0, i+15, 0, 1))
            # self.point_data.append(Vec4(i+20, 0, 0, 1))
        # axis of rotation for the cube
        self.object_rotation = Vec4(0, 1, 0, 0)
        px.run(self.update, self.draw)

    # @speed_test
    def update(self):
        # rudimentary camera control
        self.movement_speed = 1
        if px.btn(px.KEY_A):
            self.camera_position += self.camera_front.cross(self.world_up)
        if px.btn(px.KEY_D):
            self.camera_position += -self.camera_front.cross(self.world_up)
        if px.btn(px.KEY_W):
            self.camera_position += -self.camera_front * self.movement_speed
        if px.btn(px.KEY_S):
            self.camera_position += self.camera_front * self.movement_speed
        if px.btn(px.KEY_SPACE):
            self.camera_position += Vec3(0, 1, 0)
        if px.btn(px.KEY_LSHIFT):
            self.camera_position += Vec3(0, -1, 0)
        # update object rotation to make it spin
        self.object_rotation.w += 0.1
        # process pixel's positions
        self.screen_position = self.camera.get(
            self.point_data,
            camera_pos=self.camera_position,
            screen_heigth=px.height,
            screen_width=px.width,
            rotation=self.object_rotation,
            camera_front=self.camera_front,
            world_up=self.world_up,
        )
        if self.capture_mouse:
            self.mouse_position_relative = Vec2(px.mouse_x, px.mouse_y) - Vec2(
                px.width // 2, px.height // 2
            )
            if px.btn(px.MOUSE_BUTTON_LEFT):
                # take screen aspect ratio into account?
                sensitivity = 0.1
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

    # @speed_test
    def draw(self):
        px.cls(0)
        for point in self.screen_position:
            px.pset(point.x, point.y, 13)
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
        px.text(0, 0, f"X={round(self.camera_position.x,2)}", 3)
        px.text(0, 6, f"Y={round(self.camera_position.y,2)}", 4)
        px.text(0, 12, f"Z={round(self.camera_position.z,2)}", 5)
        px.text(0, 18, f"Cm={self.capture_mouse}", 6)
        px.text(0, 24, f"Mx={px.mouse_x}", 7)
        px.text(0, 30, f"My={px.mouse_y}", 8)
        px.text(0, 36, f"Rx={self.mouse_position_relative.x}", 9)
        px.text(0, 42, f"Ry={self.mouse_position_relative.y}", 10)
        px.text(0, 48, f"Dx={round(self.camera_orientation_degrees.x,2)}", 11)
        px.text(0, 56, f"Ry={round(self.camera_orientation_degrees.y,2)}", 12)


App()
