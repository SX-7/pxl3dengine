import pyxel as px
from utils import Vec3, Camera, speed_test
import math


class App:
    def __init__(self) -> None:
        px.init(128, 128)
        self.camera_position = Vec3(0, 32, 0)
        self.camera_rotation = Vec3(math.pi / 2, 0, 0)
        self.display_surface = Vec3(
            px.width // 2, px.height // 2, math.sqrt(px.height * px.width) // 4
        )
        self.camera = Camera(
            self.camera_position, self.camera_rotation, self.display_surface
        )
        self.screen_position = []
        self.point_data = []
        for x in range(-32, 32):
            for y in range(-32, 32):
                self.point_data.append(
                    Vec3(x, y, (px.noise(x, y, 0.5) + 1) * 64)
                )

        px.run(self.update, self.draw)

    @speed_test
    def update(self):
        if px.btn(px.KEY_A):
            self.camera_position += Vec3(-1, 0, 0)
        if px.btn(px.KEY_D):
            self.camera_position += Vec3(1, 0, 0)
        if px.btn(px.KEY_W):
            self.camera_position += Vec3(0, -1, 0)
        if px.btn(px.KEY_S):
            self.camera_position += Vec3(0, 1, 0)
        if px.btn(px.KEY_R):
            self.camera_position += Vec3(0, 0, -1)
        if px.btn(px.KEY_F):
            self.camera_position += Vec3(0, 0, 1)
        if px.btn(px.KEY_LEFT):
            self.camera_rotation += Vec3(0, 0, -1) / 10
        if px.btn(px.KEY_RIGHT):
            self.camera_rotation += Vec3(0, 0, 1) / 10
        if px.btn(px.KEY_UP):
            self.camera_rotation += Vec3(1, 0, 0) / 10
        if px.btn(px.KEY_DOWN):
            self.camera_rotation += Vec3(-1, 0, 0) / 10
        if px.btn(px.KEY_Q):
            self.camera_rotation += Vec3(0, 1, 0) / 10
        if px.btn(px.KEY_E):
            self.camera_rotation += Vec3(0, -1, 0) / 10
        self.camera.update_camera(
            self.camera_position, self.camera_rotation, self.display_surface
        )
        self.screen_position = []
        for point in self.point_data:
            self.screen_position.append(self.camera.get(point))

    @speed_test
    def draw(self):
        px.cls(0)
        for point in self.screen_position:
            px.pset(point.x, point.y, 13)
        px.text(0, 0, f"X={round(self.camera_position.x,2)}", 2)
        px.text(0, 6, f"Y={round(self.camera_position.y,2)}", 2)
        px.text(0, 12, f"Z={round(self.camera_position.z,2)}", 2)
        px.text(0, 18, f"Rx={round(self.camera_rotation.x,2)}", 2)
        px.text(0, 24, f"Ry={round(self.camera_rotation.y,2)}", 2)
        px.text(0, 30, f"Rz={round(self.camera_rotation.z,2)}", 2)


App()
