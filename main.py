import pyxel as px
from utils import Vec3, Camera, Mat4, Vec4  # noqa


class App:
    def __init__(self) -> None:
        px.init(136, 145)
        self.camera_position = Vec3(0, 0, -30)
        self.camera = Camera()
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
        self.object_rotation = Vec4(0, 1, 0, 0)
        px.run(self.update, self.draw)

    # @speed_test
    def update(self):
        if px.btn(px.KEY_A):
            self.camera_position += Vec3(1, 0, 0)
        if px.btn(px.KEY_D):
            self.camera_position += Vec3(-1, 0, 0)
        if px.btn(px.KEY_W):
            self.camera_position += Vec3(0, 1, 0)
        if px.btn(px.KEY_S):
            self.camera_position += Vec3(0, -1, 0)
        if px.btn(px.KEY_R):
            self.camera_position += Vec3(0, 0, 1)
        if px.btn(px.KEY_F):
            self.camera_position += Vec3(0, 0, -1)
        self.object_rotation.w += 0.1
        self.screen_position = self.camera.get(
            self.point_data,
            camera_pos=self.camera_position,
            screen_heigth=px.height,
            screen_width=px.width,
            rotation=self.object_rotation,
        )

    # @speed_test
    def draw(self):
        px.cls(0)
        for point in self.screen_position:
            px.pset(point.x, point.y, 13)
        px.text(0, 0, f"X={round(self.camera_position.x,2)}", 2)
        px.text(0, 6, f"Y={round(self.camera_position.y,2)}", 2)
        px.text(0, 12, f"Z={round(self.camera_position.z,2)}", 2)


App()
