import os
import json
from typing import List, Any, Union
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R

from .camera_pose.camera_pose import CameraPose

COLORS = [
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
]

ONE_DARK_COLORS = {
    "black": [40, 44, 52],
    "white": [171, 178, 191],
    "light red": [224, 108, 117],
    "dark red": [191, 97, 106],
    "green": [152, 195, 121],
    "light yellow": [229, 192, 123],
    "dark yellow": [209, 154, 102],
    "blue": [97, 175, 239],
    "magenta": [198, 120, 221],
    "cyan": [86, 182, 194]
}
# TODO: Add support for multiple visualizations
# TODO: Add support for different camera pose notations

def downsample_image_recursive(image:np.ndarray, max_size:int=640) -> np.ndarray:
    """Downsamples the image recursively until it fits the max size

    Args:
        image (np.ndarray): Image to downsample
        max_size (int, optional): Maximal allowed size of the image. Defaults to 640.

    Returns:
        np.ndarray: Downsampled image
    """    
    if image.shape[0] > max_size or image.shape[1] > max_size:
        image = cv2.resize(
            image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA
        )
        return downsample_image_recursive(image, max_size)
    return image

@dataclass
class PoseVisualizer:    
    pose_list: List[CameraPose] = field(default_factory=list)
    point_list: List[np.ndarray] = field(default_factory=list)
    pcd_list: List[o3d.geometry.PointCloud] = field(default_factory=list)

    coordinates: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    # coordinates: np.ndarray = np.zeros((3, 0))
    visualization: str = "axis"

    window_width: int = 1280
    window_height: int = 720

    visualizer = None

    @staticmethod
    def _create_axis_cross(
        Pose: np.ndarray, size: float = 0.1
    ) -> o3d.geometry.TriangleMesh:
        """Creates new axis cross at the given pose

        Args:
            Pose (np.ndarray): 4x4 transformation matrix
            size (float, optional): Size of the the axis vect. Defaults to 0.1.

        Returns:
            o3d.geometry.TriangleMesh: Axis cross as a triangle mesh
        """
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size, origin=np.zeros(3)
        )
        axis.rotate(R.from_matrix(Pose[:3, :3]).as_matrix(), center=np.zeros(3))
        axis.translate(Pose[:3, 3])
        return axis

    @staticmethod
    def _create_points(points: np.ndarray) -> o3d.geometry.PointCloud:
        """Creates a point cloud from the given points

        Args:
            points (np.ndarray): 3xN points

        Returns:
            o3d.geometry.PointCloud: Point cloud
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        return pcd

    @staticmethod
    def _create_3d_image(position: np.ndarray, image_path: str):
        """Creates triangle mesh in 3D space in form of rectangle

        Args:
            position (np.ndarray): 3D position of the image in space 4x3
            image_path (str): Path to the image

        Returns:
            _type_: _description_
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w, _ = image.shape

        image = downsample_image_recursive(image, max_size=360)


        map_image = o3d.t.geometry.Image(image)

        assert position.shape == (3, 4), "Position must be 3x4"

        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32

        triangle_mesh = o3d.t.geometry.TriangleMesh()
        triangle_mesh.vertex.positions = o3d.core.Tensor(position.T, dtype=dtype_f)

        triangle_mesh.triangle.indices = o3d.core.Tensor(
            [[0, 1, 3], [3, 2, 0]], dtype=dtype_i
        )
        triangle_mesh.vertex.texture_uvs = o3d.core.Tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=dtype_f
        )

        material = triangle_mesh.material
        material.material_name = "defaultLit"
        material.texture_maps["albedo"] = map_image

        # o3d.visualization.draw([triangle_mesh])
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(width=w, height=h)
        # vis.draw_geometries([triangle_mesh])

        return triangle_mesh

    @staticmethod
    def _create_point_cloud(
        K: np.ndarray,
        dist: np.ndarray,
        depth: np.ndarray,
        color: np.ndarray = None,
        T: np.ndarray = np.eye(4),
    ) -> o3d.geometry.PointCloud:
        """Creates a point cloud from the given depth map

        Args:
            K (np.ndarray): Intrinsics matrix 3x3
            dist (np.ndarray): Distortion coefficients
            depth (np.ndarray): Depth frame
            color (np.ndarray, optional): Color frame. Defaults to None.
            T (np.ndarray, optional): Camera pose. Defaults to np.eye(4).

        Returns:
            o3d.geometry.PointCloud: Point cloud
        """

        w, h = depth.shape
        depth_img = o3d.geometry.Image(depth)
        if color is not None:
            color_img = o3d.geometry.Image(color)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_img,
                depth_img,
                depth_scale=1.0,
                depth_trunc=100.0,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(
                    w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                ),
            )
        else:
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_img,
                o3d.camera.PinholeCameraIntrinsic(
                    w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                ),
            )
        pcd.transform(T)
        return pcd

    def _create_camera_pyramid(
        self, pose: CameraPose, size: float = 0.1, color: List[float] = [0, 0, 0]
    ) -> o3d.geometry.TriangleMesh:
        """Creates a camera pyramid at the given pose

        Args:
            Pose (np.ndarray): 4x4 transformation matrix
            size (float, optional): Size of the pyramid. Defaults to 0.1.
            aspect_ratio (float, optional): Aspect ratio of the camera. Defaults to 1.0.

        Returns:
            o3d.geometry.TriangleMesh: Camera pyramid as a triangle mesh
        """
        ar = pose.aspect_ratio
        s = size
        points = np.array(
            [
                [0, 0, 0, 1],
                [-ar * s / 2, -s / 2, s, 1],
                [-ar * s / 2, s / 2, s, 1],
                [ar * s / 2, -s / 2, s, 1],
                [ar * s / 2, s / 2, s, 1],
            ]
        ).T
        if pose.camera_notation == "opencv":
            points = pose.Tmx @ points
        elif pose.camera_notation == "opengl":
            points = pose.Tmx @ np.diag([1, -1, -1, 1]) @ points

        pyramid = o3d.geometry.LineSet()
        pyramid.points = o3d.utility.Vector3dVector(points[:3, :].T)
        pyramid.lines = o3d.utility.Vector2iVector(
            [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [4, 3], [3, 1]]
        )

        # Convert the color to 0-1 range from 0-255
        if max(color) > 1:
            color = [c / 255 for c in color]

        pyramid.paint_uniform_color(color)


        tl_point = points[:-1, 1]
        pcd = self._create_points(tl_point)
        pcd.paint_uniform_color(color)

        return pyramid, pcd

    def draw(self):
        # self.visualizer = o3d.visualization.Visualizer()
        # self.visualizer.create_window(
        #     width=self.window_width, height=self.window_height
        # )

        # Draw origin axis BASE
        org_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        # self.visualizer.add_geometry(org_axis)
        self.to_draw = [org_axis]

        # Draw camera poses
        for pose in self.pose_list:
            if pose.visualization == "both":
                axis = self._create_axis_cross(pose.Tmx)
                # self.visualizer.add_geometry(axis)
                self.to_draw.append(axis)
                
                pyramid, pcd = self._create_camera_pyramid(pose, color=pose.color)
                # self.visualizer.add_geometry(pyramid)
                self.to_draw.append(pyramid)

                # If there is an image path, draw the image
                if pose.image_path and os.path.exists(pose.image_path):
                    img_points = np.asarray(pyramid.points)[1:, :].T
                    image_3d = self._create_3d_image(img_points, pose.image_path)
                    # self.visualizer.add_geometry(image_3d)
                    self.to_draw.append(image_3d)

                else:
                    self.to_draw.append(pcd)
                    # self.visualizer.add_geometry(pcd)

            elif pose.visualization == "axes":
                axis = self._create_axis_cross(pose.Tmx)
                # self.visualizer.add_geometry(axis)
                self.to_draw.append(axis)

            elif pose.visualization == "pyramid":
                pyramid, pcd = self._create_camera_pyramid(pose)
                # self.visualizer.add_geometry(pyramid)
                # self.visualizer.add_geometry(pcd)
                pyramid, pcd = self._create_camera_pyramid(pose)
                # self.visualizer.add_geometry(pyramid)
                self.to_draw.append(pyramid)

                # If there is an image path, draw the image
                if pose.image_path and os.path.exists(pose.image_path):
                    img_points = np.asarray(pyramid.points)[1:, :].T
                    image_3d = self._create_3d_image(img_points, pose.image_path)
                    # self.visualizer.add_geometry(image_3d)
                    self.to_draw.append(image_3d)

                else:
                    self.to_draw.append(pcd)
                    # self.visualizer.add_geometry(pcd)

        # Draw points
        for point in self.point_list:
            pcd = self._create_points(point)
            # self.visualizer.add_geometry(pcd)
            self.to_draw.append(pcd)

        for pcd in self.pcd_list:
            # self.visualizer.add_geometry(pcd)
            self.to_draw.append(pcd)

        # self.visualizer.run()
        # self.visualizer.destroy_window()

        o3d.visualization.draw(self.to_draw)

    def add_camera(self, pose: CameraPose) -> None:
        self.pose_list.append(pose)

    def add_point(self, point: np.ndarray) -> None:
        self.point_list.append(point)

    def add_pointcloud(self, pcd: o3d.geometry.PointCloud) -> None:
        self.pcd_list.append(pcd)


def demo_from_depth():
    colors = [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    ]
    data_folder = os.path.join(os.path.dirname(__file__), "data", "crow_depth")
    pv = PoseVisualizer()
    for file in sorted(os.listdir(data_folder)):
        if not file.endswith(".jpg"):
            continue

        img = cv2.imread(os.path.join(data_folder, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(
            os.path.join(
                data_folder, file.replace(".jpg", ".png").replace("camera", "depth")
            ),
            cv2.IMREAD_UNCHANGED,
        )
        depth = depth.astype(np.float32) / 1000.0

        with open(os.path.join(data_folder, file.replace(".jpg", ".json"))) as f:
            camera_params = json.load(f)

        K = np.array(camera_params["K"])
        dist = np.array(camera_params["dist"])
        T_color2world = np.array(camera_params["T_color2world"])
        T_depth2color = np.array(camera_params["T_depth2color"])
        T = T_depth2color @ T_color2world

        pcd = pv._create_point_cloud(K, dist, depth, T=T)
        pcd.paint_uniform_color(colors[len(pv.pcd_list) % len(colors)])
        pv.add_pcd(pcd)
        pv.add_camera(
            CameraPose(T_color2world, aspect_ratio=1.0, visualization="pyramid")
        )

    pv.draw()


def demo_from_pcl():
    data_folder = os.path.join(os.path.dirname(__file__), "data", "crow_depth_params")
    pv = PoseVisualizer()
    for e, file in enumerate(sorted(os.listdir(data_folder))):
        if not file.endswith(".npy"):
            continue
        name = file.strip(".npy").strip("pcl")
        print(f"Processing {name}")

        params_names = f"camera{name}.json"
        with open(os.path.join(data_folder, params_names)) as f:
            camera_params = json.load(f)

        T_color2world = np.array(camera_params["T_color2world"])
        T_depth2color = np.array(camera_params["T_depth2color"])
        T = T_depth2color @ T_color2world
        # T = T_color2world

        pcl = np.load(os.path.join(data_folder, file))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.paint_uniform_color(COLORS[e % len(COLORS)])
        pcd.transform(T)
        pv.add_pcd(pcd)

        pv.add_camera(
            CameraPose(
                T_color2world,
                aspect_ratio=16/9,
                visualization="both",
                image_path=os.path.join(data_folder, f"camera{name}.jpg"),
                color=COLORS[e % len(COLORS)]
            )
        )

        print(f"Added {name}")


    pv.draw()

def manual_extrinsic():
    data_folder = os.path.join(os.path.dirname(__file__), "data", "crow_desk", "_box_to_showcase")
    pV = PoseVisualizer()

    extrinsic_file = "/home/testbed/Projects/camera_visualization/data/crow_desk/1/extrinsics.json"

    with open(extrinsic_file, "r") as f:
        extrinsics = json.load(f)
    
    new_extrinsics = {}
    new_base = np.array(extrinsics["2"])

    for key, value in extrinsics.items():
        Tmx = np.array(value)

        Tmx = new_base @ np.linalg.inv(Tmx)
        # print(key)
        # print(Tmx)   
        Tmx[:3, 3] = Tmx[:3, 3] * 0.001

        new_extrinsics[key] = Tmx.tolist()

        pV.add_camera(
            CameraPose(
                Tmx,
                aspect_ratio=16/9,
                visualization="both",
                # image_path=os.path.join(data_folder, f"camera{name}.jpg"),
                color=COLORS[int(key) % len(COLORS)]
            )
        )

    with open(extrinsic_file.replace(".json", "_new.json"), "w") as f:
        json.dump(new_extrinsics, f, indent=2)

    

    for file in sorted(os.listdir(data_folder)):
        if not file.endswith(".npy"):
            continue
        num = int(file.strip(".npy").strip("pcl"))
        print(f"Processing {num}")

        params_names = f"camera{num}.json"
        with open(os.path.join(data_folder, params_names)) as f:
            camera_params = json.load(f)

        T_color2world = np.array(camera_params["T_color2world"])
        T_depth2color = np.array(camera_params["T_depth2color"])

        pV.add_camera(
            CameraPose(
                T_color2world,
                aspect_ratio=16/9,
                visualization="both",
                # image_path=os.path.join(data_folder, f"camera{num}.jpg"),
                color=[c* .5 for c in COLORS[num % len(COLORS)]]
            )
        )

        pcl = np.load(os.path.join(data_folder, file))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.paint_uniform_color(COLORS[num % len(COLORS)])

        print(T_depth2color)
        print(T_color2world)
        T_depth2color[:3, 3] = T_depth2color[:3, 3] 

        T_ext = new_extrinsics[str(num)]
        T_ext = np.array(T_ext)

        T = T_ext @ T_depth2color
    
        pcd.transform(T)

        pV.add_pcd(pcd)

    pV.draw()

    




if __name__ == "__main__":
    cp = CameraPose()
    cp.name = "Camera 1"
    cp.Tmx = np.eye(4)
    cp.Tmx[:3, 3] = [0, 0, 1]
    cp.Tmx[:3, :3] = R.from_euler("ZYX", [0, 0, 180], degrees=True).as_matrix()
    cp.aspect_ratio = 1.0
    cp.units = "m"
    cp.visualization = "both"
    print(cp)

    pv = PoseVisualizer()
    pv.add_camera(cp)

    cp = CameraPose()
    cp.name = "Camera 2"
    cp.Tmx = np.eye(4)
    cp.Tmx[:3, 3] = [0, 0, -1]
    cp.aspect_ratio = 1.0
    cp.visualization = "both"
    pv.add_camera(cp)

    pv.draw()

    # demo_from_depth()
    # demo_from_pcl()

    # manual_extrinsic()
